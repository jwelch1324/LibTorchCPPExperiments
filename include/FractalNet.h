#ifndef FRACTALNET_H_
#define FRACTALNET_H_

#include <torch/torch.h>
#include <memory>
#include <iostream>
#include <random>

using namespace torch;

struct ConvolutionalNodeOptions
{
    ConvolutionalNodeOptions(
        ExpandingArray<3> input_shape,
        uint32_t filter_channels,
        ExpandingArray<2> kernel_size,
        ExpandingArray<2> stride = ExpandingArray<2>({1,1}),
        ExpandingArray<2> dilation = ExpandingArray<2>({1,1})
    ) : input_shape_(std::move(input_shape)), filter_channels_(filter_channels),kernel_size_(std::move(kernel_size)),stride_(std::move(stride)),dilation_(std::move(dilation))
    {}

    TORCH_ARG(ExpandingArray<3>, input_shape);
    TORCH_ARG(uint32_t, filter_channels);
    TORCH_ARG(ExpandingArray<2>, kernel_size);
    TORCH_ARG(ExpandingArray<2>, stride);
    TORCH_ARG(ExpandingArray<2>, dilation);
};

class ConvolutionalNodeImpl : public torch::nn::Cloneable<ConvolutionalNodeImpl>
{
    public:
    ConvolutionalNodeImpl(
        ExpandingArray<3> input_shape,
        uint32_t filter_channels,
        ExpandingArray<2> kernel_size,
        ExpandingArray<2> stride = ExpandingArray<2>({1,1}),
        ExpandingArray<2> dilation = ExpandingArray<2>({1,1})
    ) : ConvolutionalNodeImpl(ConvolutionalNodeOptions(input_shape,filter_channels,kernel_size,stride,dilation)) {}

    ConvolutionalNodeImpl(const ConvolutionalNodeOptions& options_) : m_options(options_)
    {
        //We want these convolutions to leave the H and W dimensions unchanged
        //So we need to compute appropriate padding
        auto hpadding = compute_padding(
            (*m_options.input_shape())[1],
            (*m_options.stride())[0],
            (*m_options.dilation())[0],
            (*m_options.kernel_size())[0]
        );

        auto wpadding = compute_padding(
            (*m_options.input_shape())[2],
            (*m_options.stride())[1],
            (*m_options.dilation())[1],
            (*m_options.kernel_size())[1]
        );

        m_conv = register_module(
            "convolution_layer",
            nn::Conv2d(
                nn::Conv2dOptions(
                    m_options.filter_channels(),
                    m_options.filter_channels(),
                    m_options.kernel_size()
                )
                .stride(m_options.stride())
                .dilation(m_options.dilation())
                .padding({hpadding,wpadding})
            )
        );

        m_bn = register_module(
            "batch_norm",
            nn::BatchNorm2d(
                nn::BatchNorm2dOptions(
                    m_options.filter_channels()
                )
            )
        );

        m_act = register_module(
            "leaky_relu",
            nn::LeakyReLU(
                nn::LeakyReLUOptions()
            )
        );
    }

    int32_t compute_padding(int hin, int s0, int d0, int k0)
    {
        return(int32_t(floor(((hin-1)*s0+1+d0*(k0-1)-hin)/2.0)));
    }

    Tensor forward(const Tensor& input)
    {
        return(m_act(m_bn(m_conv(input))));
    }

    void reset() override
    {
    }

    void pretty_print(std::ostream& stream) const override
    {
        stream << "ConvolutionalNode(input_shape = " << m_options.input_shape()
                << ", filter_channels = " << m_options.filter_channels()
                << ", kernel_size = " << m_options.kernel_size()
                << ", stride = " << m_options.stride()
                << ", dilation = " << m_options.dilation()
                << ")";
    }

    ConvolutionalNodeOptions m_options;
    nn::Conv2d m_conv{nullptr};
    nn::BatchNorm2d m_bn{nullptr};
    nn::LeakyReLU m_act{nullptr};

};

TORCH_MODULE(ConvolutionalNode);

struct NodeJoinLayerOptions
{
    NodeJoinLayerOptions() {}

    //The Dropout Probability
    TORCH_ARG(float, local_drop_prob) = 0.5;
};

class NodeJoinLayerImpl : public nn::Cloneable<NodeJoinLayerImpl>
{
public:
    NodeJoinLayerImpl(
        float local_drop_prob
    ) : NodeJoinLayerImpl(NodeJoinLayerOptions().local_drop_prob(local_drop_prob))
    {}

    NodeJoinLayerImpl(const NodeJoinLayerOptions& options_ ) : m_options(options_)
    {}

    Tensor forward(const Tensor& tensors)
    {
        //The tensors object will be stacked as follows [S, B, F, H ,W] where S in the number of tensors we are taking the mean over

        if(is_training())
        {
            //Use Dropout to randomly drop some of the input tensors before taking the join
            auto idx = nn::functional::dropout(
                torch::ones({tensors.size(0)}),
                nn::functional::DropoutFuncOptions().p(m_options.local_drop_prob()).training(is_training())
                ).to(kBool);
            while(!(any(idx == true).item<bool>()))
            {
                idx = nn::functional::dropout(
                torch::ones({tensors.size(0)}),
                nn::functional::DropoutFuncOptions().p(m_options.local_drop_prob()).training(is_training())
                ).to(kBool);
            }
            return tensors.index(idx.to(tensors.device())).mean(0);
        }
        else
        {
            return tensors.mean(0);
        }
    }

    void reset() override
    {}

    NodeJoinLayerOptions m_options;
};

TORCH_MODULE(NodeJoinLayer);

template <typename NodeTypeOptions>
struct FractalNetworkOptions
{
    FractalNetworkOptions(
        ExpandingArray<3> input_shape, //Channel first shape
        std::vector<uint32_t> block_filter_channels, //if size=1 then the same filter is used for all blocks
        const NodeTypeOptions node_init_options //An options structure that will be passed to each node to init
    ): input_shape_(input_shape), block_filter_channels_(block_filter_channels), node_init_options_(node_init_options)
    {}

        //Channel first shape
        TORCH_ARG(ExpandingArray<3>, input_shape); 
        //if size=1 then the same filter is used for all blocks
        TORCH_ARG(std::vector<uint32_t>, block_filter_channels); 
        //Number of fractal blocks in the network
        TORCH_ARG(uint32_t, num_blocks) = 5;
        //Number of fractal recursions per block
        TORCH_ARG(uint32_t, block_depth) = 4;
        //An options structure that will be passed to each node to init
        TORCH_ARG(NodeTypeOptions, node_init_options);
        //The probability of executing a global drop path operation
        TORCH_ARG(float, global_path_dropout_prob) = 0.5;
};

template<typename NodeTypeOptions>
struct FractalBlockOptions
{
    FractalBlockOptions(
        uint32_t depth,
        const NodeTypeOptions node_init_options
    ) : node_init_options_(node_init_options), depth_(depth)
    {}

    //An options structure that will be passed to each node and subblock
    TORCH_ARG(NodeTypeOptions, node_init_options);
    //Number of fractal recursions per block
    TORCH_ARG(uint32_t, depth);
    TORCH_ARG(float, local_drop_prob) = 0.5;
};

//Forward Declare the Block class so we can reference it recursively in the impl below
template<typename NT, typename NTO> class FractalBlock;

template<typename NodeType, typename NodeTypeOptions>
class FractalBlockImpl : public nn::Cloneable<FractalBlockImpl<NodeType,NodeTypeOptions>>
{
public:
    using BlockType = FractalBlock<NodeType, NodeTypeOptions>;
    FractalBlockImpl(
        uint32_t depth,
        const NodeTypeOptions node_init_options,
        float local_drop_prob = 0.5
    ):FractalBlockImpl(FractalBlockOptions(depth,node_init_options).local_drop_prob(local_drop_prob))
    {}

    FractalBlockImpl(const FractalBlockOptions<NodeTypeOptions> options_) : m_options(options_)
    {
        //We setup the left branch here but the right branches are setup by the parent fractal network so that we don't get incomplete type errors from GCC
        m_left_branch = this->register_module(
            "fractal_node",
            NodeType(m_options.node_init_options())
        );

        m_join_layer = this->register_module(
            "join_layer",
            NodeJoinLayer(
                NodeJoinLayerOptions().local_drop_prob(m_options.local_drop_prob())
            )
        );
    }

    void reset() override
    {

    }

    Tensor forward(const Tensor& z, int32_t global_path_diff = -1, bool return_unjoined_stack = false)
    {
        if(global_path_diff >= 0)
        {
            if(global_path_diff == int32_t(m_options.depth()))
            {
                //Left branches are always of NodeType and as such do not get passed in the global_diff_path argument
                return(m_left_branch->forward(z));
            }
            else if(global_path_diff < int32_t(m_options.depth()))
            {
                //We are not yet at the lowest level, we need to evaluate the underlying right side branches
                return(
                    m_right_branch_lower->get()->forward(
                        m_right_branch_upper->get()->forward(
                            z,
                            global_path_diff),
                        global_path_diff
                    )
                );
            }
            else
            {
                std::cerr << "Unknown Global Path Diff Relation! We should not be seeing this!!!" <<std::endl;
                assert(false); //We should not get here!
            }
        }
        else
        {
            //No global path drop is specified
            Tensor out_left = m_left_branch->forward(z);
            if(m_options.depth() > 1)
            {
                Tensor out_right = m_right_branch_lower->get()->forward(m_right_branch_upper->get()->forward(z),-1,true); //We have deeper layers -- asked for an unjoined stack so we can join higher up

                if(return_unjoined_stack) //This is to enable joins across subnetwork outputs
                {
                    if(out_right.sizes().size() > out_left.sizes().size())
                    {
                        return(cat({out_left.unsqueeze(0),out_right}));
                    }
                    else
                    {
                        return(stack({out_left,out_right}));
                    }
                }
                else
                {
                    return(
                        m_join_layer->forward(
                            (out_right.sizes().size() == out_left.sizes().size()) ? stack({out_left,out_right}) : cat({out_left.unsqueeze(0),out_right})
                        )
                    );
                }
            }
            else
            {
                return out_left;
            }
        }
        assert(0);
    }

    FractalBlockOptions<NodeTypeOptions> m_options;
    
    NodeType m_left_branch{nullptr};
    BlockType* m_right_branch_lower{nullptr};
    BlockType* m_right_branch_upper{nullptr};
    NodeJoinLayer m_join_layer{nullptr};
};

//We have to explicitly build the Module wrapper here because the 
//TORCH_MODULE macro does not support template classes with multiple template arguments
template<typename BlockNodeType, typename BlockNodeTypeOptions>
class FractalBlock : public torch::nn::ModuleHolder<FractalBlockImpl<BlockNodeType,BlockNodeTypeOptions>> {
   public:                                                         
    using torch::nn::ModuleHolder<FractalBlockImpl<BlockNodeType,BlockNodeTypeOptions>>::ModuleHolder;             
  };


template <typename BlockNodeTypeOptions>
struct FractalBlockGroupOptions
{
    FractalBlockGroupOptions(
        uint32_t input_channels,
        uint32_t output_channels,
        const FractalBlockOptions<BlockNodeTypeOptions> options
    ): input_channels_(input_channels), output_channels_(output_channels), block_options_(options)
    {}

    TORCH_ARG(uint32_t, input_channels);
    TORCH_ARG(uint32_t, output_channels);
    TORCH_ARG(FractalBlockOptions<BlockNodeTypeOptions>, block_options);
};

template <typename BlockNodeType, typename BlockNodeTypeOptions>
class FractalBlockGroupImpl : public nn::Cloneable<FractalBlockGroupImpl<BlockNodeType,BlockNodeTypeOptions>>
{
public:
    FractalBlockGroupImpl(const FractalBlockGroupOptions<BlockNodeTypeOptions>& options_) : m_options(options_)
    {
        //Each block group is composed of a fractal block, an avg pool and a filter match if req.
        m_fractalBlock = this->register_module(
            "fractal_block", 
            FractalBlock<BlockNodeType,BlockNodeTypeOptions>(m_options.block_options()));

        //Recursively Build the subblocks
        this->build_subblocks(m_fractalBlock,m_options.block_options());
        

        //Register the avg pool layer
        m_avgPool = this->register_module(
            "avgpool2d",
            nn::AvgPool2d(
                nn::AvgPool2dOptions({2,2}).ceil_mode(true)
            )
        );

        //If the filter channels are going to change between blocks then we need a new filter match layer
        if(m_options.input_channels() != m_options.output_channels())
        {
            m_filter_match = this->register_module(
                "filter_match",
                nn::Conv2d(
                        nn::Conv2dOptions(
                            m_options.input_channels(),
                            m_options.output_channels(),
                            {1,1}
                        ).stride(1).bias(false)
                )
            );
        }
    }

    void build_subblocks(FractalBlock<BlockNodeType,BlockNodeTypeOptions>& block, FractalBlockOptions<BlockNodeTypeOptions> block_options)
    {
        if(block_options.depth() > 1)
        {
            std::stringstream ss;
            //This is somehwat of a hack to allocate the pointers to the sub fractal blocks in this network. 
            //We cannot allocate them in the actual class constructor since they refer to the class type itself and hence would require definition of the ctor before the ctor was defined
            m_fractalBlock->m_right_branch_upper = new FractalBlock<BlockNodeType,BlockNodeTypeOptions>(FractalBlockOptions(1,block_options.node_init_options()));
            auto right_branch_options = FractalBlockOptions<BlockNodeTypeOptions>(
                                            block_options.depth()-1,
                                            block_options.node_init_options()
                                        );//.local_drop_prob(block_options.local_drop_prob);
            
            ss << "right_branch_upper_" << block_options.depth()<<"_"<<torch::randint(1000,{1}).item<int>();
            (*m_fractalBlock->m_right_branch_upper) = m_fractalBlock->register_module(
                ss.str(),
                FractalBlock<BlockNodeType,BlockNodeTypeOptions>(
                    right_branch_options
                )
            );
            //Now recursively call build_subblocks on this branch
            build_subblocks((*m_fractalBlock->m_right_branch_upper),right_branch_options);

            m_fractalBlock->m_right_branch_lower = new FractalBlock<BlockNodeType,BlockNodeTypeOptions>(FractalBlockOptions(1,block_options.node_init_options()));
            ss.clear();
            ss << "right_branch_lower_" << block_options.depth()<<"_"<<torch::randint(1000,{1}).item<int>();
            (*m_fractalBlock->m_right_branch_lower) = m_fractalBlock->register_module(
                ss.str(),
                FractalBlock<BlockNodeType,BlockNodeTypeOptions>(
                    right_branch_options
                )
            );

            //Now recursively call build_subblocks on this branch
            build_subblocks((*m_fractalBlock->m_right_branch_lower),right_branch_options);
        }
    }

    void reset() override
    {}

    Tensor forward(const Tensor& input,int32_t global_path_diff = -1)
    {
        auto z = m_fractalBlock->forward(input,global_path_diff);
        z = m_avgPool->forward(z);
        if(!m_filter_match.is_empty())
        {
            z = m_filter_match->forward(z);
        }
        return z;
    }

    FractalBlockGroupOptions<BlockNodeTypeOptions> m_options;

    FractalBlock<BlockNodeType,BlockNodeTypeOptions> m_fractalBlock{nullptr};
    nn::AvgPool2d m_avgPool{nullptr};
    nn::Conv2d m_filter_match{nullptr};
};

template<typename BlockNodeType, typename BlockNodeTypeOptions>
class FractalBlockGroup : public torch::nn::ModuleHolder<FractalBlockGroupImpl<BlockNodeType,BlockNodeTypeOptions>> {
   public:                                                         
    using torch::nn::ModuleHolder<FractalBlockGroupImpl<BlockNodeType,BlockNodeTypeOptions>>::ModuleHolder;             
  };

template <typename FractalNodeType, typename FractalNodeTypeOptions>
class FractalNetworkImpl : public nn::Cloneable<FractalNetworkImpl<FractalNodeType,FractalNodeTypeOptions>>
{
    
public:
    FractalNetworkImpl(
        ExpandingArray<3> input_shape, //Channel first shape
        std::vector<uint32_t> block_filter_channels, //if size=1 then the same filter is used for all blocks
        uint32_t num_blocks, //Number of fractal blocks in the network
        uint32_t block_depth, //Number of fractal recursions per block
        const FractalNodeTypeOptions node_init_options, //An options structure that will be passed to each node to init
        float global_path_dropout_prob = 0.5 //The probability of executing a global drop path operation
    ):FractalNetworkImpl(FractalNetworkOptions<FractalNodeTypeOptions>(
            input_shape,
            block_filter_channels,
            node_init_options
        ).num_blocks(num_blocks).block_depth(block_depth).global_path_dropout_prob(global_path_dropout_prob)
    )
    {}

    FractalNetworkImpl(const FractalNetworkOptions<FractalNodeTypeOptions>& options_):m_options(options_)
    {
        if(m_options.block_filter_channels().size() == 1)
        {
            //Fill the block filter channels vector with the value that was passed in
            auto fc = m_options.block_filter_channels()[0];
            for(uint32_t j = 0; j < m_options.num_blocks() - 1; j++)
            {
                m_options.block_filter_channels().push_back(fc);
            }
        }

        if(m_options.block_filter_channels().size() != m_options.num_blocks())
        {
            std::cerr << "Error: the number of block channel filter values specified does not match the number of blocks specified!" <<std::endl;
            assert(0);
        }


        //We use a simple 2dConv layer to match the filter bank channels
        //Note -- explicit use of the 'this' pointer below forces the register_module call to be treated as a dependent name which moves the lookup to phase 2 of the template instantiation
        //        without this GCC will complain that the register_module call does not depend on template parameters and hence must be resolveable outside the template scope. 
        m_filter_match = this->register_module(
            "filter_match_layer",
            nn::Conv2d(
                nn::Conv2dOptions(
                    (*m_options.input_shape())[0], //Input shape channel dimension
                    m_options.block_filter_channels()[0],
                    {1,1}
                ).stride(1).bias(false)
            )
        );

        m_block_net = this->register_module(
            "block_net",
            nn::ModuleList()
        );

        for(uint32_t i = 0; i<m_options.num_blocks(); i++)
        {
            auto options = FractalBlockOptions<FractalNodeTypeOptions>(
                        m_options.block_depth(),
                        m_options.node_init_options().filter_channels(m_options.block_filter_channels()[i])
                    );

            m_block_net->push_back(
                FractalBlockGroup<FractalNodeType,FractalNodeTypeOptions>(
                    FractalBlockGroupOptions<FractalNodeTypeOptions>(
                        m_options.block_filter_channels()[i],
                        (i+1) < m_options.block_filter_channels().size() ? m_options.block_filter_channels()[i+1] : m_options.block_filter_channels()[i],
                        options
                    )
                )
            );

                    
        }

        //Register the modules needed for global path dropout
        for(uint32_t j = 0; j<m_options.block_depth(); j++)
        {
            m_path_selections.push_back(j);
        }
        m_rnd_device = new std::random_device();
        m_engine = new std::mt19937((*m_rnd_device)());
        m_dist = new std::uniform_int_distribution<int>(0,m_path_selections.size()-1);
    }

    ~FractalNetworkImpl()
    {
        if(m_rnd_device)
        {
            delete m_rnd_device;
            m_rnd_device = nullptr;
        }
        if(m_engine)
        {
            delete m_engine;
            m_engine = nullptr;
        }

        if(m_dist)
        {
            delete m_dist;
            m_dist = nullptr;
        }
    }

    Tensor forward(const Tensor& input, std::vector<int32_t> global_path_selection = {})
    {
        //Set the filter channel dimension from the source tensor to the first layer of the network
        auto z = m_filter_match->forward(input);

        if(this->is_training() && (global_path_selection.size() == 0))
        {
            if(((rand() % 100)/100.0) <= m_options.global_path_dropout_prob())
            {
                for(uint32_t i = 0; i < m_options.num_blocks(); i++)
                {
                    global_path_selection.emplace_back(m_path_selections[(*m_dist)(*m_engine)]);
                    
                }
            }
        }

        if(global_path_selection.size() == 1)
        {
            auto p = global_path_selection[0];
            for(uint32_t i = 0; i < m_options.num_blocks() - 1; i++)
            {
                global_path_selection.emplace_back(p);
            }
        }

        auto gpsct = global_path_selection.size();
        auto c = 0;

        for(size_t j = 0; j < m_block_net->size(); j++)
        {
            std::shared_ptr<FractalBlockGroupImpl<FractalNodeType,FractalNodeTypeOptions>> g = m_block_net->template ptr<FractalBlockGroupImpl<FractalNodeType,FractalNodeTypeOptions>>(j);
            z = g->forward(
                z,
                (gpsct > 0) ? global_path_selection[c] : -1
                );
            c++;//Increment the block counter
        }

        return z;
    }

    void reset() override
    {}

    FractalNetworkOptions<FractalNodeTypeOptions> m_options;

protected:
    FORWARD_HAS_DEFAULT_ARGS({1, torch::nn::AnyValue(std::vector<int32_t>())})

private:
    nn::Conv2d m_filter_match{nullptr}; //Used to match the channel number to the input 
    nn::ModuleList m_block_net{nullptr};
    std::vector<uint32_t> m_path_selections;
    std::random_device* m_rnd_device;
    std::mt19937* m_engine;
    std::uniform_int_distribution<int>* m_dist;
};

//We have to explicitly build the Module wrapper here because the 
//TORCH_MODULE macro does not support template classes with multiple template arguments
template<typename FractalNodeType, typename FractalNodeTypeOptions>
class FractalNetwork : public torch::nn::ModuleHolder<FractalNetworkImpl<FractalNodeType,FractalNodeTypeOptions>> {
   public:                                                         
    using torch::nn::ModuleHolder<FractalNetworkImpl<FractalNodeType,FractalNodeTypeOptions>>::ModuleHolder;             
  };

#endif
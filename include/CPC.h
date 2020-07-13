#ifndef CPC_H_
#define CPC_H_


#include <torch/torch.h>
#include <memory>
#include <iostream>


struct LinearForecastOptions
{
    LinearForecastOptions(int64_t context_vector_dim, int64_t embedding_vector_dim, int64_t num_patches) 
        : context_vector_dim_(context_vector_dim), embedding_vector_dim_(embedding_vector_dim), num_patches_(num_patches) {}

    TORCH_ARG(int64_t, context_vector_dim);
    TORCH_ARG(int64_t, embedding_vector_dim);
    TORCH_ARG(int64_t, num_patches);
};

class LinearForecastFunctionImpl : public torch::nn::Cloneable<LinearForecastFunctionImpl>{
public:
    LinearForecastFunctionImpl(
        int64_t context_vector_dim, 
        int64_t embedding_vector_dim, 
        int64_t num_patches
    ) : LinearForecastFunctionImpl(LinearForecastOptions(context_vector_dim, embedding_vector_dim, num_patches)) {}

    LinearForecastFunctionImpl(const LinearForecastOptions& options_): m_options(options_)
    {
        m_linear_matrix = register_parameter("linear_forcast_matrix",torch::randn({m_options.num_patches(),m_options.embedding_vector_dim(),m_options.context_vector_dim()}));
        torch::nn::init::kaiming_uniform_(m_linear_matrix);
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor patch_idx)
    {
        std::cout << "LM - Size: " << m_linear_matrix.sizes() <<std::endl;
        auto s_matrices = m_linear_matrix.index(patch_idx);
        std::cout << "S-Mat Size: " << s_matrices.sizes() << std::endl << "X Mat Size: " << x.sizes() <<std::endl << "patch_idx: " << patch_idx <<std::endl;
        
        auto res = torch::matmul(s_matrices,x);
        std::cout << res.sizes() << std::endl;
        return res;
    }

    void reset() override
    {
        torch::nn::init::kaiming_uniform_(m_linear_matrix);
    }

    void pretty_print(std::ostream& stream) const override
    {
        stream << "LinearForecastFunction(context_vector_dim=" << m_options.context_vector_dim()
                << ", embedding_vector_dim =" << m_options.embedding_vector_dim()
                << ", num_patches = " << m_options.num_patches()
                << ")";
    }

private:
    torch::Tensor m_linear_matrix;
    LinearForecastOptions m_options;
};

TORCH_MODULE(LinearForecastFunction);

struct StridedUnfoldOptions
{
    StridedUnfoldOptions(torch::ExpandingArray<2> patch_shape, int patch_overlap_px)
        : patch_shape_(std::move(patch_shape)), patch_overlap_px_(patch_overlap_px) {}

    TORCH_ARG(torch::ExpandingArray<2>, patch_shape);

    TORCH_ARG(int, patch_overlap_px) = 0;
};

class StridedUnfoldLayerImpl : public torch::nn::Cloneable<StridedUnfoldLayerImpl> {
public:
    StridedUnfoldLayerImpl(
        torch::ExpandingArray<2> patch_shape,
        int patch_overlap_px,
        int channels = 3
    ) : StridedUnfoldLayerImpl(StridedUnfoldOptions(patch_shape, patch_overlap_px))
    {}

    explicit StridedUnfoldLayerImpl(const StridedUnfoldOptions& options_) : m_options(options_)
    {
        //m_options = std::move(options_); //Take over the options struct
        torch::ExpandingArray<2>& patch_shape = m_options.patch_shape();
        auto ps = (*patch_shape)[0];

        //If the patch shape is not square, then for purposes of the unfold operation we will treat it as an average square.
        if((*patch_shape)[0] != (*patch_shape)[1])
        {
            ps = int((*patch_shape)[0]+(*patch_shape)[1]/2.0);
        }

        //Compute the stride size based on the difference of the patch shape from the overlap pixels
        auto pstride = ps-m_options.patch_overlap_px();

        if(pstride < 0)
        {
            //Bound the stride by the square shape, we do not want to allow an unfold which skips pixels anyway
            pstride = ps;
        }

        m_unfoldLayer = register_module("unfold_layer",torch::nn::Unfold(torch::nn::UnfoldOptions(patch_shape).stride(pstride)));
    }

    torch::Tensor forward(torch::Tensor x)
    {
        auto batch_size = x.size(0);
        return m_unfoldLayer->forward(x).transpose(2,1).contiguous().view({batch_size,-1,3,(*m_options.patch_shape())[0],(*m_options.patch_shape())[1]});
    }

    void reset() override
    {}

    void pretty_print(std::ostream& stream) const override
    {
        stream << "StridedUnfoldLayer(patch_shape=" << m_options.patch_shape()
                << ", patch_overlap_px =" << m_options.patch_overlap_px()
                << ")";
    }


    StridedUnfoldOptions m_options;
    torch::nn::Unfold m_unfoldLayer{nullptr};

};

TORCH_MODULE(StridedUnfoldLayer);

template<typename EncoderType>
struct ImageContrastivePredictiveCoderOptions
{
    ImageContrastivePredictiveCoderOptions(
        torch::ExpandingArray<3> input_shape,
        torch::ExpandingArray<2> patch_shape,
        std::shared_ptr<EncoderType> encoder,
        int patch_overlap_px,
        int ar_unit_hidden_state_size = 128,
        int ar_unit_layers = 1,   
        int encoder_embedding_dim = 128
    ):
        input_shape_(std::move(input_shape)),patch_shape_(std::move(patch_shape)),encoder_(encoder),
        patch_overlap_px_(patch_overlap_px),ar_unit_hidden_state_size_(ar_unit_hidden_state_size),
        ar_unit_layers_(ar_unit_layers), encoder_embedding_dim_(encoder_embedding_dim)
    {}

        TORCH_ARG(torch::ExpandingArray<3>, input_shape);
        TORCH_ARG(torch::ExpandingArray<2>, patch_shape);
        TORCH_ARG(std::shared_ptr<EncoderType>, encoder);
        TORCH_ARG(int, patch_overlap_px);
        TORCH_ARG(int, ar_unit_hidden_state_size);
        TORCH_ARG(int, ar_unit_layers);  
        TORCH_ARG(int, encoder_embedding_dim);

};

template <typename EncoderType>
struct ImageContrastivePredictiveCoder : public torch::nn::Module {
    ImageContrastivePredictiveCoder(
        torch::ExpandingArray<3> input_shape,
        torch::ExpandingArray<2> patch_shape,
        std::shared_ptr<EncoderType> encoder,
        int patch_overlap_px,
        int ar_unit_hidden_state_size = 128,
        int ar_unit_layers = 1,   
        int encoder_embedding_dim = 128
    ) : ImageContrastivePredictiveCoder(ImageContrastivePredictiveCoderOptions<EncoderType>(input_shape, patch_shape, encoder, patch_overlap_px, ar_unit_hidden_state_size, ar_unit_layers, encoder_embedding_dim))
    {}

    ImageContrastivePredictiveCoder(const ImageContrastivePredictiveCoderOptions<EncoderType>& options_) : m_options(options_)
    {
        //We grab all these references here to make the code a little more readable down the line.
        auto& input_shape = m_options.input_shape();
        auto& patch_shape = m_options.patch_shape();
        auto& patch_overlap_px = m_options.patch_overlap_px();
        auto& encoder_embedding_dim = m_options.encoder_embedding_dim();
        auto& ar_unit_hidden_state_size = m_options.ar_unit_hidden_state_size();
        auto& ar_unit_layers = m_options.ar_unit_layers();

        int H = (*input_shape)[1];
        int W = (*input_shape)[2];

        std::cout << "Images will be expected in shape " << H << "," << W << std::endl;

        m_BatchArrangeLayer = register_module("batch_arrange_layer",StridedUnfoldLayer(StridedUnfoldOptions(patch_shape,patch_overlap_px)));

        //Configure the encoder and get the number of patches produced by the batch arrange layer
        {
            torch::NoGradGuard no_grad; //Set no grad context
            m_nPatches = m_BatchArrangeLayer->forward(torch::zeros(input_shape).unsqueeze(0)).squeeze(0).size(0);
            
            //We also need to see if the encoder outputs directly into the embedding dim or if we need to reshape with an FC layer
            c10::IntArrayRef out_shape = m_options.encoder()->forward(torch::zeros({1,3,(*patch_shape)[0],(*patch_shape)[1]})).sizes();
        
        
            if(out_shape.size() == 2)
            {
                //This encoder outputs something of shape [B, E] -- Check if the embedding dimension matches the one we want
                if(out_shape[1] != encoder_embedding_dim)
                {
                    //Register an encoder reshape layer
                    //m_encoder = register_module("encoder",encoder);
                    //m_encoderReshapeLayer = register_module("encoder_reshape",torch::nn::Linear(out_shape[1],encoder_embedding_dim));
                    m_encoderSequence = register_module(
                        "encoder_sequence",
                        torch::nn::Sequential(
                            {
                                {"encoder",m_options.encoder()},
                                {"encoder_reshape",torch::nn::Linear(out_shape[1],encoder_embedding_dim)}
                            }
                        )
                    );
                }
                else
                {
                    //The encoder outputs directly in the expected dimension, just wrap the encoder with a sequential so we can store it in the sequential member object
                    m_encoderSequence = register_module(
                        "encoder_sequence",
                        torch::nn::Sequential(
                            {
                                {"encoder",m_options.encoder()}
                            }
                        )
                    );
                }
                
            }
            else if(out_shape.size() == 4)
            {
                //This encoder outputs a shape of [B, C, H ,W]
                //We will use an adaptive avg pool to reduce it to [B, C, 1, 1]
                m_encoderSequence = register_module(
                    "encoder_sequence",
                    torch::nn::Sequential(
                        {
                            {"encoder",m_options.encoder()},
                            {"avg_pool",torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1,1}))},
                            {"flatten_layer",torch::nn::Flatten()},
                            {"encoder_reshape", torch::nn::Linear(out_shape[1],encoder_embedding_dim)}
                        }
                    )
                );
            }
            else
            {
                //This encoder has some other shape we are not famimilar with so we will just flatten all the dimensions together
                auto shape = out_shape.slice(1).vec();
                m_encoderSequence = register_module(
                    "encoder_sequence",
                    torch::nn::Sequential(
                        {
                            {"encoder",m_options.encoder()},
                            {"flatten_layer",torch::nn::Flatten()},
                            {
                                "encoder_reshape",
                                torch::nn::Linear(std::accumulate(shape.begin(),shape.end(),0),encoder_embedding_dim)
                            }
                        }
                    )
                );
            }
        }
        
        //Autoregressive Layer -- GRU
        m_autoRegressiveLayer = register_module(
            "autoregressive_layer",
            torch::nn::GRU(torch::nn::GRUOptions(encoder_embedding_dim,ar_unit_hidden_state_size).num_layers(ar_unit_layers).bias(false).batch_first(true))
        );

    }

    //Returns a tuple of <Enc, CV>
    std::tuple<torch::Tensor,torch::Tensor> forward(torch::Tensor x) {
        //Input tensor has shape [B, C, H , W]
        x = m_BatchArrangeLayer->forward(x); //[B, P, C, H , W]

        auto sizes = x.sizes().vec();
        //Now we fold together the batch and patch channels so they can be run through the encoder as a single operation
        x = x.view({sizes[0]*sizes[1],sizes[2],sizes[3],sizes[4]}); // [(B P), C, H ,W]

        //Run the encoder sequence over the patches
        x = m_encoderSequence->forward(x); //[(B P), E]

        //Reshape the tensor and separate the batch and patch numbers again.
        auto enc = x.reshape({sizes[0],sizes[1],-1}); //[B, P, E]
        //Normalize the embedding vectors
        enc = torch::nn::functional::normalize(enc,torch::nn::functional::NormalizeFuncOptions().dim(2));//[B, P, E]

        //Now feed this into the GRU -- The first element of the returned tuple is the full context vector
        x = std::get<0>(m_autoRegressiveLayer->forward(enc)); //[B, P, AE]
        x = torch::nn::functional::normalize(x,torch::nn::functional::NormalizeFuncOptions().dim(2));//[B, P, AE]

        return(std::make_tuple(enc, x));
    }



    int64_t Patches() const
    {
        return m_nPatches;
    }

    ImageContrastivePredictiveCoderOptions<EncoderType> m_options;

private:
    
    StridedUnfoldLayer m_BatchArrangeLayer{nullptr};
    torch::nn::Sequential m_encoderSequence{nullptr};
    torch::nn::GRU m_autoRegressiveLayer{nullptr};
    int64_t m_nPatches;
};

template <typename ForecastNetworkType>
class RandomNoiseContrastiveLoss : public torch::nn::Module
{
public:
    RandomNoiseContrastiveLoss(
        uint32_t context_vector_dim,
        uint32_t embedding_vector_dim,
        uint32_t num_patches,
        uint32_t num_distractors = 10
    )
    {
        m_forecast = register_module("forecast_network",ForecastNetworkType(context_vector_dim, embedding_vector_dim, num_patches));
        m_num_distractors = num_distractors;
    }

    torch::Tensor forward(torch::Tensor embedding_vectors, torch::Tensor context_vectors)
    {
        auto evsize = embedding_vectors.sizes();
        auto batch_size = evsize[0];
        auto num_patches = evsize[1];
        auto device = embedding_vectors.device();

        std::cout << context_vectors.sizes() << std::endl;
        
        //Chooses the patch indices serving as the source patches for enc prediction
        auto i = torch::randint(num_patches-1,{batch_size}).to(device);

        auto batch_patch_indexer = torch::arange(num_patches).expand({batch_size,-1}).to(device);

        //Allowed K step mask
        auto k_chooser = (batch_patch_indexer <= (num_patches - 2 - i.unsqueeze(1)));
        
        //Actual k values that indicate the forward step for a given i
        auto k = torch::multinomial(k_chooser.toType(torch::ScalarType::Float),1).toType(torch::ScalarType::Long).squeeze();
        std::cout << k.sizes() << std::endl << k << std::endl;

        //Indices of the ground truth encoding that we will be computing the loss against
        auto gt_indices = i + k + 1;

        //Selects out the ground truth encoding vectors we are trying to predict
        auto br = torch::arange(batch_size,torch::kLong).to(device);

        //Since it seems that libtorch doesn't support indexing with multi dim tensors -- we will take an einsum approach
        //First we transpose the batch and patch dimensions then index out the specific gt indices. 
        //Then we use an einsum to select out the 'diagonal' tensor which is where our desired embedding vectors reside.
        auto gt_vectors = torch::einsum(
            "bbp -> bp",
            embedding_vectors.transpose(0,1).index(gt_indices.to(torch::kLong))
        ); //[B, E]

        //Selects out the associated context vectors that will be used to predict the above encodings.
        auto c_chosen = torch::einsum(
            "bbp -> bp",
            context_vectors.transpose(0,1).index(i.to(torch::kLong))
            ); //[B, E]

        std::cout << "C_Chosen Size: " <<c_chosen.sizes() << std::endl << "GT_Vec Size : " << gt_vectors.sizes() << std::endl;

        //Normalize to N-Sphere
        auto pred_vectors = torch::nn::functional::normalize(
            m_forecast->forward(c_chosen.unsqueeze(2),k).squeeze(), //Predict the embedding based on the context vector - Output will be shape [B, E, 1] so we squeeze to get [B, E]
            torch::nn::functional::NormalizeFuncOptions().dim(1) //Normalize along the embedding dimension 
            );

        //This tensor will be the numerator in the NCE loss function -- We are computing the inner product of the embedding vectors 
        auto f_k = torch::exp(torch::einsum("bv,bv->b",{gt_vectors,pred_vectors}));

        //These are the indicies in the list np.arange(batch_size*patch_size) which correspond to the actual
        //ground truth encodings that we are predicting and hence do not want as distractors.
        auto gt_indices_in_bp = gt_indices + (num_patches*torch::arange(batch_size));

        //This is the list of indices into the [BP, E] ground truth tensor
        auto bp_indexer = torch::arange(num_patches*batch_size).expand({batch_size,-1});

        //Mask of allowed distractors for each GT encoding
        auto distractor_chooser = bp_indexer != gt_indices_in_bp.unsqueeze(1);

        auto distractor_indexer = torch::multinomial(distractor_chooser.to(torch::ScalarType::Float),m_num_distractors).to(torch::ScalarType::Long);

        auto distractors = embedding_vectors.view({batch_size*num_patches,-1}).index(distractor_indexer);
        std::cout << "Distractors Size: " << distractors.sizes() << std::endl << "Distractor Indexer: " << distractor_indexer << std::endl;

        //Finally compute the denominator
        auto df_k = torch::exp(torch::einsum("bde,be->bd",{distractors,pred_vectors})).sum({1});

        //Compute the loss
        return -torch::log(f_k/(df_k+f_k)).mean();
    }

private:
    ForecastNetworkType m_forecast{nullptr};
    uint32_t m_num_distractors;
};


#endif
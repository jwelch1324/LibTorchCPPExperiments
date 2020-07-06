#ifndef CPC_H_
#define CPC_H_


#include <torch/torch.h>
#include <memory>
#include <iostream>


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
        m_options = std::move(options_); //Take over the options struct
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
    )
    {
        int H = (*input_shape)[0];
        int W = (*input_shape)[1];

        std::cout << "Images will be expected in shape " << H << "," << W << std::endl;

        m_BatchArrangeLayer = register_module("batch_arrange_layer",StridedUnfoldLayer(StridedUnfoldOptions(patch_shape,patch_overlap_px)));

        //Configure the encoder and get the number of patches produced by the batch arrange layer
        {
            torch::NoGradGuard no_grad; //Set no grad context
            m_nPatches = m_BatchArrangeLayer->forward(torch::zeros(input_shape).unsqueeze(0)).squeeze(0).size(0);
            
            //We also need to see if the encoder outputs directly into the embedding dim or if we need to reshape with an FC layer
            c10::IntArrayRef out_shape = encoder->forward(torch::zeros({1,3,(*patch_shape)[0],(*patch_shape)[1]})).sizes();
        
        
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
                                {"encoder",encoder},
                                {"encoder_reshape",torch::nn::Linear(out_shape[1],encoder_embedding_dim)}
                            }
                        )
                    );
                }
                else
                {
                    //The encoder outputs directly in the expected dimension, just wrap the encoder with a sequential 
                    m_encoderSequence = register_module(
                        "encoder_sequence",
                        torch::nn::Sequential(
                            {
                                {"encoder",encoder}
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
                            {"encoder",encoder},
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
                            {"encoder",encoder},
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

    StridedUnfoldLayer m_BatchArrangeLayer{nullptr};
    torch::nn::Sequential m_encoderSequence{nullptr};
    torch::nn::GRU m_autoRegressiveLayer{nullptr};
    int64_t m_nPatches;
    

};

#endif
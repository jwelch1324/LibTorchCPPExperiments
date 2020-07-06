#include <torch/torch.h>
#include "include/CPC.h"
#include <iostream>
#include <memory>
#include <numeric>


struct Net : public torch::nn::Module{
     Net() {
          // Construct and register two Linear submodules
          fc1 = register_module("fc1", torch::nn::Linear(3072, 64));
          fc2 = register_module("fc2", torch::nn::Linear(64,32));
          fc3 = register_module("fc3", torch::nn::Linear(32,10));
     }

     //Implement the forward algorithm
     torch::Tensor forward(torch::Tensor x) {
          // Use one of many tensor functions
          x = torch::relu(fc1->forward(x.reshape({x.size(0), -1})));
          x = torch::dropout(x, 0.5,is_training());
          x = torch::relu(fc2->forward(x));
          x = torch::log_softmax(fc3->forward(x),1);
          return x;
     }

     torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};


int main() {
     // auto t = torch::randn({1,3,256,256});
     // auto uf = torch::nn::Unfold(torch::nn::UnfoldOptions({64,64}).stride(62));
     // auto zz = uf->forward(t).transpose(2,1).contiguous().view({1,-1,3,64,64}).squeeze(0);

     // std::cout << zz.sizes() << std::endl;


     auto t = torch::randn({5,2,22,33});
     std::cout << t.view({-1,22,33}).sizes() << std::endl;
     std::cout << t.is_contiguous() << std::endl;
     auto ss = t.sizes().slice(1);
     std::cout << std::accumulate(ss.begin(),ss.end(),0) <<std::endl;

     auto net = std::make_shared<ImageContrastivePredictiveCoder<Net>>(
          torch::ExpandingArray<3>({3,256,256}),
          torch::ExpandingArray<2>({32,32}),
          std::make_shared<Net>(),
          0
     );

     auto res = net->forward(torch::randn({2,3,256,256}));
     std::cout << std::get<0>(res).sizes() << std::endl << std::get<1>(res).sizes()<<std::endl;

     return 0;
}

int main2() {
     auto net = std::make_shared<Net>();
     torch::Device device(torch::kCUDA);
     net->to(device);
     //Create a MNIST dataset
     auto data_loader = torch::data::make_data_loader(
          torch::data::datasets::MNIST("/media/ramdisk/data/MNIST/raw").map(
               torch::data::transforms::Stack<>()),
               256
          );

     //Use SGD
     torch::optim::SGD optimizer(net->parameters(), 0.01);

     for (size_t epoch = 1; epoch <= 10; ++epoch)
     {
          size_t batch_index = 0;
          //Iterate the data loader to yield batches from the dataset
          for(auto& batch: *data_loader){
               //Reset the gradients
               optimizer.zero_grad();
               //Execute the model on the input data.
               torch::Tensor prediction = net->forward(batch.data.to(device));

               //Compute loss 
               torch::Tensor loss = torch::nll_loss(prediction, batch.target.to(device));

               //Compute gradients of the loss w.r.t. the parameters of our model.
               loss.backward();

               //Update the parameters based on the gradients
               optimizer.step();

               //Output the loss and checkpoint every 100 batches.
               if(++batch_index % 100 == 0) {
                    std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
                              << " | Loss: " << loss.item<float>() << std::endl;

                    torch::save(net, "net.pt");
               }
          }
     }

     return 0;
}

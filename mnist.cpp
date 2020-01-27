#include <torch/torch.h>
#include <iostream>

using namespace torch;
using std::cout, std::endl;
/*
struct Net : nn::Module{

    // torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};

    nn::Conv2d conv1, conv2;
    nn::Linear fc;
    int tail;

    Net(): 
        conv1(nn::Conv2dOptions(1, tail, 3)),
        conv2(nn::Conv2dOptions(tail, tail, 3)),
        fc(tail, 10) {
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("fc", fc);
    }
     
    Tensor forward(Tensor x) {
        x = max_pool2d(relu(conv1->forward(x)),2);
        x = max_pool2d(relu(conv2->forward(x)),2);
        x = sigmoid(fc->forward(x));
        return x;
    }
};*/


struct Print : nn::Module {
    Print() {
    }
    Tensor forward(Tensor x) {
        cout << x.sizes() << endl;
        return x;
    }
};


struct Softmax : nn::Module {
    Softmax() {}
    Tensor forward(Tensor x) {
        x = log_softmax(x, 1);
        return x;
    }
};

auto Net = nn::Sequential(
    nn::Conv2d(1,10,3),
    nn::MaxPool2d(2),
    nn::ReLU(),
    nn::Conv2d(10,10,3),
    nn::MaxPool2d(2),
    nn::ReLU(),
    nn::Flatten(),
    nn::Linear(250, 10),
    Softmax()
);

int main() {
    // Create a new Net.
  //auto net = std::make_shared<Net>();
    auto net = Net; 
  // Create a multi-threaded data loader for the MNIST dataset.
    auto data_loader = torch::data::make_data_loader(
      torch::data::datasets::MNIST("./data").map(
          torch::data::transforms::Stack<>()),
      /*batch_size=*/64);
     // Instantiate an SGD optimization algorithm to update our Net's parameters.
     torch::optim::SGD optimizer(net->parameters(), /*lr=*/0.01);

    for (size_t epoch = 1; epoch <= 10; ++epoch) {
        size_t batch_index = 0;
        // Iterate the data loader to yield batches from the dataset.
        for (auto& batch : *data_loader) {
            // Reset gradients.
            optimizer.zero_grad();
            // Execute the model on the input data.
            torch::Tensor prediction = net->forward(batch.data);
            // Compute a loss value to judge the prediction of our model.
            torch::Tensor loss = torch::nll_loss(prediction, batch.target);
            // Compute gradients of the loss w.r.t. the parameters of our model.
            loss.backward();
            // Update the parameters based on the calculated gradients.
            optimizer.step();
            // Output the loss and checkpoint every 100 batches.
            if (++batch_index % 100 == 0) {
                std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
                        << " | Loss: " << loss.item<float>() << std::endl;
                // cout<<"prediction:\n"<<(Tensor)prediction << "\ntarget:\n"<< batch.target<<endl;
                // Serialize your model periodically as a checkpoint.
                torch::save(net, "net.pt");
            }
        }
    }
}
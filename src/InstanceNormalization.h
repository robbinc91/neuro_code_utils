#pragma once
#include <torch/torch.h>

// C++/LibTorch implementation of the simple InstanceNormalization from the
// Python module. Usage:
//   InstanceNormalization model(/*epsilon=*/1e-6);
//   auto out = model->forward(input);

struct InstanceNormalizationImpl : public torch::nn::Module {
    InstanceNormalizationImpl(double epsilon = 1e-6) : epsilon_(epsilon) {}

    torch::Tensor forward(const torch::Tensor& x) {
        // compute mean and variance over spatial dims (H,W) i.e., dims 2 and 3
        auto mean = x.mean({2, 3}, /*keepdim=*/true);
        // torch::var defaults match torch.var in Python (unbiased=true), use same
        auto var = x.var({2, 3}, /*unbiased=*/true, /*keepdim=*/true);
        auto y = (x - mean) / torch::sqrt(var + epsilon_);
        return y;
    }

    double epsilon_;
};

TORCH_MODULE(InstanceNormalization);

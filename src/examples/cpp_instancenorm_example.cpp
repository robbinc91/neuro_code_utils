// Small example demonstrating the InstanceNormalization module in C++ (LibTorch)
#include <torch/torch.h>
#include <iostream>
#include "../InstanceNormalization.h"

int main() {
    // Create a random tensor with shape [batch, channels, H, W]
    auto input = torch::rand({2, 4, 8, 8});
    InstanceNormalization inst(1e-6);
    auto out = inst->forward(input);
    std::cout << "Input mean (per-channel, spatial avg)\n" << input.mean({2,3}) << std::endl;
    std::cout << "Output mean (should be ~0)\n" << out.mean({2,3}) << std::endl;
    std::cout << "Output var (should be ~1)\n" << out.var({2,3}, /*unbiased=*/true) << std::endl;
    return 0;
}

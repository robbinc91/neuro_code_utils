// Example C++ program to run inference with a PyTorch ScriptModule (LibTorch)
// Build: see LibTorch docs. Minimal example using CPU.
// Usage: ./cpp_predict_torch model.pt

#include <torch/script.h>
#include <iostream>
#include <memory>
#ifdef HAS_CUSTOM_FUNCS
#include "custom_funcs.h"
#endif

int main(int argc, const char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <model.pt>\n";
    return 1;
  }
  std::string model_path = argv[1];

  try {
    torch::jit::script::Module module = torch::jit::load(model_path);
    module.eval();

    // create a dummy input: 1x3x224x224
    std::vector<int64_t> dims = {1, 3, 224, 224};
    at::Tensor input = torch::rand(dims);

  #ifdef HAS_CUSTOM_FUNCS
    // Apply custom activation (if provided) element-wise to the input tensor
    // The generated header should provide an `extern "C" double <func>(double)` function.
    // Example uses a function named `custom_activation` if present.
    bool has_custom = false;
    // Note: presence of HAS_CUSTOM_FUNCS indicates header compiled in; user must ensure
    // the function name matches one in the header.
    try {
      // apply to CPU tensor data
      input = input.to(torch::kCPU);
      auto accessor = input.accessor<float,4>();
      for (int b=0;b<input.size(0);++b)
        for (int c=0;c<input.size(1);++c)
          for (int h=0;h<input.size(2);++h)
            for (int w=0;w<input.size(3);++w) {
              double v = static_cast<double>(accessor[b][c][h][w]);
              double outv = custom_activation(v); // name must match generated header
              accessor[b][c][h][w] = static_cast<float>(outv);
            }
    } catch (...) {
      std::cerr << "Custom activation application failed; continuing without it.\n";
    }
  #endif

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);

    at::Tensor out = module.forward(inputs).toTensor();
    std::cout << "Output shape: ";
    for (auto d : out.sizes()) std::cout << d << " ";
    std::cout << std::endl;
  }
  catch (const c10::Error& e) {
    std::cerr << "Error loading the model\n";
    return -1;
  }

  return 0;
}

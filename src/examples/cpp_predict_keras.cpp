// Example C++ program to run inference with a TensorFlow SavedModel using the
// TensorFlow C++ API. Building and linking TensorFlow C++ is non-trivial and
// depends on your platform; this is a minimal illustrative example.
// Usage: ./cpp_predict_keras /path/to/saved_model

#include <tensorflow/cc/saved_model/loader.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/tensor.h>
#include <iostream>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <saved_model_dir>\n";
    return 1;
  }
  std::string model_dir = argv[1];

  tensorflow::SavedModelBundle bundle;
  tensorflow::SessionOptions session_options;
  tensorflow::RunOptions run_options;

  auto status = tensorflow::LoadSavedModel(session_options, run_options, model_dir, {"serve"}, &bundle);
  if (!status.ok()) {
    std::cerr << "Error loading SavedModel: " << status.ToString() << std::endl;
    return 1;
  }

  // Construct a dummy input tensor (batch=1,224,224,3) of floats
  tensorflow::Tensor input(tensorflow::DT_FLOAT, tensorflow::TensorShape({1,224,224,3}));
  auto flat = input.flat<float>();
  for (int i = 0; i < flat.size(); ++i) flat(i) = static_cast<float>(rand()) / RAND_MAX;

  std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {{"serving_default_input_1:0", input}}; // adjust input name to your model
  std::vector<tensorflow::Tensor> outputs;
  status = bundle.session->Run(inputs, {"StatefulPartitionedCall:0"}, {}, &outputs);
  if (!status.ok()) {
    std::cerr << "Error running model: " << status.ToString() << std::endl;
    return 1;
  }

  std::cout << "Got " << outputs.size() << " outputs" << std::endl;
  return 0;
}

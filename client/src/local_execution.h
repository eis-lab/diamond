#include "image_processing.h"

at::Tensor execute_local_parts(torch::jit::script::Module model, torch::Tensor input_tensor, int partitioning_point, std::vector<int64_t> &serverside_shape);

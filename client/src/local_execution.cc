#include "local_execution.h"

at::Tensor execute_local_parts(torch::jit::script::Module model, torch::Tensor input_tensor, int partitioning_point, std::vector<int64_t> &serverside_shape)
{
	std::vector<torch::jit::IValue> local_inputs;
	local_inputs.push_back(input_tensor);	

	at::Tensor indexTensor = torch::tensor(partitioning_point).to(torch::kCUDA);
	indexTensor = torch::reshape(indexTensor, {-1,1});
	local_inputs.push_back(indexTensor);

	at::Tensor local_output = model.forward(local_inputs).toTensor();

	std::vector<int64_t> shape;
	for (int i =0 ; i < local_output.dim(); i ++)
	{
		serverside_shape.push_back(local_output.size(i));
	}
	
	local_output = local_output.flatten().to(torch::kCPU);
	return local_output;
}	
	//////////////////////////LOCAL EXECUTION DONE//////////////////////////


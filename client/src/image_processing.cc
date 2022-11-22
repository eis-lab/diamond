//
// Created by jungmo on 21. 6. 27..
// This code is from Triton Inference Server repository on GitHub.
//

#include "image_processing.h"
#define IMAGE_SIZE 224
#define CHANNELS 3


bool tensor_transform(cv::Mat img_rgb_u8, torch::Tensor& input_tensor)
{
	input_tensor = torch::from_blob(img_rgb_u8.data, {img_rgb_u8.rows, img_rgb_u8.cols, 3}, torch::kByte);
	input_tensor = input_tensor.permute({2, 0, 1});
	input_tensor = input_tensor.toType(at::kFloat);

	const float mean[3] = {0.485, 0.456, 0.406};
	const float std[3] = {0.229, 0.224, 0.225};

	input_tensor.div_(255.);
	for(int ch=0; ch < 3; ch++)
	{
		input_tensor[ch].sub_(mean[ch]).div_(std[ch]);
	}
	input_tensor.unsqueeze_(0);
	input_tensor = input_tensor.to(torch::kCUDA);
	return true;
}

bool loadimage(std::string file_name, torch::Tensor& input_tensor, int width)
{
	cv::Mat img_bgr_u8 = cv::imread(file_name,cv::IMREAD_COLOR);
	cv::Mat img_rgb_u8;
	cv::resize(img_bgr_u8, img_bgr_u8, cv::Size(width, width));
	cv::cvtColor(img_bgr_u8, img_rgb_u8, cv::COLOR_BGR2RGB);

	tensor_transform(img_rgb_u8, input_tensor);
}
	


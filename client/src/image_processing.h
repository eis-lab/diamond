//
// Created by jungmo on 21. 6. 27..
//

#ifndef DIAMOND_CLIENT_IMAGE_PROCESSING_H
#define DIAMOND_CLIENT_IMAGE_PROCESSING_H

#include <opencv2/core/version.hpp>

#if CV_MAJOR_VERSION == 2
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#elif CV_MAJOR_VERSION >= 3

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <iterator>
#include <fstream>

#endif
#include <torch/script.h>
#include <torch/library.h>

#if CV_MAJOR_VERSION == 4
#define GET_TRANSFORMATION_CODE(x) cv::COLOR_##x
#else
#define GET_TRANSFORMATION_CODE(x) CV_##x
#endif


enum ScaleType { NONE = 0, VGG = 1, INCEPTION = 2 };

/*
void
FileToInputData(
        const std::string& filename, size_t c, size_t h, size_t w,
        const std::string& format, int type1, int type3, ScaleType scale,
        std::vector<uint8_t>* input_data);
*/
//bool loadimage(std::string file_name, cv::Mat& image);

bool loadimage(std::string file_name, torch::Tensor& input_tensor, int width);
#endif //DIAMOND_CLIENT_IMAGE_PROCESSING_H

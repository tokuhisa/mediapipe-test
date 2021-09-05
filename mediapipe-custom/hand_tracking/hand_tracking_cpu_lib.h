#pragma once

#define CPPLIBRARY_API __declspec(dllexport) 

#include <cstdlib>

#include "mediapipe/framework/formats/image_frame.h"


extern "C"
{
	CPPLIBRARY_API float count_up(void);
	CPPLIBRARY_API void count_init(void);
	CPPLIBRARY_API void test_uint8_array(uint8* data);
	CPPLIBRARY_API void test_float_array(float* data);
	CPPLIBRARY_API void create_image_frame1(int width, int height, uint8* input_pixel_data);
	CPPLIBRARY_API void create_image_frame2(int width, int height, uint8* input_pixel_data);
	CPPLIBRARY_API void create_image_frame3(int width, int height, uint8* input_pixel_data);
	CPPLIBRARY_API void create_image_frame4(int width, int height, uint8* input_pixel_data);
}
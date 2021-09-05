#pragma once

#define CPPLIBRARY_API __declspec(dllexport) 

#include <cstdlib>
#include "mediapipe/framework/deps/safe_int.h"

extern "C"
{
	CPPLIBRARY_API void test_pose_tracking(void);
	CPPLIBRARY_API void init_pose_tracking(void);
	CPPLIBRARY_API void create_image_frame(int width, int height, uint8* input_pixel_data);
	CPPLIBRARY_API void process_pose_tracking(int width, int height, uint8* input_pixel_data, float* output_segmentation_mask, int64 frame_timestamp_us);
}
#pragma once

#define CPPLIBRARY_API __declspec(dllexport) 

#include <cstdlib>
#include "mediapipe/framework/deps/safe_int.h"

extern "C"
{
	CPPLIBRARY_API int init_pose_tracking(void);
	CPPLIBRARY_API int process_pose_tracking(int width, int height, uint8* input_pixel_data, int64 frame_timestamp_us);
	CPPLIBRARY_API int get_segmentation_mask(int width, int height, float* output_segmentation_mask);
}
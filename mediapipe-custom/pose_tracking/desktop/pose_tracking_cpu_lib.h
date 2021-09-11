#pragma once

#define CPPLIBRARY_API __declspec(dllexport) 

#include <cstdlib>
#include "mediapipe/framework/deps/safe_int.h"

extern "C"
{
	CPPLIBRARY_API int init_pose_tracking(void);
	CPPLIBRARY_API int process_pose_tracking(int width, int height, uint8* input_pixel_data, int64 frame_timestamp_us);
	CPPLIBRARY_API int get_segmentation_mask(int width, int height, float* output_segmentation_mask);
	CPPLIBRARY_API int get_landmarks(float* x_array, float* y_array, float* z_array, float* visibilities, float* presences, int size);
}
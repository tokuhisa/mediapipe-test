#pragma once

#define CPPLIBRARY_API __declspec(dllexport) 

#include <cstdlib>
#include "mediapipe/framework/deps/safe_int.h"
#include <csetjmp>
#include <csignal>
#include <string>
#include "mediapipe/util/resource_util_custom.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/framework/deps/ret_check.h"

extern "C"
{
	CPPLIBRARY_API int init_pose_tracking(void);
	CPPLIBRARY_API int process_pose_tracking(int width, int height, uint8* input_pixel_data, int64 frame_timestamp_us, int rgba);
	CPPLIBRARY_API int get_segmentation_mask(int width, int height, float* output_segmentation_mask);
	CPPLIBRARY_API int apply_segmentation_mask(int width, int height, uint8* rgba_pixel_data, float* segmentation_mask, float threshold);
	CPPLIBRARY_API int get_landmarks(float* x_array, float* y_array, float* z_array, float* visibilities, float* presences, int size);
    CPPLIBRARY_API int get_pose_landmarks(float* x_array, float* y_array, float* z_array, float* visibilities, float* presences, int size);
	
    typedef bool ResourceProvider(const char* path, std::string* output);

    CPPLIBRARY_API void set_custom_global_resource_provider(ResourceProvider* resource_provider);
}
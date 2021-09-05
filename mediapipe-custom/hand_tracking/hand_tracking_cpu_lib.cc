#include "hand_tracking_cpu_lib.h"

#include "absl/memory/memory.h"

float count = 0;
CPPLIBRARY_API float count_up(void)
{
	count += 0.1;
	return count;
}

CPPLIBRARY_API void count_init(void)
{
	count = 0;
}


CPPLIBRARY_API void test_uint8_array(uint8* data) {

}

CPPLIBRARY_API void test_float_array(float* data) {

}

CPPLIBRARY_API void create_image_frame1(int width, int height, uint8* input_pixel_data)
{
	int number_Of_channels = 3; // SRGB format
	int byte_depth = 1; // SRGB format
	int width_step = width * number_Of_channels * byte_depth;
    auto input_frame = absl::make_unique<mediapipe::ImageFrame>(mediapipe::ImageFormat::SRGB, width, height, width_step, input_pixel_data);
}

CPPLIBRARY_API void create_image_frame2(int width, int height, uint8* input_pixel_data)
{
	int number_Of_channels = 3; // SRGB format
	int byte_depth = 1; // SRGB format
	int width_step = width * number_Of_channels * byte_depth;
    auto input_frame = mediapipe::ImageFrame(mediapipe::ImageFormat::SRGB, width, height, width_step, input_pixel_data);
}


CPPLIBRARY_API void create_image_frame3(int width, int height, uint8* input_pixel_data)
{
    auto input_frame = absl::make_unique<mediapipe::ImageFrame>(mediapipe::ImageFormat::SRGB, width, height, mediapipe::ImageFrame::kDefaultAlignmentBoundary);
}

CPPLIBRARY_API void create_image_frame4(int width, int height, uint8* input_pixel_data)
{
    auto input_frame = absl::make_unique<mediapipe::ImageFrame>(mediapipe::ImageFormat::SRGB, width, height, mediapipe::ImageFrame::kDefaultAlignmentBoundary);
	int number_Of_channels = 3; // SRGB format
	int byte_depth = 1; // SRGB format
	int width_step = width * number_Of_channels * byte_depth;
	input_frame->CopyPixelData(mediapipe::ImageFormat::SRGB, width, height, width_step, input_pixel_data, mediapipe::ImageFrame::kDefaultAlignmentBoundary); 
}

CPPLIBRARY_API void create_image_frame5(int width, int height, float* output_frame)
{
    auto input_frame = absl::make_unique<mediapipe::ImageFrame>(mediapipe::ImageFormat::VEC32F1, width, height, mediapipe::ImageFrame::kDefaultAlignmentBoundary);
	input_frame->CopyToBuffer(output_frame, width * height)
}

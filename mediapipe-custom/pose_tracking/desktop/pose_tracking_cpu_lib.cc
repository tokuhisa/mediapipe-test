#include "pose_tracking_cpu_lib.h"

// #include "mediapipe/framework/calculator_framework.h"
// #include "mediapipe/framework/formats/image_frame.h"
// #include "mediapipe/framework/port/parse_text_proto.h"
// #include "mediapipe/framework/port/status.h"

// constexpr char kInputStream[] = "image";
// constexpr char kOutputStreamSegmentationMask[] = "segmentation_mask";

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
#include "pose_tracking_camera_lib.h"

#include <cstdlib>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"



std::unique_ptr<mediapipe::CalculatorGraph> graph;
std::unique_ptr<mediapipe::OutputStreamPoller> poller;
auto segmentation_mask = absl::make_unique<mediapipe::ImageFrame>();
cv::VideoCapture capture;

constexpr char kWindowName[] = "MediaPipe";


mediapipe::CalculatorGraphConfig build_graph_config_from_file(void) {
  
  std::string calculator_graph_config_contents;
  mediapipe::file::GetContents("graph.pbtxt", &calculator_graph_config_contents);
  LOG(INFO) << "Get calculator graph config contents: " << calculator_graph_config_contents;
  return mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(calculator_graph_config_contents);

}

absl::Status InitPoseTracking() {
	LOG(INFO) << "Initialize the calculator graph.";
	graph = absl::make_unique<::mediapipe::CalculatorGraph>();
	// MP_RETURN_IF_ERROR(graph->Initialize(build_graph_config()));
	MP_RETURN_IF_ERROR(graph->Initialize(build_graph_config_from_file()));
	
	LOG(INFO) << "Start running the calculator graph.";
	
  char kOutputStreamSegmentationMask[] = "segmentation_mask";
	auto status_or_poller = graph->AddOutputStreamPoller(kOutputStreamSegmentationMask);
	poller = absl::make_unique<mediapipe::OutputStreamPoller>(std::move(status_or_poller.value()));

	MP_RETURN_IF_ERROR(graph->StartRun({}));
	
	return absl::OkStatus();
}

CPPLIBRARY_API int init_pose_tracking(void)
{
  capture.open(0);
  cv::namedWindow(kWindowName, /*flags=WINDOW_AUTOSIZE*/ 1);
	absl::Status status = InitPoseTracking();
	return status.raw_code();
}

absl::Status ProcessPoseTracking(int width, int height, uint8* input_pixel_data, int64 frame_timestamp_us) {
  auto input_frame = absl::make_unique<mediapipe::ImageFrame>(mediapipe::ImageFormat::SRGB, width, height, mediapipe::ImageFrame::kDefaultAlignmentBoundary);
	// input_frame->CopyPixelData(mediapipe::ImageFormat::SRGB, width, height, width_step, input_pixel_data, mediapipe::ImageFrame::kDefaultAlignmentBoundary);
	input_frame->CopyPixelData(mediapipe::ImageFormat::SRGB, width, height, input_pixel_data, mediapipe::ImageFrame::kDefaultAlignmentBoundary);
    // auto input_frame = absl::make_unique<mediapipe::ImageFrame>(mediapipe::ImageFormat::SRGB, width, height, width_step, input_pixel_data);

	// Send image packet into the graph.
	char kInputStream[] = "image";
  MP_RETURN_IF_ERROR(graph->AddPacketToInputStream(kInputStream, mediapipe::Adopt(input_frame.release()).At(mediapipe::Timestamp(frame_timestamp_us))));
    // auto input_frame = mediapipe::ImageFrame(mediapipe::ImageFormat::SRGB, width, height, width_step, input_pixel_data);
    // MP_RETURN_IF_ERROR(graph->AddPacketToInputStream("image", mediapipe::MakePacket<mediapipe::ImageFrame>(mediapipe::ImageFormat::SRGB, width, height, width_step, input_pixel_data).At(mediapipe::Timestamp(frame_timestamp_us))));

    // Get the graph result packet.
  mediapipe::Packet packet;
  if (poller->QueueSize() == 0) {
		return absl::UnavailableError("Could not get segmentation_mask.");
	}
  if (!poller->Next(&packet)) {
		return absl::UnavailableError("Could not get segmentation_mask.");
	}
	
	auto& output_frame = packet.Get<mediapipe::ImageFrame>();
	segmentation_mask->CopyFrom(output_frame, mediapipe::ImageFrame::kDefaultAlignmentBoundary);

  cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
  cv::imshow(kWindowName, output_frame_mat);

	return absl::OkStatus();
}

CPPLIBRARY_API int process_pose_tracking()
{
  
  // Capture opencv camera or video frame.
  cv::Mat camera_frame_raw;
  capture >> camera_frame_raw;
  if (camera_frame_raw.empty()) {
    LOG(INFO) << "Ignore empty frames from camera.";
    return -1;
  }
  cv::Mat camera_frame;
  cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);
  cv::flip(camera_frame, camera_frame, /*flipcode=HORIZONTAL*/ 1);
  

  // Wrap Mat into an ImageFrame.
  auto input_frame = absl::make_unique<mediapipe::ImageFrame>(mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,mediapipe::ImageFrame::kDefaultAlignmentBoundary);
  cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
  camera_frame.copyTo(input_frame_mat);
  
  size_t frame_timestamp_us = (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
	absl::Status status = ProcessPoseTracking(camera_frame.cols, camera_frame.rows, input_frame->MutablePixelData(), frame_timestamp_us);
	return status.raw_code();
}

CPPLIBRARY_API int get_segmentation_mask(int width, int height, float* output_segmentation_mask) {
	int segmentation_mask_size = width * height; // VEC32F1
	if (segmentation_mask && segmentation_mask->Width() == width && segmentation_mask->Height() == height) {
		segmentation_mask->CopyToBuffer(output_segmentation_mask, segmentation_mask_size);
		return 0;
	} else {
		return 14;
	}
}




absl::Status ProcessPoseTrackingTest1(int width, int height, uint8* input_pixel_data, int64 frame_timestamp_us) {
	int number_Of_channels = 3; // SRGB format
	int byte_depth = 1; // SRGB format
	int width_step = width * number_Of_channels * byte_depth;

    auto input_frame = absl::make_unique<mediapipe::ImageFrame>(mediapipe::ImageFormat::SRGB, width, height, mediapipe::ImageFrame::kDefaultAlignmentBoundary);
	input_frame->CopyPixelData(mediapipe::ImageFormat::SRGB, width, height, width_step, input_pixel_data, mediapipe::ImageFrame::kDefaultAlignmentBoundary);

	return absl::OkStatus();
}

absl::Status ProcessPoseTrackingTest2(int width, int height, uint8* input_pixel_data, int64 frame_timestamp_us) {
	int number_Of_channels = 3; // SRGB format
	int byte_depth = 1; // SRGB format
	int width_step = width * number_Of_channels * byte_depth;

    auto input_frame = absl::make_unique<mediapipe::ImageFrame>(mediapipe::ImageFormat::SRGB, width, height, mediapipe::ImageFrame::kDefaultAlignmentBoundary);
	input_frame->CopyPixelData(mediapipe::ImageFormat::SRGB, width, height, width_step, input_pixel_data, mediapipe::ImageFrame::kDefaultAlignmentBoundary);

	// Send image packet into the graph.
	char kInputStream[] = "image";
    MP_RETURN_IF_ERROR(graph->AddPacketToInputStream(kInputStream, mediapipe::Adopt(input_frame.release()).At(mediapipe::Timestamp(frame_timestamp_us))));

	return absl::OkStatus();
}

absl::Status ProcessPoseTrackingTest3(int width, int height, uint8* input_pixel_data, int64 frame_timestamp_us) {
	int number_Of_channels = 3; // SRGB format
	int byte_depth = 1; // SRGB format
	int width_step = width * number_Of_channels * byte_depth;

    auto input_frame = absl::make_unique<mediapipe::ImageFrame>(mediapipe::ImageFormat::SRGB, width, height, mediapipe::ImageFrame::kDefaultAlignmentBoundary);
	input_frame->CopyPixelData(mediapipe::ImageFormat::SRGB, width, height, width_step, input_pixel_data, mediapipe::ImageFrame::kDefaultAlignmentBoundary);

	// Send image packet into the graph.
	char kInputStream[] = "image";
    MP_RETURN_IF_ERROR(graph->AddPacketToInputStream(kInputStream, mediapipe::Adopt(input_frame.release()).At(mediapipe::Timestamp(frame_timestamp_us))));

    // Get the graph result packet.
    mediapipe::Packet packet;
    if (poller->QueueSize() == 0) {
		return absl::UnavailableError("Could not get segmentation_mask.");
	}

	return absl::OkStatus();
}


absl::Status ProcessPoseTrackingTest4(int width, int height, uint8* input_pixel_data, int64 frame_timestamp_us) {
	int number_Of_channels = 3; // SRGB format
	int byte_depth = 1; // SRGB format
	int width_step = width * number_Of_channels * byte_depth;

	auto input_frame = absl::make_unique<mediapipe::ImageFrame>(mediapipe::ImageFormat::SRGB, width, height, mediapipe::ImageFrame::kDefaultAlignmentBoundary);
	input_frame->CopyPixelData(mediapipe::ImageFormat::SRGB, width, height, width_step, input_pixel_data, mediapipe::ImageFrame::kDefaultAlignmentBoundary);

	// Send image packet into the graph.
	char kInputStream[] = "image";
    MP_RETURN_IF_ERROR(graph->AddPacketToInputStream(kInputStream, mediapipe::Adopt(input_frame.release()).At(mediapipe::Timestamp(frame_timestamp_us))));

    // Get the graph result packet.
    mediapipe::Packet packet;
    if (poller->QueueSize() == 0) {
		return absl::UnavailableError("Could not get segmentation_mask.");
	}
    if (!poller->Next(&packet)) {
		return absl::UnavailableError("Could not get segmentation_mask.");
	}

	return absl::OkStatus();
}


CPPLIBRARY_API int process_pose_tracking_test1(int width, int height, uint8* input_pixel_data, int64 frame_timestamp_us)
{
	absl::Status status = ProcessPoseTrackingTest1(width, height, input_pixel_data, frame_timestamp_us);
	return status.raw_code();
}


CPPLIBRARY_API int process_pose_tracking_test2(int width, int height, uint8* input_pixel_data, int64 frame_timestamp_us)
{
	absl::Status status = ProcessPoseTrackingTest2(width, height, input_pixel_data, frame_timestamp_us);
	return status.raw_code();
}


CPPLIBRARY_API int process_pose_tracking_test3(int width, int height, uint8* input_pixel_data, int64 frame_timestamp_us)
{
	absl::Status status = ProcessPoseTrackingTest3(width, height, input_pixel_data, frame_timestamp_us);
	return status.raw_code();
}


CPPLIBRARY_API int process_pose_tracking_test4(int width, int height, uint8* input_pixel_data, int64 frame_timestamp_us)
{
	absl::Status status = ProcessPoseTrackingTest4(width, height, input_pixel_data, frame_timestamp_us);
	return status.raw_code();
}
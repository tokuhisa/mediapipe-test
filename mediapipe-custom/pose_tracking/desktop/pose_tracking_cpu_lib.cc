#include "pose_tracking_cpu_lib.h"

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/formats/landmark.pb.h"


std::unique_ptr<mediapipe::CalculatorGraph> graph;
std::unique_ptr<mediapipe::OutputStreamPoller> segmentation_mask_poller;
std::unique_ptr<mediapipe::OutputStreamPoller> landmarks_poller;
std::unique_ptr<mediapipe::OutputStreamPoller> pose_landmarks_poller;

auto segmentation_mask = absl::make_unique<mediapipe::ImageFrame>();
auto landmarks = absl::make_unique<mediapipe::LandmarkList>();
auto pose_landmarks = absl::make_unique<mediapipe::NormalizedLandmarkList>();

mediapipe::CalculatorGraphConfig build_graph_config(void) {
	return mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(R"pb(
# MediaPipe graph to detect/predict pose landmarks. (CPU input, and inference is
# executed on CPU.) This graph tries to skip pose detection as much as possible
# by using previously detected/predicted landmarks for new images.
#
# It is required that "pose_detection.tflite" is available at
# "mediapipe/modules/pose_detection/pose_detection.tflite"
# path during execution.
#
# It is required that "pose_landmark_lite.tflite" or
# "pose_landmark_full.tflite" or "pose_landmark_heavy.tflite" is available at
# "mediapipe/modules/pose_landmark/pose_landmark_lite.tflite" or
# "mediapipe/modules/pose_landmark/pose_landmark_full.tflite" or
# "mediapipe/modules/pose_landmark/pose_landmark_heavy.tflite"
# path respectively during execution, depending on the specification in the
# MODEL_COMPLEXITY input side packet.
#
# EXAMPLE:
#   node {
#     calculator: "PoseLandmarkCpu"
#     input_side_packet: "MODEL_COMPLEXITY:model_complexity"
#     input_side_packet: "SMOOTH_LANDMARKS:smooth_landmarks"
#     input_side_packet: "ENABLE_SEGMENTATION:enable_segmentation"
#     input_side_packet: "SMOOTH_SEGMENTATION:smooth_segmentation"
#     input_stream: "IMAGE:image"
#     output_stream: "LANDMARKS:pose_landmarks"
#     output_stream: "SEGMENTATION_MASK:segmentation_mask"
#   }
type: "PoseLandmarkCpu"
# CPU image. (ImageFrame)
input_stream: "IMAGE:image"
# Whether to filter landmarks across different input images to reduce jitter.
# If unspecified, functions as set to true. (bool)
# input_side_packet: "SMOOTH_LANDMARKS:smooth_landmarks"
node {
  calculator: "ConstantSidePacketCalculator"
  output_side_packet: "PACKET:smooth_landmarks"
  node_options: {
    [type.googleapis.com/mediapipe.ConstantSidePacketCalculatorOptions]: {
      packet { bool_value: true }
    }
  }
}
# Whether to predict the segmentation mask. If unspecified, functions as set to
# false. (bool)
# input_side_packet: "ENABLE_SEGMENTATION:enable_segmentation"
# Generates side packet to enable segmentation.
node {
  calculator: "ConstantSidePacketCalculator"
  output_side_packet: "PACKET:enable_segmentation"
  node_options: {
    [type.googleapis.com/mediapipe.ConstantSidePacketCalculatorOptions]: {
      packet { bool_value: true }
    }
  }
}
# Whether to filter segmentation mask across different input images to reduce
# jitter. If unspecified, functions as set to true. (bool)
# input_side_packet: "SMOOTH_SEGMENTATION:smooth_segmentation"
node {
  calculator: "ConstantSidePacketCalculator"
  output_side_packet: "PACKET:smooth_segmentation"
  node_options: {
    [type.googleapis.com/mediapipe.ConstantSidePacketCalculatorOptions]: {
      packet { bool_value: true }
    }
  }
}
# Complexity of the pose landmark model: 0, 1 or 2. Landmark accuracy as well as
# inference latency generally go up with the model complexity. If unspecified,
# functions as set to 1. (int)
input_side_packet: "MODEL_COMPLEXITY:model_complexity"
# Pose landmarks. (NormalizedLandmarkList)
# We have 33 landmarks (see pose_landmark_topology.svg), and there are other
# auxiliary key points.
# 0 - nose
# 1 - left eye (inner)
# 2 - left eye
# 3 - left eye (outer)
# 4 - right eye (inner)
# 5 - right eye
# 6 - right eye (outer)
# 7 - left ear
# 8 - right ear
# 9 - mouth (left)
# 10 - mouth (right)
# 11 - left shoulder
# 12 - right shoulder
# 13 - left elbow
# 14 - right elbow
# 15 - left wrist
# 16 - right wrist
# 17 - left pinky
# 18 - right pinky
# 19 - left index
# 20 - right index
# 21 - left thumb
# 22 - right thumb
# 23 - left hip
# 24 - right hip
# 25 - left knee
# 26 - right knee
# 27 - left ankle
# 28 - right ankle
# 29 - left heel
# 30 - right heel
# 31 - left foot index
# 32 - right foot index
#
# NOTE: if a pose is not present within the given ROI, for this particular
# timestamp there will not be an output packet in the LANDMARKS stream. However,
# the MediaPipe framework will internally inform the downstream calculators of
# the absence of this packet so that they don't wait for it unnecessarily.
output_stream: "LANDMARKS:pose_landmarks"
# Pose world landmarks. (LandmarkList)
# World landmarks are real-world 3D coordinates in meters with the origin at the
# center between hips. WORLD_LANDMARKS shares the same landmark topology as
# LANDMARKS. However, LANDMARKS provides coordinates (in pixels) of a 3D object
# projected onto the 2D image surface, while WORLD_LANDMARKS provides
# coordinates (in meters) of the 3D object itself.
output_stream: "WORLD_LANDMARKS:pose_world_landmarks"
# Segmentation mask. (ImageFrame in ImageFormat::VEC32F1)
output_stream: "SEGMENTATION_MASK:segmentation_mask"
# Extra outputs (for debugging, for instance).
# Detected poses. (Detection)
output_stream: "DETECTION:pose_detection"
# Regions of interest calculated based on landmarks. (NormalizedRect)
output_stream: "ROI_FROM_LANDMARKS:pose_rect_from_landmarks"
# Regions of interest calculated based on pose detections. (NormalizedRect)
output_stream: "ROI_FROM_DETECTION:pose_rect_from_detection"
# Defines whether landmarks on the previous image should be used to help
# localize landmarks on the current image.
node {
  name: "ConstantSidePacketCalculator"
  calculator: "ConstantSidePacketCalculator"
  output_side_packet: "PACKET:use_prev_landmarks"
  options: {
    [mediapipe.ConstantSidePacketCalculatorOptions.ext]: {
      packet { bool_value: true }
    }
  }
}
node {
  calculator: "GateCalculator"
  input_side_packet: "ALLOW:use_prev_landmarks"
  input_stream: "prev_pose_rect_from_landmarks"
  output_stream: "gated_prev_pose_rect_from_landmarks"
}
# Checks if there's previous pose rect calculated from landmarks.
node: {
  calculator: "PacketPresenceCalculator"
  input_stream: "PACKET:gated_prev_pose_rect_from_landmarks"
  output_stream: "PRESENCE:prev_pose_rect_from_landmarks_is_present"
}
# Calculates size of the image.
node {
  calculator: "ImagePropertiesCalculator"
  input_stream: "IMAGE_CPU:image"
  output_stream: "SIZE:image_size"
}
# Drops the incoming image if the pose has already been identified from the
# previous image. Otherwise, passes the incoming image through to trigger a new
# round of pose detection.
node {
  calculator: "GateCalculator"
  input_stream: "image"
  input_stream: "image_size"
  input_stream: "DISALLOW:prev_pose_rect_from_landmarks_is_present"
  output_stream: "image_for_pose_detection"
  output_stream: "image_size_for_pose_detection"
  options: {
    [mediapipe.GateCalculatorOptions.ext] {
      empty_packets_as_allow: true
    }
  }
}
# Detects poses.
node {
  calculator: "PoseDetectionCpu"
  input_stream: "IMAGE:image_for_pose_detection"
  output_stream: "DETECTIONS:pose_detections"
}
# Gets the very first detection from "pose_detections" vector.
node {
  calculator: "SplitDetectionVectorCalculator"
  input_stream: "pose_detections"
  output_stream: "pose_detection"
  options: {
    [mediapipe.SplitVectorCalculatorOptions.ext] {
      ranges: { begin: 0 end: 1 }
      element_only: true
    }
  }
}
# Calculates region of interest based on pose detection, so that can be used
# to detect landmarks.
node {
  calculator: "PoseDetectionToRoi"
  input_stream: "DETECTION:pose_detection"
  input_stream: "IMAGE_SIZE:image_size_for_pose_detection"
  output_stream: "ROI:pose_rect_from_detection"
}
# Selects either pose rect (or ROI) calculated from detection or from previously
# detected landmarks if available (in this case, calculation of pose rect from
# detection is skipped).
node {
  calculator: "MergeCalculator"
  input_stream: "pose_rect_from_detection"
  input_stream: "gated_prev_pose_rect_from_landmarks"
  output_stream: "pose_rect"
}
# Detects pose landmarks within specified region of interest of the image.
node {
  calculator: "PoseLandmarkByRoiCpu"
  input_side_packet: "MODEL_COMPLEXITY:model_complexity"
  input_side_packet: "ENABLE_SEGMENTATION:enable_segmentation"
  input_stream: "IMAGE:image"
  input_stream: "ROI:pose_rect"
  output_stream: "LANDMARKS:unfiltered_pose_landmarks"
  output_stream: "AUXILIARY_LANDMARKS:unfiltered_auxiliary_landmarks"
  output_stream: "WORLD_LANDMARKS:unfiltered_world_landmarks"
  output_stream: "SEGMENTATION_MASK:unfiltered_segmentation_mask"
}
# Smoothes landmarks to reduce jitter.
node {
  calculator: "PoseLandmarkFiltering"
  input_side_packet: "ENABLE:smooth_landmarks"
  input_stream: "IMAGE_SIZE:image_size"
  input_stream: "NORM_LANDMARKS:unfiltered_pose_landmarks"
  input_stream: "AUX_NORM_LANDMARKS:unfiltered_auxiliary_landmarks"
  input_stream: "WORLD_LANDMARKS:unfiltered_world_landmarks"
  output_stream: "FILTERED_NORM_LANDMARKS:pose_landmarks"
  output_stream: "FILTERED_AUX_NORM_LANDMARKS:auxiliary_landmarks"
  output_stream: "FILTERED_WORLD_LANDMARKS:pose_world_landmarks"
}
# Calculates region of interest based on the auxiliary landmarks, to be used in
# the subsequent image.
node {
  calculator: "PoseLandmarksToRoi"
  input_stream: "LANDMARKS:auxiliary_landmarks"
  input_stream: "IMAGE_SIZE:image_size"
  output_stream: "ROI:pose_rect_from_landmarks"
}
# Caches pose rects calculated from landmarks, and upon the arrival of the next
# input image, sends out the cached rects with timestamps replaced by that of
# the input image, essentially generating a packet that carries the previous
# pose rects. Note that upon the arrival of the very first input image, a
# timestamp bound update occurs to jump start the feedback loop.
node {
  calculator: "PreviousLoopbackCalculator"
  input_stream: "MAIN:image"
  input_stream: "LOOP:pose_rect_from_landmarks"
  input_stream_info: {
    tag_index: "LOOP"
    back_edge: true
  }
  output_stream: "PREV_LOOP:prev_pose_rect_from_landmarks"
}
# Smoothes segmentation to reduce jitter.
node {
  calculator: "PoseSegmentationFiltering"
  input_side_packet: "ENABLE:smooth_segmentation"
  input_stream: "SEGMENTATION_MASK:unfiltered_segmentation_mask"
  output_stream: "FILTERED_SEGMENTATION_MASK:filtered_segmentation_mask"
}
# Converts the incoming segmentation mask represented as an Image into the
# corresponding ImageFrame type.
node: {
  calculator: "FromImageCalculator"
  input_stream: "IMAGE:filtered_segmentation_mask"
  output_stream: "IMAGE_CPU:segmentation_mask"
}
      )pb");
}


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
	auto segmentation_mask_sop = graph->AddOutputStreamPoller(kOutputStreamSegmentationMask);
	segmentation_mask_poller = absl::make_unique<mediapipe::OutputStreamPoller>(std::move(segmentation_mask_sop.value()));

  char kOutputStreamLandmarks[] = "pose_world_landmarks";
	auto landmarks_sop = graph->AddOutputStreamPoller(kOutputStreamLandmarks);
	landmarks_poller = absl::make_unique<mediapipe::OutputStreamPoller>(std::move(landmarks_sop.value()));

  char kOutputStreamPoseLandmarks[] = "pose_landmarks";
	auto pose_landmarks_sop = graph->AddOutputStreamPoller(kOutputStreamPoseLandmarks);
	pose_landmarks_poller = absl::make_unique<mediapipe::OutputStreamPoller>(std::move(pose_landmarks_sop.value()));

	MP_RETURN_IF_ERROR(graph->StartRun({}));
	
	return absl::OkStatus();
}

CPPLIBRARY_API int init_pose_tracking(void)
{
	absl::Status status = InitPoseTracking();
	return status.raw_code();
}

absl::Status ProcessPoseTracking(int width, int height, uint8* input_pixel_data, int64 frame_timestamp_us) {
	
  auto input_frame = absl::make_unique<mediapipe::ImageFrame>(mediapipe::ImageFormat::SRGB, width, height, mediapipe::ImageFrame::kDefaultAlignmentBoundary);
	input_frame->CopyPixelData(mediapipe::ImageFormat::SRGB, width, height, input_pixel_data, mediapipe::ImageFrame::kDefaultAlignmentBoundary);

	// Send image packet into the graph.
	char kInputStream[] = "image";
  MP_RETURN_IF_ERROR(graph->AddPacketToInputStream(kInputStream, mediapipe::Adopt(input_frame.release()).At(mediapipe::Timestamp(frame_timestamp_us))));

	return absl::OkStatus();
}

absl::Status ProcessPoseTrackingSRGBA(int width, int height, uint8* input_pixel_data, int64 frame_timestamp_us) {
	
  auto input_frame = absl::make_unique<mediapipe::ImageFrame>(mediapipe::ImageFormat::SRGBA, width, height, mediapipe::ImageFrame::kDefaultAlignmentBoundary);
	input_frame->CopyPixelData(mediapipe::ImageFormat::SRGBA, width, height, input_pixel_data, mediapipe::ImageFrame::kDefaultAlignmentBoundary);

	// Send image packet into the graph.
	char kInputStream[] = "image";
  MP_RETURN_IF_ERROR(graph->AddPacketToInputStream(kInputStream, mediapipe::Adopt(input_frame.release()).At(mediapipe::Timestamp(frame_timestamp_us))));

	return absl::OkStatus();
}

CPPLIBRARY_API int process_pose_tracking(int width, int height, uint8* input_pixel_data, int64 frame_timestamp_us, int rgba)
{
  if (rgba == 1) {
	  absl::Status status = ProcessPoseTrackingSRGBA(width, height, input_pixel_data, frame_timestamp_us);
	  return status.raw_code();
  } else {
	  absl::Status status = ProcessPoseTracking(width, height, input_pixel_data, frame_timestamp_us);
	  return status.raw_code();
  }
}

absl::Status GetSegmentationMask() {
  // Get the graph result packet.
  mediapipe::Packet packet;
  if (segmentation_mask_poller->QueueSize() == 0) {
		return absl::UnavailableError("segmentation_mask_poller->QueueSize() is 0.");
	}
  if (!segmentation_mask_poller->Next(&packet)) {
		return absl::UnavailableError("Could not get segmentation_mask.");
	}
	
	auto& output_frame = packet.Get<mediapipe::ImageFrame>();
	segmentation_mask->CopyFrom(output_frame, mediapipe::ImageFrame::kDefaultAlignmentBoundary);

	return absl::OkStatus();
}

CPPLIBRARY_API int get_segmentation_mask(int width, int height, float* output_segmentation_mask) {
  absl::Status status = GetSegmentationMask();

  if (status.raw_code() == 0) {
    int segmentation_mask_size = width * height; // VEC32F1
    if (segmentation_mask && segmentation_mask->Width() == width && segmentation_mask->Height() == height) {
      segmentation_mask->CopyToBuffer(output_segmentation_mask, segmentation_mask_size);
      return 0;
    } else {
	    return -1;
	  }
  } else {
    return status.raw_code();
  }
}

absl::Status GetLandmarks() {
  // Get the graph result packet.
  mediapipe::Packet packet;
  if (landmarks_poller->QueueSize() == 0) {
		return absl::UnavailableError("landmarks_poller->QueueSize() is 0");
	}
  if (!landmarks_poller->Next(&packet)) {
		return absl::UnavailableError("Could not get landmarks.");
	}
	
	auto& landmark_list = packet.Get<mediapipe::LandmarkList>();
  landmarks = absl::make_unique<mediapipe::LandmarkList>(landmark_list);

	return absl::OkStatus();
}

CPPLIBRARY_API int get_landmarks(float* x_array, float* y_array, float* z_array, float* visibilities, float* presences, int size) {
  absl::Status status = GetLandmarks();

  if (status.raw_code() == 0) {
    int landmark_size = landmarks->landmark_size();
    if (landmark_size == size) {
      for (int i = 0; i < landmark_size; i++) {
        x_array[i] = landmarks->landmark(i).x();
        y_array[i] = landmarks->landmark(i).y();
        z_array[i] = landmarks->landmark(i).z();
        visibilities[i] = landmarks->landmark(i).visibility();
        presences[i] = landmarks->landmark(i).presence();
      }
      return 0;
    } else {
	    return (landmark_size + 1) * -1;
	  }
  } else {
    return status.raw_code();
  }
}

absl::Status GetPoseLandmarks() {
  // Get the graph result packet.
  mediapipe::Packet packet;
  if (pose_landmarks_poller->QueueSize() == 0) {
		return absl::UnavailableError("pose_landmarks_poller->QueueSize() is 0");
	}
  if (!pose_landmarks_poller->Next(&packet)) {
		return absl::UnavailableError("Could not get pose landmarks.");
	}
	
	auto& landmark_list = packet.Get<mediapipe::NormalizedLandmarkList>();
  pose_landmarks = absl::make_unique<mediapipe::NormalizedLandmarkList>(landmark_list);

	return absl::OkStatus();
}

CPPLIBRARY_API int get_pose_landmarks(float* x_array, float* y_array, float* z_array, float* visibilities, float* presences, int size) {
  absl::Status status = GetPoseLandmarks();

  if (status.raw_code() == 0) {
    int landmark_size = pose_landmarks->landmark_size();
    if (landmark_size == size) {
      for (int i = 0; i < landmark_size; i++) {
        x_array[i] = pose_landmarks->landmark(i).x();
        y_array[i] = pose_landmarks->landmark(i).y();
        z_array[i] = pose_landmarks->landmark(i).z();
        visibilities[i] = pose_landmarks->landmark(i).visibility();
        presences[i] = pose_landmarks->landmark(i).presence();
      }
      return 0;
    } else {
	    return (landmark_size + 1) * -1;
	  }
  } else {
    return status.raw_code();
  }
}


CPPLIBRARY_API int apply_segmentation_mask(int width, int height, uint8* rgba_pixel_data, float* segmentation_mask, float threshold) {
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = y * width + x;
      if (segmentation_mask[idx] <= threshold) {
        rgba_pixel_data[idx * 4 + 3] = 0; // Alpha channel
      }
    }
  }
  return 0;
}

CPPLIBRARY_API void set_custom_global_resource_provider(ResourceProvider* resource_provider) {
  mediapipe::SetCustomGlobalResourceProvider([resource_provider](const std::string& path, std::string* output) -> ::absl::Status {
    if (resource_provider(path.c_str(), output)) {
      return absl::OkStatus();
    }
    return absl::FailedPreconditionError(absl::StrCat("Failed to read ", path));
  });
}

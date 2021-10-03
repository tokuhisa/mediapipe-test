#include "mppose.h"

#include "mediapipe/util/resource_util_custom.h"
#include "mediapipe/framework/deps/ret_check.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"

std::map<int, std::unique_ptr<mediapipe::CalculatorGraph>> graph_data; // key=graph_id
std::map<int, int> input_graph_ids; // key=input_id, value=graph_id
std::map<int, std::string> input_names; // key=input_id
std::map<int, int> output_ids; // key=graph_id, value=input_id
std::map<int, std::unique_ptr<mediapipe::OutputStreamPoller>> output_streams; // key=output_id


absl::Status InitializeGraphMp(int graph_id, char* data, int data_size) {
  LOG(INFO) << "Initialize the calculator graph.";
  graph_data[graph_id] = absl::make_unique<::mediapipe::CalculatorGraph>();
  MP_RETURN_IF_ERROR(graph_data[graph_id]->Initialize(mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(std::string(data, data_size))));
  return absl::OkStatus();
}

CPPLIBRARY_API int InitializeGraph(int graph_id, char* data, int data_size) {
  auto status = InitializeGraphMp(graph_id, data, data_size);
  return status.raw_code();
}

CPPLIBRARY_API int OpenInputStream(int graph_id, int input_id, char* name, int name_length) {
  input_graph_ids[input_id] = graph_id;
  input_names[input_id] = std::string(name, name_length);
  return 0;
}

CPPLIBRARY_API int OpenOutputStream(int graph_id, int output_id, char* name, int name_length) {
  auto status_or_poller = graph_data[graph_id]->AddOutputStreamPoller(std::string(name, name_length));
  if (status_or_poller.ok()) {
    output_streams[output_id] = absl::make_unique<mediapipe::OutputStreamPoller>(std::move(status_or_poller.value()));
  }
  return status_or_poller.status().raw_code();
}

absl::Status RunGraphMp(int graph_id) {
  MP_RETURN_IF_ERROR(graph_data[graph_id]->StartRun({}));
  return absl::OkStatus();
}

CPPLIBRARY_API int RunGraph(int graph_id) {
  auto status = RunGraphMp(graph_id);
  return status.raw_code();
}

absl::Status WriteRGBAImageFrameMp(int input_id, int width, int height, uint8* data, int64 frame_timestamp_us) {
  auto input_frame = absl::make_unique<mediapipe::ImageFrame>(mediapipe::ImageFormat::SRGBA, width, height, mediapipe::ImageFrame::kDefaultAlignmentBoundary);
  input_frame->CopyPixelData(mediapipe::ImageFormat::SRGBA, width, height, data, mediapipe::ImageFrame::kDefaultAlignmentBoundary);
  auto graph_id = input_graph_ids[input_id];
  auto input_name = input_names[input_id];

  // Send image packet into the graph.
  MP_RETURN_IF_ERROR(graph_data[graph_id]->AddPacketToInputStream(input_name, mediapipe::Adopt(input_frame.release()).At(mediapipe::Timestamp(frame_timestamp_us))));

  return absl::OkStatus();
}

CPPLIBRARY_API int WriteRGBAImageFrame(int input_id, int width, int height, uint8* data, int64 frame_timestamp_us) {
  auto status = WriteRGBAImageFrameMp(input_id, width, height, data, frame_timestamp_us);
  return status.raw_code();
}

absl::Status ReadVEC32F1ImageFrameMp(int output_id, int width, int height, float* image_data) {
  // Get the graph result packet.
  mediapipe::Packet packet;
  if (output_streams[output_id]->QueueSize() == 0) {
    return absl::UnavailableError("poller->QueueSize() is 0.");
  }
  if (!output_streams[output_id]->Next(&packet)) {
    return absl::UnavailableError("Could not get packet.");
  }
  
  auto& output_frame = packet.Get<mediapipe::ImageFrame>();
  int data_size = width * height; // VEC32F1
  if (output_frame.Width() == width && output_frame.Height() == height) {
    output_frame.CopyToBuffer(image_data, data_size);
    return absl::OkStatus();
  } else {
    return absl::UnavailableError("Unavailable data size.");
  }
}

CPPLIBRARY_API int ReadVEC32F1ImageFrame(int output_id, int width, int height, float* image_data) {
  auto status = ReadVEC32F1ImageFrameMp(output_id, width, height, image_data);
  return status.raw_code();
}

absl::Status ReadLandmarkListMp(int output_id, int landmark_size, float* landmark_data) {
  // Get the graph result packet.
  mediapipe::Packet packet;
  if (output_streams[output_id]->QueueSize() == 0) {
    return absl::UnavailableError("poller->QueueSize() is 0.");
  }
  if (!output_streams[output_id]->Next(&packet)) {
    return absl::UnavailableError("Could not get packet.");
  }

  auto& landmark_list = packet.Get<mediapipe::LandmarkList>();
  if (landmark_list.landmark_size() == landmark_size) {
    for (int i = 0; i < landmark_size; i++) {
      landmark_data[i * 5] = landmark_list.landmark(i).x();
      landmark_data[i * 5 + 1] = landmark_list.landmark(i).y();
      landmark_data[i * 5 + 2] = landmark_list.landmark(i).z();
      landmark_data[i * 5 + 3] = landmark_list.landmark(i).visibility();
      landmark_data[i * 5 + 4] = landmark_list.landmark(i).presence();
    }
    return absl::OkStatus();
  } else {
    return absl::UnavailableError("Unavailable data size.");
  }
}

CPPLIBRARY_API int ReadLandmarkList(int output_id, int landmark_size, float* landmark_data) {
  auto status = ReadLandmarkListMp(output_id, landmark_size, landmark_data);
  return status.raw_code();
}

absl::Status ReadNormalizedLandmarkListMp(int output_id, int landmark_size, float* landmark_data) {
  // Get the graph result packet.
  mediapipe::Packet packet;
  if (output_streams[output_id]->QueueSize() == 0) {
    return absl::UnavailableError("poller->QueueSize() is 0.");
  }
  if (!output_streams[output_id]->Next(&packet)) {
    return absl::UnavailableError("Could not get packet.");
  }

  auto& landmark_list = packet.Get<mediapipe::NormalizedLandmarkList>();
  if (landmark_list.landmark_size() == landmark_size) {
    for (int i = 0; i < landmark_size; i++) {
      landmark_data[i * 5] = landmark_list.landmark(i).x();
      landmark_data[i * 5 + 1] = landmark_list.landmark(i).y();
      landmark_data[i * 5 + 2] = landmark_list.landmark(i).z();
      landmark_data[i * 5 + 3] = landmark_list.landmark(i).visibility();
      landmark_data[i * 5 + 4] = landmark_list.landmark(i).presence();
    }
    return absl::OkStatus();
  } else {
    return absl::UnavailableError("Unavailable data size.");
  }
}

CPPLIBRARY_API int ReadNormalizedLandmarkList(int output_id, int landmark_size, float* landmark_data) {
  auto status = ReadNormalizedLandmarkListMp(output_id, landmark_size, landmark_data);
  return status.raw_code();
}

absl::Status ReadNormalizedLandmarkListCollectionMp(int output_id, int collection_size, int landmark_size, float* landmark_data, int* num_of_detections) {
  // Get the graph result packet.
  mediapipe::Packet packet;
  if (output_streams[output_id]->QueueSize() == 0) {
    return absl::UnavailableError("poller->QueueSize() is 0.");
  }
  if (!output_streams[output_id]->Next(&packet)) {
    return absl::UnavailableError("Could not get packet.");
  }

  auto& landmarks_collection = packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();
  auto total_size = landmarks_collection.size();
  *num_of_detections = total_size;
  auto size = collection_size > total_size ? total_size : collection_size;
  for (int idx = 0; idx < size; idx++) {
    auto landmark_list = landmarks_collection[idx];
    if (landmark_list.landmark_size() != landmark_size) {
      return absl::UnavailableError("Unavailable data size.");
    }
    for (int i = 0; i < landmark_size; i++) {
      landmark_data[idx * landmark_size * 5 + i * 5] = landmark_list.landmark(i).x();
      landmark_data[idx * landmark_size * 5 + i * 5 + 1] = landmark_list.landmark(i).y();
      landmark_data[idx * landmark_size * 5 + i * 5 + 2] = landmark_list.landmark(i).z();
      landmark_data[idx * landmark_size * 5 + i * 5 + 3] = landmark_list.landmark(i).visibility();
      landmark_data[idx * landmark_size * 5 + i * 5 + 4] = landmark_list.landmark(i).presence();
    }
  }
  return absl::OkStatus();
}

CPPLIBRARY_API int ReadNormalizedLandmarkListCollection(int output_id, int collection_size, int landmark_size, float* landmark_data, int* num_of_detections) {
  auto status = ReadNormalizedLandmarkListCollectionMp(output_id, collection_size, landmark_size, landmark_data, num_of_detections);
  return status.raw_code();
}

absl::Status ReadClassificationListCollectionMp(int output_id, int collection_size, int classification_list_size, int32* index_data, float* score_data, int* num_of_detections) {
  // Get the graph result packet.
  mediapipe::Packet packet;
  if (output_streams[output_id]->QueueSize() == 0) {
    return absl::UnavailableError("poller->QueueSize() is 0.");
  }
  if (!output_streams[output_id]->Next(&packet)) {
    return absl::UnavailableError("Could not get packet.");
  }

  auto& classifications_collection = packet.Get<std::vector<mediapipe::ClassificationList>>();
  auto total_size = classifications_collection.size();
  *num_of_detections = total_size;
  auto size = collection_size > total_size ? total_size : collection_size;
  for (int idx = 0; idx < size; idx++) {
    auto classification_list = classifications_collection[idx];
    if (classification_list.classification_size() != classification_list_size) {
      return absl::UnavailableError("Unavailable data size.");
    }
    for (int i = 0; i < classification_list_size; i++) {
      index_data[idx * classification_list_size + i] = classification_list_size.classification(i).index();
      score_data[idx * classification_list_size + i] = classification_list_size.classification(i).score();
    }
  }
  return absl::OkStatus();
}

CPPLIBRARY_API int ReadClassificationListCollection(int output_id, int collection_size, int classification_list_size, int32* index_data, float* score_data, int* num_of_detections) {
  auto status = ReadClassificationListCollectionMp(output_id, collection_size, classification_list_size, index_data, score_data, num_of_detections);
  return status.raw_code();
}

absl::Status CloseInputStreamMp(int input_id) {
  auto graph_id = input_graph_ids[input_id];
  auto input_name = input_names[input_id];

  auto itr1 = input_graph_ids.find(input_id);
  if(itr1 != input_graph_ids.end()) {
    input_graph_ids.erase(itr1);
  }
  auto itr2 = input_names.find(input_id);
  if(itr2 != input_names.end()) {
    input_names.erase(itr2);
  }
  
  MP_RETURN_IF_ERROR(graph_data[graph_id]->CloseInputStream(input_name));
  
  return absl::OkStatus();
}

CPPLIBRARY_API int CloseInputStream(int input_id) {
  auto status = CloseInputStreamMp(input_id);
  return status.raw_code();
}

CPPLIBRARY_API int CloseOutputStream(int output_id) {
  auto itr1 = output_streams.find(output_id);
  if(itr1 != output_streams.end()) {
    output_streams.erase(itr1);
  }
  return 0;
}

CPPLIBRARY_API int TerminateGraph(int graph_id) {
  graph_data[graph_id]->WaitUntilDone();

  auto itr1 = graph_data.find(graph_id);
  if(itr1 != graph_data.end()) {
    graph_data.erase(itr1);
  }
  return 0;
}

// Util
CPPLIBRARY_API int ApplyImageMask(int width, int height, uint8* data, float* mask, float threshold, uint8 fg_alpha, uint8 bg_alpha) {
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = y * width + x;
      int alpha_channel_idx = idx * 4 + 3;
      if (mask[idx] >= threshold) {
        data[alpha_channel_idx] = fg_alpha;
      } else {
        data[alpha_channel_idx] = bg_alpha;
      }
    }
  }
  return 0;
}

// Resource Provider
CPPLIBRARY_API void SetCustomResourceProvider(ResourceProvider* resource_provider) {
  mediapipe::SetCustomGlobalResourceProvider([resource_provider](const std::string& path, std::string* output) -> ::absl::Status {
    if (resource_provider(path.c_str(), output)) {
      return absl::OkStatus();
    }
    return absl::FailedPreconditionError(absl::StrCat("Failed to read ", path));
  });
}

CPPLIBRARY_API void SetStringData(char* src_data, int src_data_size, std::string* dst) {
  auto src = std::string(src_data, src_data_size);
  src.swap(*dst);
}

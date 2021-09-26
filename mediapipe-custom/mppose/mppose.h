#pragma once

#define CPPLIBRARY_API __declspec(dllexport) 

#include "mediapipe/framework/deps/safe_int.h"
#include <string>

extern "C"
{
  // Graph
  CPPLIBRARY_API int InitializeGraph(int graph_id, char* data, int data_size);
  CPPLIBRARY_API int OpenInputStream(int graph_id, int input_id, char* name, int name_length);
  CPPLIBRARY_API int OpenOutputStream(int graph_id, int output_id, char* name, int name_length);
  CPPLIBRARY_API int RunGraph(int graph_id);

  CPPLIBRARY_API int WriteRGBAImageFrame(int input_id, int width, int height, uint8* data, int64 frame_timestamp_us);
  CPPLIBRARY_API int ReadVEC32F1ImageFrame(int output_id, int width, int height, float* image_data);
  CPPLIBRARY_API int ReadLandmarkList(int output_id, int landmark_count, float* landmark_data);
  CPPLIBRARY_API int ReadNormalizedLandmarkList(int output_id, int landmark_count, float* landmark_data);

  CPPLIBRARY_API int CloseInputStream(int input_id);
  CPPLIBRARY_API int CloseOutputStream(int output_id);
  CPPLIBRARY_API int TerminateGraph(int graph_id);

  // Util
  CPPLIBRARY_API int ApplyImageMask(int width, int height, uint8* data, float* mask, float threshold, unit8 fg_alpha, uint8 bg_alpha); 

  // Resource Provider
  typedef bool ResourceProvider(const char* path, std::string* output);
  CPPLIBRARY_API void SetCustomResourceProvider(ResourceProvider* resource_provider);
  CPPLIBRARY_API void SetStringData(char* src_data, int src_data_size, std::string* dst);
}
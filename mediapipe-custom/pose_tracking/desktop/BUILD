# Copyright 2020 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

licenses(["notice"])

package(default_visibility = ["//mediapipe/examples:__subpackages__"])

cc_binary(
    name = "pose_tracking_cpu",
    deps = [
        "//mediapipe-custom/pose_tracking/desktop:demo_run_graph_main",
        "//mediapipe/modules/pose_landmark:pose_landmark_cpu",
        "//mediapipe/calculators/core:flow_limiter_calculator",
    ],
)

# Linux only
cc_binary(
    name = "pose_tracking_gpu",
    deps = [
        "//mediapipe/examples/desktop:demo_run_graph_main_gpu",
        "//mediapipe/graphs/pose_tracking:pose_tracking_gpu_deps",
    ],
)

cc_library(
    name = "demo_run_graph_main",
    srcs = ["demo_run_graph_main.cc"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_video",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
    ],
)


filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//mediapipe/examples:__subpackages__"],
)

cc_binary(
    name = "pose_tracking_cpu_lib",
    srcs = ["pose_tracking_cpu_lib.cc",
            "pose_tracking_cpu_lib.h"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:status",
        "//mediapipe/modules/pose_landmark:pose_landmark_cpu",
        "//mediapipe/calculators/core:flow_limiter_calculator",
    ],
    linkshared = 1,
)

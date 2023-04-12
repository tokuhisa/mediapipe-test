package(default_visibility = ["//mediapipe/examples:__subpackages__"])

cc_library(
    name = "mplib",
    srcs = ["mplib.cc"],
    hdrs = ["mplib.h"],
    visibility = ["//visibility:public"],
    copts = ["/MD", "/LD"],
    linkopts = ["/DLL"],
    deps = [
        "//mediapipe/util:resource_util",
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/modules/pose_landmark:pose_landmark_cpu",
        "//mediapipe/modules/hand_landmark:hand_landmark_tracking_cpu",
        "//mediapipe/modules/face_landmark:face_landmark_front_cpu",
        "//mediapipe/modules/holistic_landmark:holistic_landmark_cpu",
        "@com_google_absl//absl/strings",
    ],
)

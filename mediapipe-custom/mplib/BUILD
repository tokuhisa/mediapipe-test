package(default_visibility = ["//mediapipe/examples:__subpackages__"])

cc_binary(
    target_compatible_with = [
        "@platforms//os:windows",
    ],
    name = "mplib.dll",
    deps = [":mplib"],
    linkshared = 1,
)

cc_library(
    target_compatible_with = [
        "@platforms//os:windows",
    ],
    name = "mplib",
    srcs = ["mplib.cc"],
    hdrs = ["mplib.h"],
    visibility = ["//visibility:public"],
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
    defines = ["COMPILING_DLL=1"],
    alwayslink = 1,
)

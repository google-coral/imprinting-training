# Description:
#   Contains files for training and evaluating mobilenet_v1_l2norm models and useful libraries.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "conv2d_l2norm",
    srcs = ["conv2d_l2norm.py"],
    srcs_version = "PY2AND3",
    deps = [
        # "//tensorflow",
    ],
)

py_library(
    name = "mobilenet_v1_l2norm",
    srcs = ["mobilenet_v1_l2norm.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":conv2d_l2norm",
        # "//tensorflow",
        "//models/research/slim:mobilenet_v1",
    ],
)

py_binary(
    name = "mobilenet_v1_l2norm_train",
    srcs = ["mobilenet_v1_l2norm_train.py"],
    python_version = "PY3",
    deps = [
        ":mobilenet_v1_l2norm",
        # "//tensorflow",
        "//models/research/slim:dataset_factory",
        "//models/research/slim:mobilenet_v1",
        "//models/research/slim:preprocessing_factory",
    ],
)

py_binary(
    name = "mobilenet_v1_l2norm_eval",
    srcs = ["mobilenet_v1_l2norm_eval.py"],
    python_version = "PY3",
    deps = [
        ":mobilenet_v1_l2norm",
        # "//tensorflow",
        "//models/research/slim:dataset_factory",
        "//models/research/slim:mobilenet_v1",
        "//models/research/slim:preprocessing_factory",
    ],
)

py_binary(
    name = "export_inference_graph_l2norm",
    srcs = ["export_inference_graph_l2norm.py"],
    # paropts = ["--compress"],
    python_version = "PY3",
    deps = [":export_inference_graph_l2norm_lib"],
)

py_library(
    name = "export_inference_graph_l2norm_lib",
    srcs = ["export_inference_graph_l2norm.py"],
    deps = [
        ":mobilenet_v1_l2norm",
        # "//tensorflow",
        "//models/research/slim:mobilenet_v1",
    ],
)

py_test(
    name = "export_inference_graph_l2norm_test",
    size = "medium",
    srcs = ["export_inference_graph_l2norm_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    tags = [
        "manual",
    ],
    deps = [
        ":export_inference_graph_l2norm_lib",
        # "//tensorflow",
    ],
)

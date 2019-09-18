# Copyright 2017-2018 Intel Corporation.
load("//bzl:plaidml.bzl", "plaidml_cc_library")

plaidml_cc_library(
    name = "util",
    srcs = [
        "features.cc",
        "tile_file.cc",
        "train_status.cc",
        "xgb_invoker.cc",
    ],
    hdrs = [
        "features.h",
        "tile_file.h",
        "train_const.h",
        "train_status.h",
        "xgb_invoker.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//base/util",
        "//tile/lang",
        "//tile/proto:metadata_cc",
        "//tile/proto:proto_cc",
        "@boost",
        "@boost//:filesystem",
        "//external:xgboost",
    ],
)

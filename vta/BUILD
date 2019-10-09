# Copyright 2018, Intel Corp.

load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_proto_library")

plaidml_cc_library(
    name = "src",
    srcs = [
        "src/sim/sim_driver.cc",
        "src/runtime.cc",
        "src/vta_api.cc"
        ],
    hdrs = [
        "include/vta/vta_api.h",
        "include/vta/driver.h",
        "include/vta/hw_spec.h",
        "include/vta/runtime.h",
        "3rdparty/dmlc-core/include/dmlc/logging.h",
        "3rdparty/dmlc-core/include/dmlc/base.h",
        "3rdparty/dmlc-core/include/dmlc/build_config_default.h",
        ],
    visibility = ["//visibility:public"],
)

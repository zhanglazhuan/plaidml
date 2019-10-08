# Copyright 2018, Intel Corp.

load("//bzl:plaidml.bzl", "plaidml_cc_library", "plaidml_proto_library")

plaidml_cc_library(
    name = "src",
    srcs = [
        "src/sim/sim_driver.cc",
        "src/runtime.cc",
        ],
    hdrs = [
        "include/vta/driver.h",
        "include/vta/hw_spec.h",
        "include/vta/runtime.h",
        "3rdparty/dmlc-core/include/dmlc/logging.h",
        "3rdparty/dmlc-core/include/dmlc/base.h",
        "3rdparty/dmlc-core/include/dmlc/build_config_default.h",
        ],
    visibility = ["//visibility:public"],
)
'''
plaidml_cc_library(
    name = "test",
    srcs = [
        "tests/hardware/common/test_lib.cc",
        "tests/hardware/metal_test/metal_test.cc",
        ],
    hdrs = [
        "include/vta/driver.h",
        "include/vta/hw_spec.h",
        "include/vta/runtime.h",
        "3rdparty/dmlc-core/include/dmlc/logging.h",
        "3rdparty/dmlc-core/include/dmlc/base.h",
        "3rdparty/dmlc-core/include/dmlc/build_config_default.h",
        "tests/hardware/common/test_lib.h",
        ],
    visibility = ["//visibility:public"],
)
'''

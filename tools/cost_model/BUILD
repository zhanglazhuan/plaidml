package(default_visibility = ["//visibility:public"])

load("@rules_python//python:defs.bzl", "py_test", "py_binary")
load("//tools/py_setup:build_defs.bzl", "py_setup")
load("//bzl:plaidml.bzl", "plaidml_py_library")

py_binary(
    name = "generator",
    srcs = [
        "constants.py",
        "generator.py",
    ],
    tags = [
    ],
    deps = [
        "//plaidml/keras:keras",
    ],
)

py_binary(
    name = "preprocess",
    srcs = [
        "constants.py",
        "preprocess.py",
    ],
    tags = [
    ],
    deps = [
    ],
)

py_binary(
    name = "train",
    srcs = [
        "constants.py",
        "feature.py",
        "perf_learner.py",
        "train.py",
    ],
    tags = [
    ],
    deps = [
    ],
)

py_binary(
    name = "test",
    srcs = [
        "constants.py",
        "feature.py",
        "perf_learner.py",
        "test.py",
    ],
    tags = [
    ],
    deps = [
    ],
)

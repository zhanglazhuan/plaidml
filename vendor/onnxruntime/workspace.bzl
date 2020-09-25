# Copyright 2020 Intel Corporation

load("//vendor/bazel:repo.bzl", "http_archive")

def onnxruntime_workspace():
    # TODO: add dependencies
    http_archive(
        name = "onnxruntime",
        url = "https://github.com/PlaidML/onnxruntime/archive/4228d22ea11448d2e38f8a2dfa450d9f7edf5f0d.zip",
        strip_prefix = "onnxruntime-4228d22ea11448d2e38f8a2dfa450d9f7edf5f0d",
        sha256 = "",
        build_file = clean_dep("//vendor/onnxruntime:onnxruntime.BUILD"),
    )

// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/range.hpp"

using LayerTestsDefinitions::RangeLayerTest;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,  //
};

INSTANTIATE_TEST_CASE_P(RangeCheck, RangeLayerTest,
                        // ::testing::Combine(::testing::Values(-1, 0, 4),                                      //
                        //                    ::testing::Values(1, 4.5, 15),                             //
                        //                    ::testing::Values(1, 3, -0.5),                             //
                        ::testing::Combine(::testing::Values(-1),                                //
                                           ::testing::Values(7),                                 //
                                           ::testing::Values(0.5),                               //
                                           ::testing::ValuesIn(netPrecisions),                   //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),  //
                        RangeLayerTest::getTestCaseName);
}  // namespace

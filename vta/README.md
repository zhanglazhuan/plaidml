<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

VTA: Open, Modular, Deep Learning Accelerator Stack
===================================================
VTA (versatile tensor accelerator) is an open-source deep learning accelerator complemented with an end-to-end TVM-based compiler stack.

The key features of VTA include:

- Generic, modular, open-source hardware
  - Streamlined workflow to deploy to FPGAs.
  - Simulator support to prototype compilation passes on regular workstations.
- Driver and JIT runtime for both simulator and FPGA hardware back-end.
- End-to-end TVM stack integration
  - Direct optimization and deployment of models from deep learning frameworks via TVM.
  - Customized and extensible TVM compiler back-end.
  - Flexible RPC support to ease deployment, and program FPGAs with the convenience of Python.

Learn more about VTA [here](https://docs.tvm.ai/vta/index.html).

To use the GEMM functionality proposed by VTA, please cheack vta_api.cc and vta_api.h.

Usage:
Make sure matrices for GEMM follow the VTA computation rules:
Input matrix: (X, K)
Weight matrix: (Y, K)

Result matrix: (X, Y)

VTA will do the reduction along the K axis.


For matrices (X, K) and (K, Y), transpose the second matrix to (Y, K).
Then we can do GEMM using VTA.

Please check the testing file test_mm.cpp for more VTA computation information.
For reference, please check the python file test_mm.py.

Step:
1. Compile: make
2. VTA: ./tests/test_mm
3. Reference: python ./tests/test_mm.py



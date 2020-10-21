// RUN: pmlc-opt -convert-linalg-to-loops -x86-convert-pxa-to-affine -lower-affine \
// RUN:     -canonicalize -convert-scf-to-std -x86-convert-std-to-llvm %s | \
// RUN:   pmlc-jit -e baseline 

// RUN: pmlc-opt -convert-linalg-to-loops -x86-affine-stencil-xsmm -x86-convert-pxa-to-affine -lower-affine \
// RUN:     -canonicalize -convert-scf-to-std -x86-convert-std-to-llvm %s | \
// RUN:   pmlc-jit -e baseline 


// Command lines:
// bazel-bin/pmlc/opt -convert-linalg-to-loops -x86-convert-pxa-to-affine -lower-affine  -canonicalize -convert-scf-to-std -x86-convert-std-to-llvm pmlc/target/x86/tests/conv_NCHW_transformed.mlir | bazel-bin/pmlc/jit -e baseline

// bazel-bin/pmlc/opt -convert-linalg-to-loops -x86-affine-stencil-xsmm -x86-convert-pxa-to-affine -lower-affine  -canonicalize -convert-scf-to-std -x86-convert-std-to-llvm pmlc/target/x86/tests/conv_NCHW_transformed.mlir | bazel-bin/pmlc/jit -e baseline


!orig_I_memref = type memref<1x64x56x56xf32> // NCHW -> NHWC
!orig_K_memref = type memref<64x64x1x1xf32>  // nofm-nifm-kh-kw -> kh-kw-nofm-nifm
!orig_O_memref = type memref<1x64x56x56xf32> // N-nofm-H-W -> N-nofm/c-H-W-c -> N-H-W-nofm

!mod_I_memref = type memref<1x56x56x64xf32> // NCHW -> NHWC
!mod_K_memref = type memref<1x1x64x64xf32>  // nofm-nifm-kh-kw -> kh-kw-nofm-nifm
!mod_O_memref = type memref<1x56x56x64xf32> // N-nofm-H-W -> N-nofm/c-H-W-c -> N-H-W-nofm


func @print_memref_f32(memref<*xf32>)

func @baseline() {
  %conv2 = constant @conv2 : (!mod_I_memref, !mod_K_memref, !mod_O_memref) -> ()
  call @test_conv2(%conv2) : ((!mod_I_memref, !mod_K_memref, !mod_O_memref) -> ()) -> ()

  return
}

func @test_conv2(%impl : (!mod_I_memref, !mod_K_memref, !mod_O_memref) -> ()) {
  %false = constant 0 : i1
  %true = constant 1 : i1
  %f0 = constant 0.0 : f32
  %f1 = constant 1.0 : f32
  %f2 = constant 2.0 : f32

  %orig_I = alloc() : !orig_I_memref
  %orig_K = alloc() : !orig_K_memref
  %orig_O = alloc() : !orig_O_memref

  linalg.fill(%orig_O, %f0) : !orig_O_memref, f32
  linalg.fill(%orig_I, %f1) : !orig_I_memref, f32
  linalg.fill(%orig_K, %f2) : !orig_K_memref, f32

  %mod_I = alloc() : !mod_I_memref
  %mod_K = alloc() : !mod_K_memref
  %mod_O = alloc() : !mod_O_memref

  call @copyI(%orig_I, %mod_I) : (!orig_I_memref, !mod_I_memref) -> ()
  call @copyK(%orig_K, %mod_K) : (!orig_K_memref, !mod_K_memref) -> ()
  call @copyO(%orig_O, %mod_O) : (!orig_O_memref, !mod_O_memref) -> ()

  call_indirect %impl(%mod_I, %mod_K, %mod_O) : (!mod_I_memref, !mod_K_memref, !mod_O_memref) -> ()

  %O_ud = memref_cast %mod_O : !mod_O_memref to memref<*xf32>
  call @print_memref_f32(%O_ud) : (memref<*xf32>) -> ()

  dealloc %orig_O : !orig_O_memref
  dealloc %orig_K : !orig_K_memref
  dealloc %orig_I : !orig_I_memref

  dealloc %mod_O : !mod_O_memref
  dealloc %mod_K : !mod_K_memref
  dealloc %mod_I : !mod_I_memref

  return
}

func @conv2(%I: !mod_I_memref, %K: !mod_K_memref, %O: !mod_O_memref) {
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c3 = constant 3 : index
  %X = dim %I, %c1 : !mod_I_memref
  %Y = dim %I, %c2 : !mod_I_memref
  %CI = dim %I, %c3 : !mod_I_memref
  %CO = dim %O, %c3 : !mod_O_memref
  affine.parallel (%x, %y, %ci, %co) = (0, 0, 0, 0) to (56, 56, 64, 64) reduce ("assign") -> (memref<1x56x56x64xf32>) {
    %0 = pxa.load %I[0, %x, %y, %ci] : !mod_I_memref
    %1 = pxa.load %K[0, 0, %ci, %co] : !mod_K_memref
    %2 = mulf %0, %1 : f32
    %3 = pxa.reduce addf %2, %O[0, %x, %y, %co] : !mod_O_memref
    affine.yield %3 : memref<1x56x56x64xf32>
  }
  return
}

// NCHW -> NHWC
func @copyI(%origI: !orig_I_memref, %modI: !mod_I_memref) {
  affine.parallel (%x, %y, %ci) = (0, 0, 0) to (56, 56, 64) {
    %0 = pxa.load %origI[0, %ci, %x, %y] : !orig_I_memref
    affine.store %0, %modI[0, %x, %y, %ci] : !mod_I_memref 
  }

  return 
}

// nifm-nofm-kh-kw -> kh-kw-nifm-nofm
func @copyK(%origK: !orig_K_memref, %modK: !mod_K_memref) {
  affine.parallel (%kh, %kw, %ci, %co) = (0, 0, 0, 0) to (1, 1, 64, 64) {
    %0 = pxa.load %origK[%ci, %co, 0, 0] : !orig_K_memref
    affine.store %0, %modK[0, 0, %ci, %co] : !mod_K_memref
  }

  return
}

// NCHW -> NHWC
func @copyO(%origO: !orig_O_memref, %modO: !mod_O_memref) {
  affine.parallel (%x, %y, %co) = (0, 0, 0) to (56, 56, 64) {
    %0 = pxa.load %origO[0, %co, %x, %y] : !orig_O_memref
    affine.store %0, %modO[0, %x, %y, %co] : !mod_O_memref
  }

  return
}


// Command line:  bazel-bin/pmlc/opt -convert-linalg-to-loops --normalize-memrefs --simplify-affine-structures -x86-affine-stencil-xsmm -x86-convert-pxa-to-affine -lower-affine  -canonicalize -convert-scf-to-std -x86-convert-std-to-llvm pmlc/target/x86/tests/conv_NCHW_3X3_user_layouts.mlir | bazel-bin/pmlc/jit -e baseline

#K_map = affine_map<(K,C,R,S) -> (R, S, C, K)>
#NCHW_to_NHWC = affine_map<(N,C,H,W) -> (N,H,W,C)>


// If no user data layout maps are specified the code works just fine
// !I_memref = type memref<1x64x56x56xf32> 
// !K_memref = type memref<64x64x3x3xf32>  
!O_memref = type memref<1x56x56x64xf32> 

// FIXME: When the user data layout maps are specified, the code does not work
 !I_memref = type memref<1x64x56x56xf32, #NCHW_to_NHWC>
 !K_memref = type memref<64x64x3x3xf32, #K_map>


func @print_memref_f32(memref<*xf32>)

func @baseline() {
  %f0 = constant 0.0 : f32
  %f1 = constant 1.0 : f32
  %f2 = constant 2.0 : f32

  %I = alloc() : !I_memref
  %K = alloc() : !K_memref
  %O = alloc() : !O_memref

    linalg.fill(%O, %f0) : !O_memref, f32

  affine.parallel (%x, %y, %ci) = (0, 0, 0) to (56, 56, 64) {
    %ar1 = addi %x, %y : index
    %ar2 = addi %ar1, %ci : index
    %ar2_1 = subi %ar2, %x : index
    %ar3 = index_cast %ar2_1 : index to i32
    %ar4 = sitofp %ar3 : i32 to f32
    affine.store %ar4, %I[0, %ci, %x, %y] : !I_memref
  }

  affine.parallel (%kh, %kw, %ci, %co) = (0, 0, 0, 0) to (1, 1, 64, 64) {
    %ar1 = addi %kh, %kw : index
    %ar2 = addi %ar1, %ci : index
    %ar3 = addi %ar2, %co : index
    %ar4 = index_cast %ar3 : index to i32
    %ar5 = sitofp %ar4 : i32 to f32
    affine.store %ar5, %K[%ci, %co, 0, 0] : !K_memref
  }

  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c3 = constant 3 : index
  %X = dim %I, %c2 : !I_memref
  %Y = dim %I, %c3 : !I_memref
  %CI = dim %I, %c1 : !I_memref
  %CO = dim %O, %c1 : !O_memref

// Code block 1
//  affine.parallel (%x, %y, %ci, %co, %kh, %kw) = (0, 0, 0, 0, 0, 0) to (56, 56, 64, 64, 3, 3) { 
//    %0 = affine.load %I[0, %ci, %x, %y] : !I_memref 
//    %1 = affine.load %K[%co, %ci, %kh, %kw] : !K_memref 
//    %2 = mulf %0, %1 : f32 
//    %3 = affine.load %O[0, %x, %y, %co] : !O_memref
//    %4 = addf %2, %3 : f32
//    affine.store %4, %O[0, %x, %y, %co] : !O_memref
//  } 

// Code block 2
  affine.parallel (%x, %y, %ci, %co, %kh, %kw) = (0, 0, 0, 0, 0, 0) to (56, 56, 64, 64, 3, 3) reduce ("assign") -> (memref<1x56x56x64xf32>) { 
    %0 = pxa.load %I[0, %ci, %x, %y] : !I_memref 
    %1 = pxa.load %K[%co, %ci, %kh, %kw] : !K_memref 
    %2 = mulf %0, %1 : f32 
    %3 = pxa.reduce addf %2, %O[0, %x, %y, %co] : !O_memref 
    affine.yield %3 : memref<1x56x56x64xf32> 
  } 

  %O_ud = memref_cast %O : !O_memref to memref<*xf32>
  call @print_memref_f32(%O_ud) : (memref<*xf32>) -> ()
 
  dealloc %O : !O_memref
  dealloc %K : !K_memref
  dealloc %I : !I_memref

  return
}


set(_SRCS 
  conv_1d.cc
  max_pool_1d.cc
)

pml_cc_library(
  NAME examples
  TYPE STATIC
  SRCS ${_SRCS}
  DEPS
    plaidml::edsl
    plaidml::testenv
)

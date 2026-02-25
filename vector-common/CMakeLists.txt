SET(VECTOR_COMMON_SOURCES
  vector_conversion.cc
  vector_operations.cc
)
ADD_CONVENIENCE_LIBRARY(vector_common ${VECTOR_COMMON_SOURCES})
TARGET_LINK_LIBRARIES(vector_common mysys)

# Optional AVX2+FMA SIMD for vector distance calculations
OPTION(WITH_VECTOR_AVX2 "Enable AVX2+FMA SIMD for vector operations" ON)

IF(WITH_VECTOR_AVX2)
  MY_ADD_COMPILE_DEFINITIONS(vector_operations.cc
    COMPILE_DEFINITIONS VECTOR_OPS_AVX2)
  IF(MSVC AND NOT WIN32_CLANG)
    ADD_COMPILE_FLAGS(vector_operations.cc COMPILE_FLAGS "/arch:AVX2")
  ELSEIF(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    ADD_COMPILE_FLAGS(vector_operations.cc COMPILE_FLAGS "-mavx2 -mfma")
  ENDIF()
ENDIF()

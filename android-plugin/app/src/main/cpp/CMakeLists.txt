cmake_minimum_required(VERSION 3.22.1)
project(poweramp-start-radio-native C CXX)

# libsoxr - Android build with NEON SIMD (CR32 + CR32S + CR64 + VR32, PFFFT)
set(SOXR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/soxr)

add_library(soxr STATIC
    ${SOXR_DIR}/soxr.c
    ${SOXR_DIR}/data-io.c
    ${SOXR_DIR}/dbesi0.c
    ${SOXR_DIR}/filter.c
    ${SOXR_DIR}/fft4g64.c
    ${SOXR_DIR}/cr.c
    ${SOXR_DIR}/cr32.c
    ${SOXR_DIR}/cr32s.c
    ${SOXR_DIR}/cr64.c
    ${SOXR_DIR}/fft4g32.c
    ${SOXR_DIR}/fft4g.c
    ${SOXR_DIR}/pffft32s.c
    ${SOXR_DIR}/util32s.c
    ${SOXR_DIR}/vr32.c
)

target_include_directories(soxr PRIVATE ${SOXR_DIR})
target_compile_definitions(soxr PRIVATE SOXR_LIB)
target_compile_options(soxr PRIVATE -std=gnu89 -O2)
target_link_libraries(soxr PRIVATE m)

# JNI wrapper
add_library(soxr-jni SHARED soxr_jni.c)
target_include_directories(soxr-jni PRIVATE ${SOXR_DIR})
target_link_libraries(soxr-jni PRIVATE soxr log)

# 16KB page size alignment (required for Google Play targeting Android 15+)
target_link_options(soxr-jni PRIVATE -Wl,-z,max-page-size=16384)

# NEON-accelerated math for k-means clustering, kNN graph, and audio conversion
add_library(math-jni SHARED math_jni.c)
target_compile_options(math-jni PRIVATE -O2)
target_link_libraries(math-jni PRIVATE log)
target_link_options(math-jni PRIVATE -Wl,-z,max-page-size=16384)

# Use the checkpoint's serialized SentencePiece model directly. The former
# Kotlin approximation could not reproduce the model's embedded nmt_nfkc
# normalization table for all Unicode input.
set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
set(SPM_BUILD_TEST OFF CACHE BOOL "" FORCE)
set(SPM_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "" FORCE)
set(SPM_ENABLE_NFKC_COMPILE OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/sentencepiece EXCLUDE_FROM_ALL)

add_library(sentencepiece-jni SHARED sentencepiece_jni.cpp)
target_include_directories(sentencepiece-jni PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/sentencepiece/src
)
target_compile_options(sentencepiece-jni PRIVATE -O3 -fvisibility=hidden)
target_link_libraries(sentencepiece-jni PRIVATE sentencepiece-static log)
target_link_options(sentencepiece-jni PRIVATE
    -Wl,--gc-sections
    -Wl,-z,max-page-size=16384
)

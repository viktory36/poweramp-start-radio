cmake_minimum_required(VERSION 3.22.1)
project(soxr-jni C)

# libsoxr - minimal build for Android (CR32 + CR64 + VR32, no SIMD, no OpenMP)
set(SOXR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/soxr)

add_library(soxr STATIC
    ${SOXR_DIR}/soxr.c
    ${SOXR_DIR}/data-io.c
    ${SOXR_DIR}/dbesi0.c
    ${SOXR_DIR}/filter.c
    ${SOXR_DIR}/fft4g64.c
    ${SOXR_DIR}/cr.c
    ${SOXR_DIR}/cr32.c
    ${SOXR_DIR}/cr64.c
    ${SOXR_DIR}/fft4g32.c
    ${SOXR_DIR}/fft4g.c
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

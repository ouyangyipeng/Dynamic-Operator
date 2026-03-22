cmake_minimum_required(VERSION 3.16)
project(CholeskyOperatorPass)

find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

add_library(CholeskyOperatorPass SHARED
    CholeskyOperatorPass.cpp
)

target_compile_features(CholeskyOperatorPass PRIVATE cxx_std_17)

if(LLVM_ENABLE_RTTI)
    target_compile_options(CholeskyOperatorPass PRIVATE -frtti)
endif()

target_link_libraries(CholeskyOperatorPass PRIVATE
    LLVMCore
    LLVMAnalysis
    LLVMTransformUtils
    LLVMSupport
)

# Install target
install(TARGETS CholeskyOperatorPass LIBRARY DESTINATION lib)
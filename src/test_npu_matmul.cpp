/**
 * NPU GEMM测试程序 - 使用aclopExecuteV2
 * 测试MatMul算子
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>

// CANN头文件
#include "acl/acl.h"
#include "acl/ops/acl_cblas.h"

#define ACL_CHECK(call) \
    do { \
        aclError err = call; \
        if (err != ACL_SUCCESS) { \
            fprintf(stderr, "ACL error at %s:%d: %d\n", __FILE__, __LINE__, err); \
            return err; \
        } \
    } while(0)

int main() {
    printf("=== NPU MatMul Test (aclopExecuteV2) ===\n");
    
    // 初始化ACL
    printf("Initializing ACL...\n");
    ACL_CHECK(aclInit(nullptr));
    
    uint32_t deviceCount = 0;
    ACL_CHECK(aclrtGetDeviceCount(&deviceCount));
    printf("Found %u NPU devices\n", deviceCount);
    
    ACL_CHECK(aclrtSetDevice(0));
    printf("Device 0 set\n");
    
    // 创建流
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));
    printf("Stream created\n");
    
    // 测试矩阵乘法: C = A * B
    int m = 64, k = 64, n = 64;
    size_t sizeA = m * k * sizeof(float);
    size_t sizeB = k * n * sizeof(float);
    size_t sizeC = m * n * sizeof(float);
    
    printf("\nMatrix sizes: A(%dx%d), B(%dx%d), C(%dx%d)\n", m, k, k, n, m, n);
    
    // 分配主机内存
    float *hostA = nullptr, *hostB = nullptr, *hostC = nullptr;
    ACL_CHECK(aclrtMallocHost((void**)&hostA, sizeA));
    ACL_CHECK(aclrtMallocHost((void**)&hostB, sizeB));
    ACL_CHECK(aclrtMallocHost((void**)&hostC, sizeC));
    
    // 初始化矩阵数据
    printf("Initializing matrices...\n");
    for (int i = 0; i < m * k; i++) hostA[i] = (float)(i % 10) / 10.0f;
    for (int i = 0; i < k * n; i++) hostB[i] = (float)(i % 10) / 10.0f;
    for (int i = 0; i < m * n; i++) hostC[i] = 0.0f;
    
    // 分配设备内存
    void *devA = nullptr, *devB = nullptr, *devC = nullptr;
    ACL_CHECK(aclrtMalloc(&devA, sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&devB, sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&devC, sizeC, ACL_MEM_MALLOC_HUGE_FIRST));
    printf("Device memory allocated\n");
    
    // 拷贝数据到设备
    ACL_CHECK(aclrtMemcpy(devA, sizeA, hostA, sizeA, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(devB, sizeB, hostB, sizeB, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(devC, sizeC, hostC, sizeC, ACL_MEMCPY_HOST_TO_DEVICE));
    printf("Data copied to device\n");
    
    // 创建Tensor描述符
    printf("\nCreating tensor descriptors...\n");
    
    // A: [m, k]
    std::vector<int64_t> dimsA = {m, k};
    aclTensorDesc *descA = aclCreateTensorDesc(ACL_FLOAT, 2, dimsA.data(), ACL_FORMAT_ND);
    if (!descA) {
        fprintf(stderr, "Failed to create tensor desc for A\n");
        return 1;
    }
    printf("Tensor A desc created: [%ld, %ld]\n", dimsA[0], dimsA[1]);
    
    // B: [k, n]
    std::vector<int64_t> dimsB = {k, n};
    aclTensorDesc *descB = aclCreateTensorDesc(ACL_FLOAT, 2, dimsB.data(), ACL_FORMAT_ND);
    if (!descB) {
        fprintf(stderr, "Failed to create tensor desc for B\n");
        return 1;
    }
    printf("Tensor B desc created: [%ld, %ld]\n", dimsB[0], dimsB[1]);
    
    // C: [m, n]
    std::vector<int64_t> dimsC = {m, n};
    aclTensorDesc *descC = aclCreateTensorDesc(ACL_FLOAT, 2, dimsC.data(), ACL_FORMAT_ND);
    if (!descC) {
        fprintf(stderr, "Failed to create tensor desc for C\n");
        return 1;
    }
    printf("Tensor C desc created: [%ld, %ld]\n", dimsC[0], dimsC[1]);
    
    // 创建DataBuffer
    printf("\nCreating data buffers...\n");
    aclDataBuffer *bufA = aclCreateDataBuffer(devA, sizeA);
    aclDataBuffer *bufB = aclCreateDataBuffer(devB, sizeB);
    aclDataBuffer *bufC = aclCreateDataBuffer(devC, sizeC);
    
    if (!bufA || !bufB || !bufC) {
        fprintf(stderr, "Failed to create data buffers\n");
        return 1;
    }
    printf("Data buffers created\n");
    
    // 创建属性
    aclopAttr *attr = aclopCreateAttr();
    if (!attr) {
        fprintf(stderr, "Failed to create attr\n");
        return 1;
    }
    
    // 设置transpose属性 (MatMul可能需要)
    // 尝试设置一些常见属性
    aclError ret = aclopSetAttrBool(attr, "transpose_x1", 0);
    if (ret != ACL_SUCCESS) {
        printf("Warning: transpose_x1 not set\n");
    }
    ret = aclopSetAttrBool(attr, "transpose_x2", 0);
    if (ret != ACL_SUCCESS) {
        printf("Warning: transpose_x2 not set\n");
    }
    
    // 准备执行MatMul
    printf("\nExecuting MatMul on NPU...\n");
    
    aclTensorDesc *inputDesc[] = {descA, descB};
    aclTensorDesc *outputDesc[] = {descC};
    aclDataBuffer *inputs[] = {bufA, bufB};
    aclDataBuffer *outputs[] = {bufC};
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // 尝试MatMul算子
    ret = aclopExecuteV2("MatMul", 2, inputDesc, inputs, 1, outputDesc, outputs, attr, stream);
    
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "MatMul failed with error: %d, trying BatchMatMul...\n", ret);
        
        // 尝试BatchMatMul
        ret = aclopExecuteV2("BatchMatMul", 2, inputDesc, inputs, 1, outputDesc, outputs, attr, stream);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "BatchMatMul also failed: %d\n", ret);
        }
    }
    
    // 同步等待完成
    ACL_CHECK(aclrtSynchronizeStream(stream));
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    printf("Operation completed in %.6f seconds\n", elapsed);
    
    // 拷贝结果回主机
    ACL_CHECK(aclrtMemcpy(hostC, sizeC, devC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));
    printf("Result copied to host\n");
    
    // 验证结果
    printf("\nVerifying result...\n");
    float maxError = 0.0f;
    int errorCount = 0;
    for (int i = 0; i < m && errorCount < 10; i++) {
        for (int j = 0; j < n && errorCount < 10; j++) {
            float expected = 0.0f;
            for (int p = 0; p < k; p++) {
                expected += hostA[i * k + p] * hostB[p * n + j];
            }
            float error = fabsf(hostC[i * n + j] - expected);
            if (error > maxError) maxError = error;
            if (error > 1e-3f && errorCount < 5) {
                printf("Error at (%d,%d): expected %.6f, got %.6f\n", i, j, expected, hostC[i * n + j]);
                errorCount++;
            }
        }
    }
    printf("Max error: %.6e\n", maxError);
    printf("Result: %s\n", maxError < 1e-3f ? "PASS" : "FAIL");
    
    // 计算GFLOPS
    double gflops = 2.0 * m * n * k / elapsed / 1e9;
    printf("Performance: %.2f GFLOPS\n", gflops);
    
    // 清理
    aclopDestroyAttr(attr);
    aclDestroyDataBuffer(bufA);
    aclDestroyDataBuffer(bufB);
    aclDestroyDataBuffer(bufC);
    aclDestroyTensorDesc(descA);
    aclDestroyTensorDesc(descB);
    aclDestroyTensorDesc(descC);
    ACL_CHECK(aclrtFree(devA));
    ACL_CHECK(aclrtFree(devB));
    ACL_CHECK(aclrtFree(devC));
    ACL_CHECK(aclrtFreeHost(hostA));
    ACL_CHECK(aclrtFreeHost(hostB));
    ACL_CHECK(aclrtFreeHost(hostC));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclFinalize());
    
    printf("\n=== Test completed ===\n");
    return 0;
}
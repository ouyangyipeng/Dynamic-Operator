/**
 * NPU GEMM测试程序 - 使用FP16
 * 测试aclblasGemmEx功能
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

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

// FP16转换函数 (简化版)
uint16_t float_to_fp16(float f) {
    // 简化的float到fp16转换
    uint32_t x = *((uint32_t*)&f);
    uint16_t h = ((x >> 16) & 0x8000) |  // sign
                 ((((x >> 23) & 0xff) - 127 + 15) << 10) |  // exponent
                 ((x >> 13) & 0x3ff);  // mantissa
    return h;
}

float fp16_to_float(uint16_t h) {
    // 简化的fp16到float转换
    uint32_t x = ((h & 0x8000) << 16) |  // sign
                 (((((h >> 10) & 0x1f) - 15 + 127) & 0xff) << 23) |  // exponent
                 ((h & 0x3ff) << 13);  // mantissa
    return *((float*)&x);
}

int main() {
    printf("=== NPU GEMM FP16 Test ===\n");
    
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
    // 使用较小的矩阵测试
    int m = 64, k = 64, n = 64;
    size_t sizeA = m * k * sizeof(uint16_t);  // FP16
    size_t sizeB = k * n * sizeof(uint16_t);
    size_t sizeC = m * n * sizeof(uint16_t);
    
    printf("\nMatrix sizes: A(%dx%d), B(%dx%d), C(%dx%d) in FP16\n", m, k, k, n, m, n);
    
    // 分配主机内存
    uint16_t *hostA = nullptr, *hostB = nullptr, *hostC = nullptr;
    ACL_CHECK(aclrtMallocHost((void**)&hostA, sizeA));
    ACL_CHECK(aclrtMallocHost((void**)&hostB, sizeB));
    ACL_CHECK(aclrtMallocHost((void**)&hostC, sizeC));
    
    // 初始化矩阵数据 (FP16)
    printf("Initializing matrices...\n");
    for (int i = 0; i < m * k; i++) hostA[i] = float_to_fp16((float)(i % 10) / 10.0f);
    for (int i = 0; i < k * n; i++) hostB[i] = float_to_fp16((float)(i % 10) / 10.0f);
    for (int i = 0; i < m * n; i++) hostC[i] = float_to_fp16(0.0f);
    
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
    
    // 执行GEMM: C = 1.0 * A * B + 0.0 * C
    printf("\nExecuting GEMM on NPU...\n");
    
    // 使用FP16的alpha和beta
    uint16_t alpha_fp16 = float_to_fp16(1.0f);
    uint16_t beta_fp16 = float_to_fp16(0.0f);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    aclError ret = aclblasGemmEx(
        ACL_TRANS_N, ACL_TRANS_N, ACL_TRANS_N,
        m, n, k,
        &alpha_fp16,
        devA, m, ACL_FLOAT16,
        devB, k, ACL_FLOAT16,
        &beta_fp16,
        devC, m, ACL_FLOAT16,
        ACL_COMPUTE_HIGH_PRECISION,
        stream
    );
    
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "aclblasGemmEx failed with error: %d\n", ret);
        // 尝试使用ACL_COMPUTE_LOW_PRECISION
        printf("Trying ACL_COMPUTE_LOW_PRECISION...\n");
        ret = aclblasGemmEx(
            ACL_TRANS_N, ACL_TRANS_N, ACL_TRANS_N,
            m, n, k,
            &alpha_fp16,
            devA, m, ACL_FLOAT16,
            devB, k, ACL_FLOAT16,
            &beta_fp16,
            devC, m, ACL_FLOAT16,
            ACL_COMPUTE_LOW_PRECISION,
            stream
        );
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "Still failed with error: %d\n", ret);
        }
    }
    
    if (ret == ACL_SUCCESS) {
        printf("GEMM launched successfully\n");
    }
    
    // 同步等待完成
    ACL_CHECK(aclrtSynchronizeStream(stream));
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    printf("GEMM completed in %.6f seconds\n", elapsed);
    
    // 拷贝结果回主机
    ACL_CHECK(aclrtMemcpy(hostC, sizeC, devC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));
    printf("Result copied to host\n");
    
    // 验证结果 (计算一个参考值)
    printf("\nVerifying result...\n");
    float maxError = 0.0f;
    int errorCount = 0;
    for (int i = 0; i < m && errorCount < 10; i++) {
        for (int j = 0; j < n && errorCount < 10; j++) {
            float expected = 0.0f;
            for (int p = 0; p < k; p++) {
                expected += fp16_to_float(hostA[i * k + p]) * fp16_to_float(hostB[p * n + j]);
            }
            float got = fp16_to_float(hostC[i * n + j]);
            float error = fabsf(got - expected);
            if (error > maxError) maxError = error;
            if (error > 0.1f && errorCount < 5) {
                printf("Error at (%d,%d): expected %.4f, got %.4f\n", i, j, expected, got);
                errorCount++;
            }
        }
    }
    printf("Max error: %.6e\n", maxError);
    printf("Result: %s\n", maxError < 0.1f ? "PASS" : "FAIL");
    
    // 计算GFLOPS
    double gflops = 2.0 * m * n * k / elapsed / 1e9;
    printf("Performance: %.2f GFLOPS\n", gflops);
    
    // 清理
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
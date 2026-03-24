/**
 * 简单的NPU测试程序
 * 用于诊断CANN环境问题
 */

#include <cstdio>
#include <cstdlib>

// CANN头文件
#include "acl/acl.h"

#define ACL_CHECK(call) \
    do { \
        aclError err = call; \
        if (err != ACL_SUCCESS) { \
            fprintf(stderr, "ACL error at %s:%d: %d\n", __FILE__, __LINE__, err); \
            return err; \
        } \
    } while(0)

int main() {
    printf("=== NPU Simple Test ===\n");
    
    // 步骤1: 初始化ACL
    printf("Step 1: Initializing ACL...\n");
    aclError ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "aclInit failed with error: %d\n", ret);
        return 1;
    }
    printf("aclInit succeeded!\n");
    
    // 步骤2: 获取设备数量
    printf("\nStep 2: Getting device count...\n");
    uint32_t deviceCount = 0;
    ret = aclrtGetDeviceCount(&deviceCount);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "aclrtGetDeviceCount failed with error: %d\n", ret);
        aclFinalize();
        return 1;
    }
    printf("Found %u NPU devices\n", deviceCount);
    
    // 步骤3: 设置设备
    printf("\nStep 3: Setting device 0...\n");
    ret = aclrtSetDevice(0);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "aclrtSetDevice failed with error: %d\n", ret);
        aclFinalize();
        return 1;
    }
    printf("Device 0 set successfully!\n");
    
    // 步骤4: 创建流
    printf("\nStep 4: Creating stream...\n");
    aclrtStream stream = nullptr;
    ret = aclrtCreateStream(&stream);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "aclrtCreateStream failed with error: %d\n", ret);
        aclFinalize();
        return 1;
    }
    printf("Stream created successfully!\n");
    
    // 步骤5: 分配内存
    printf("\nStep 5: Allocating device memory...\n");
    size_t memSize = 1024 * 1024;  // 1MB
    void* devPtr = nullptr;
    ret = aclrtMalloc(&devPtr, memSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "aclrtMalloc failed with error: %d\n", ret);
        aclrtDestroyStream(stream);
        aclFinalize();
        return 1;
    }
    printf("Allocated %zu bytes at %p\n", memSize, devPtr);
    
    // 步骤6: 分配主机内存
    printf("\nStep 6: Allocating host memory...\n");
    void* hostPtr = nullptr;
    ret = aclrtMallocHost(&hostPtr, memSize);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "aclrtMallocHost failed with error: %d\n", ret);
        aclrtFree(devPtr);
        aclrtDestroyStream(stream);
        aclFinalize();
        return 1;
    }
    printf("Host memory allocated at %p\n", hostPtr);
    
    // 步骤7: 内存拷贝测试
    printf("\nStep 7: Testing memory copy...\n");
    // 初始化主机内存
    double* hostData = (double*)hostPtr;
    for (int i = 0; i < 128; i++) {
        hostData[i] = (double)i;
    }
    
    ret = aclrtMemcpy(devPtr, memSize, hostPtr, memSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "aclrtMemcpy H2D failed with error: %d\n", ret);
    } else {
        printf("Host to device copy succeeded!\n");
    }
    
    // 清零主机内存
    for (int i = 0; i < 128; i++) {
        hostData[i] = 0.0;
    }
    
    ret = aclrtMemcpy(hostPtr, memSize, devPtr, memSize, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "aclrtMemcpy D2H failed with error: %d\n", ret);
    } else {
        printf("Device to host copy succeeded!\n");
        // 验证数据
        bool dataCorrect = true;
        for (int i = 0; i < 128; i++) {
            if (hostData[i] != (double)i) {
                dataCorrect = false;
                printf("Data mismatch at %d: expected %f, got %f\n", i, (double)i, hostData[i]);
                break;
            }
        }
        if (dataCorrect) {
            printf("Data verification passed!\n");
        }
    }
    
    // 步骤8: 同步流
    printf("\nStep 8: Synchronizing stream...\n");
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "aclrtSynchronizeStream failed with error: %d\n", ret);
    } else {
        printf("Stream synchronized!\n");
    }
    
    // 清理
    printf("\nCleaning up...\n");
    aclrtFreeHost(hostPtr);
    aclrtFree(devPtr);
    aclrtDestroyStream(stream);
    
    printf("\nStep 9: Finalizing ACL...\n");
    ret = aclFinalize();
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "aclFinalize failed with error: %d\n", ret);
        return 1;
    }
    printf("ACL finalized successfully!\n");
    
    printf("\n=== All tests passed! ===\n");
    return 0;
}
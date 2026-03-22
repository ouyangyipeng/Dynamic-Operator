/**
 * Cholesky算子依赖分析Pass
 * 2026年毕昇杯编译系统挑战赛 - 动态算子图编译与并行调度
 * 
 * 本Pass分析分块Cholesky分解中的算子依赖关系，
 * 并生成并行调度代码
 */

#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include <vector>
#include <map>
#include <set>
#include <string>

using namespace llvm;

#define DEBUG_TYPE "cholesky-operator-pass"

namespace {

// 算子类型
enum class OperatorType {
    CHOLESKY,   // 对角块Cholesky分解
    TRSM,       // 三角求解
    MADDS,      // Schur补更新
    UNKNOWN
};

// 算子信息结构
struct OperatorInfo {
    OperatorType type;
    CallInst* callInst;
    int blockI;  // 块行索引
    int blockJ;  // 块列索引
    int blockK;  // 块索引（用于madd）
    std::set<CallInst*> dependencies;  // 依赖的算子调用
    
    OperatorInfo() : type(OperatorType::UNKNOWN), callInst(nullptr), 
                     blockI(-1), blockJ(-1), blockK(-1) {}
};

struct CholeskyOperatorPass : public FunctionPass {
    static char ID;
    
    CholeskyOperatorPass() : FunctionPass(ID) {}
    
    bool runOnFunction(Function &F) override {
        bool changed = false;
        
        // 只处理block_cholesky函数
        if (F.getName() != "block_cholesky" && 
            F.getName() != "_Z14block_choleskyPdS_ii") {
            return false;
        }
        
        LLVM_DEBUG(dbgs() << "Analyzing function: " << F.getName() << "\n");
        
        // 收集所有算子调用
        std::vector<OperatorInfo> operators;
        std::map<CallInst*, int> callToIndex;
        
        int opIndex = 0;
        for (auto &BB : F) {
            for (auto &I : BB) {
                if (auto *callInst = dyn_cast<CallInst>(&I)) {
                    Function *calledFunc = callInst->getCalledFunction();
                    if (!calledFunc) continue;
                    
                    StringRef funcName = calledFunc->getName();
                    OperatorInfo opInfo;
                    opInfo.callInst = callInst;
                    
                    if (funcName == "cholesky" || funcName == "_Z8choleskyPdS_ii") {
                        opInfo.type = OperatorType::CHOLESKY;
                        LLVM_DEBUG(dbgs() << "Found cholesky call\n");
                    } else if (funcName == "trsm" || funcName == "_Z4trsmPdS_S_iii") {
                        opInfo.type = OperatorType::TRSM;
                        LLVM_DEBUG(dbgs() << "Found trsm call\n");
                    } else if (funcName == "madd" || funcName == "_Z4maddPdS_S_iiii") {
                        opInfo.type = OperatorType::MADDS;
                        LLVM_DEBUG(dbgs() << "Found madd call\n");
                    } else {
                        continue;
                    }
                    
                    operators.push_back(opInfo);
                    callToIndex[callInst] = opIndex++;
                }
            }
        }
        
        LLVM_DEBUG(dbgs() << "Found " << operators.size() << " operator calls\n");
        
        // 分析依赖关系
        analyzeDependencies(operators);
        
        // 生成并行调度代码
        if (!operators.empty()) {
            changed = generateParallelCode(F, operators);
        }
        
        return changed;
    }
    
    void analyzeDependencies(std::vector<OperatorInfo>& operators) {
        // 分块Cholesky算法的依赖关系：
        // 1. cholesky(i) 依赖于之前所有更新到块(i,i)的madd
        // 2. trsm(j,i) 依赖于 cholesky(i) 和之前更新到块(j,i)的madd
        // 3. madd(j,k,i) 依赖于 trsm(j,i) 和 trsm(k,i)
        
        // 按类型分组
        std::vector<OperatorInfo*> choleskyOps;
        std::vector<OperatorInfo*> trsmOps;
        std::vector<OperatorInfo*> maddOps;
        
        for (auto &op : operators) {
            switch (op.type) {
                case OperatorType::CHOLESKY:
                    choleskyOps.push_back(&op);
                    break;
                case OperatorType::TRSM:
                    trsmOps.push_back(&op);
                    break;
                case OperatorType::MADDS:
                    maddOps.push_back(&op);
                    break;
                default:
                    break;
            }
        }
        
        LLVM_DEBUG(dbgs() << "Cholesky ops: " << choleskyOps.size() 
                         << ", TRSM ops: " << trsmOps.size() 
                         << ", MADDS ops: " << maddOps.size() << "\n");
        
        // 建立依赖关系
        // 这里简化处理：假设算子按顺序执行，每个算子依赖于它之前的算子
        // 实际实现需要更精确的依赖分析
        
        for (size_t i = 1; i < operators.size(); i++) {
            // 简化：每个算子依赖于前一个算子
            // 实际应该根据块索引建立精确依赖
            operators[i].dependencies.insert(operators[i-1].callInst);
        }
    }
    
    bool generateParallelCode(Function &F, std::vector<OperatorInfo>& operators) {
        // 这里应该生成并行调度代码
        // 由于这是一个示例Pass，我们只输出分析结果
        
        LLVM_DEBUG(dbgs() << "Generating parallel code for " 
                         << operators.size() << " operators\n");
        
        // 输出依赖关系图
        for (const auto &op : operators) {
            LLVM_DEBUG({
                dbgs() << "Operator ";
                switch (op.type) {
                    case OperatorType::CHOLESKY: dbgs() << "CHOLESKY"; break;
                    case OperatorType::TRSM: dbgs() << "TRSM"; break;
                    case OperatorType::MADDS: dbgs() << "MADDS"; break;
                    default: dbgs() << "UNKNOWN"; break;
                }
                dbgs() << " has " << op.dependencies.size() << " dependencies\n";
            });
        }
        
        return false;  // 暂时不修改代码
    }
    
    void getAnalysisUsage(AnalysisUsage &AU) const override {
        AU.setPreservesAll();
    }
};

} // anonymous namespace

char CholeskyOperatorPass::ID = 0;

// 注册Pass
static RegisterPass<CholeskyOperatorPass> X(
    "cholesky-operator",
    "Cholesky Operator Dependency Analysis Pass",
    false,  // Only looks at CFG
    false   // Analysis pass
);

// 新Pass管理器版本
#ifdef LLVM_ENABLE_NEW_PASS_MANAGER
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

struct CholeskyOperatorPassPlugin : public PassInfoMixin<CholeskyOperatorPassPlugin> {
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
        // 简化实现
        return PreservedAnalyses::all();
    }
};

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
    return {
        LLVM_PLUGIN_API_VERSION, "CholeskyOperatorPass", LLVM_VERSION_STRING,
        [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                    if (Name == "cholesky-operator") {
                        FPM.addPass(CholeskyOperatorPassPlugin());
                        return true;
                    }
                    return false;
                });
        }};
}
#endif
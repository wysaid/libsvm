# Upstream Comparison Tests

这个目录包含用于对比当前 fork 与 upstream libsvm 功能的测试。

## 架构设计

### 1. 隔离构建策略

- **Current 版本**: 使用项目的 CMake 构建系统
- **Upstream 版本**: 使用 git worktree 签出到独立目录，用原始 Makefile 构建
- **完全隔离**: 两个版本的库和头文件存储在不同位置，避免命名冲突

### 2. 测试用例结构

每个测试用例是 `test_cases/` 目录下的独立 `.cpp` 文件，例如：

```
tests/comparison/
├── compare_runner.sh           # 对比测试运行器
├── test_cases/
│   ├── basic_train_predict.cpp # 基本训练和预测测试
│   ├── cross_validation.cpp    # 交叉验证测试
│   └── ...                     # 更多测试用例
└── README.md                   # 本文件
```

### 3. 编译流程

CMake 会自动为每个测试用例生成两个可执行文件：

- `compare_current_<test_name>` - 链接当前版本的 libsvm
- `compare_upstream_<test_name>` - 链接 upstream 版本的 libsvm

### 4. 对比流程

1. `compare_runner.sh` 执行两个版本的可执行文件
2. 捕获各自的输出
3. 使用 `diff` 对比输出
4. 报告结果

## 如何添加新的对比测试

### 步骤 1: 创建测试用例文件

在 `test_cases/` 目录下创建新的 `.cpp` 文件，例如 `model_save_load.cpp`:

```cpp
#include <iostream>
#include <iomanip>
#include "svm.h"

int main() {
    // 禁用 libsvm 输出
    svm_set_print_string_function([](const char*){});
    
    // ... 你的测试逻辑 ...
    
    // 输出测试结果（确保格式一致）
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "version:" << libsvm_version << std::endl;
    std::cout << "result:" << some_value << std::endl;
    
    return 0;
}
```

### 步骤 2: 测试用例编写指南

**重要规则**:

1. **包含头文件**: 只需 `#include "svm.h"`
2. **禁用输出**: 使用 `svm_set_print_string_function` 禁用 libsvm 的训练输出
3. **格式化输出**: 
   - 使用 `key:value` 格式
   - 浮点数使用 `std::setprecision` 固定精度
   - 每行一个值，便于对比
4. **确定性**: 避免随机性（如不设置随机种子）

**输出示例**:

```
version:337
nr_class:2
nr_sv:4
pred_0:1.000000
pred_1:1.000000
accuracy:1.000000
```

### 步骤 3: 重新构建

CMake 会自动检测新的测试用例文件并生成相应的构建目标。只需重新运行：

```bash
./scripts/run_tests.sh --comparison
```

或者：

```bash
cd build
cmake .. -DLIBSVM_BUILD_UPSTREAM_COMPARISON=ON
make -j$(nproc)
```

### 步骤 4: 验证

测试会自动运行并对比输出。如果输出不一致，`compare_runner.sh` 会显示差异。

## 运行对比测试

### 使用测试脚本（推荐）

```bash
# 只运行对比测试
./scripts/run_tests.sh --comparison

# 运行所有测试（包括对比测试）
./scripts/run_tests.sh --all
```

脚本会自动：
1. 检查 upstream 是否已构建
2. 如果没有，自动签出并构建 upstream 版本
3. 配置 CMake 并构建对比测试
4. 运行对比并报告结果

### 手动运行

```bash
# 1. 设置 upstream（如果还没有）
cd build
mkdir -p upstream_source upstream_build/install

# 使用 git worktree 签出 upstream
git worktree add upstream_source upstream

# 构建 upstream
cd upstream_source
make lib -j$(nproc)
ar rcs libsvm.a svm.o

# 安装
mkdir -p ../upstream_build/install/{lib,include}
cp libsvm.a ../upstream_build/install/lib/
cp svm.h ../upstream_build/install/include/
cd ../..

# 2. 构建对比测试
cd build
cmake .. -DLIBSVM_BUILD_UPSTREAM_COMPARISON=ON
make -j$(nproc)

# 3. 运行对比
../tests/comparison/compare_runner.sh bin/comparison
```

## 测试用例示例

### basic_train_predict.cpp

测试基本的训练和预测功能：
- 创建简单的 2D 分类问题
- 训练 SVM 模型
- 预测并输出结果

### cross_validation.cpp

测试交叉验证功能：
- 创建线性可分数据
- 执行 5-fold 交叉验证
- 输出预测结果和准确率

## 故障排查

### upstream 构建失败

确保 upstream 分支存在：
```bash
git branch -a | grep upstream
```

如果不存在，添加 upstream remote 并获取：
```bash
git remote add upstream https://github.com/cjlin1/libsvm.git
git fetch upstream
git branch upstream upstream/master
```

### 输出不匹配

1. **检查随机性**: 确保测试是确定性的
2. **检查精度**: 使用固定的浮点数精度
3. **检查版本差异**: 某些功能可能在不同版本有不同行为
4. **手动测试**: 分别运行两个版本查看详细输出

```bash
./build/bin/comparison/compare_current_<test_name>
./build/bin/comparison/compare_upstream_<test_name>
```

### CMake 找不到 upstream

确保 `build/upstream_build/install` 目录存在且包含：
- `lib/libsvm.a`
- `include/svm.h`

可以重新运行：
```bash
rm -rf build/upstream_*
./scripts/run_tests.sh --comparison
```

## 最佳实践

1. **保持测试简单**: 每个测试用例应该专注于一个功能点
2. **使用描述性输出**: 输出应该清楚地标识测试的内容
3. **文档化差异**: 如果发现预期的差异，在代码中添加注释说明
4. **增量开发**: 先确保简单的测试通过，再添加复杂的测试
5. **代码复用**: 可以创建共享的辅助函数（但要确保两个版本都能编译）

## 当前测试覆盖

- ✅ 基本训练和预测
- ✅ 交叉验证
- ⬜ 模型保存和加载
- ⬜ 概率估计
- ⬜ 决策值
- ⬜ 回归
- ⬜ 多类分类
- ⬜ 不同核函数

欢迎贡献更多测试用例！

# LibSVM 项目现代化改造方案

**创建日期**: 2026-01-06  
**项目版本**: LIBSVM 3.37  
**文档版本**: v1.0

---

## 📋 目录

1. [项目概述](#1-项目概述)
2. [改造目标](#2-改造目标)
3. [当前项目结构分析](#3-当前项目结构分析)
4. [目标项目结构](#4-目标项目结构)
5. [详细改动计划](#5-详细改动计划)
6. [进度跟踪](#6-进度跟踪)
7. [风险与注意事项](#7-风险与注意事项)

---

## 1. 项目概述

LibSVM 是一个广泛使用的支持向量机（SVM）库，支持 C/C++、Python、Java、MATLAB 等多种语言接口。当前项目使用传统的 Makefile 构建系统，本次改造旨在将其迁移到现代化的 CMake 构建系统。

---

## 2. 改造目标

| 序号 | 目标 | 描述 |
|------|------|------|
| 1 | **CMake 化** | 将整个项目改为 CMake 构建系统，移除所有 Makefile |
| 2 | **清理预编译文件** | 删除 `windows/` 目录下的 exe、dll、mex 等预编译文件 |
| 3 | **重组示例目录** | 将 `svm-toy/` 移动到 `examples/` 目录，使用 CMake 管理 |
| 4 | **重组源码目录** | 将核心源代码移至 `src/` 目录 |
| 5 | **语言绑定 CMake 化** | 改造 Python、Java、MATLAB 绑定，统一使用 CMake 构建 |

---

## 3. 当前项目结构分析

### 3.1 当前目录结构

```
libsvm/
├── COPYRIGHT                    # 版权声明
├── FAQ.html                     # FAQ 文档
├── heart_scale                  # 示例数据文件
├── Makefile                     # 主 Makefile (Linux/macOS)
├── Makefile.win                 # Windows Makefile
├── README                       # 说明文档
├── svm-predict.c                # 预测工具源码
├── svm-scale.c                  # 数据缩放工具源码
├── svm-train.c                  # 训练工具源码
├── svm.cpp                      # 核心库实现
├── svm.def                      # Windows DLL 导出定义
├── svm.h                        # 核心库头文件
├── java/                        # Java 绑定
│   ├── Makefile
│   ├── svm_predict.java
│   ├── svm_scale.java
│   ├── svm_toy.java
│   ├── svm_train.java
│   └── libsvm/                  # Java 包
│       ├── svm.java
│       ├── svm.m4               # m4 宏处理文件
│       ├── svm_model.java
│       ├── svm_node.java
│       ├── svm_parameter.java
│       ├── svm_print_interface.java
│       └── svm_problem.java
├── matlab/                      # MATLAB/Octave 绑定
│   ├── Makefile
│   ├── README
│   ├── make.m
│   ├── libsvmread.c
│   ├── libsvmwrite.c
│   ├── svm_model_matlab.c
│   ├── svm_model_matlab.h
│   ├── svmpredict.c
│   └── svmtrain.c
├── python/                      # Python 绑定
│   ├── Makefile
│   ├── MANIFEST.in
│   ├── README
│   ├── setup.py
│   └── libsvm/
│       ├── __init__.py
│       ├── commonutil.py
│       ├── svm.py
│       └── svmutil.py
├── svm-toy/                     # GUI 演示程序
│   ├── qt/                      # Qt 版本
│   │   ├── Makefile
│   │   └── svm-toy.cpp
│   └── windows/                 # Windows 版本
│       └── svm-toy.cpp
├── tools/                       # Python 工具脚本
│   ├── checkdata.py
│   ├── easy.py
│   ├── grid.py
│   ├── README
│   └── subset.py
└── windows/                     # 预编译的 Windows 二进制文件
    ├── libsvm.dll               # ⚠️ 需删除
    ├── libsvmread.mexw64        # ⚠️ 需删除
    ├── libsvmwrite.mexw64       # ⚠️ 需删除
    ├── svm-predict.exe          # ⚠️ 需删除
    ├── svm-scale.exe            # ⚠️ 需删除
    ├── svm-toy.exe              # ⚠️ 需删除
    ├── svm-train.exe            # ⚠️ 需删除
    ├── svmpredict.mexw64        # ⚠️ 需删除
    └── svmtrain.mexw64          # ⚠️ 需删除
```

### 3.2 需要删除的文件

| 类型 | 文件 | 位置 |
|------|------|------|
| Makefile | `Makefile` | 根目录 |
| Makefile | `Makefile.win` | 根目录 |
| Makefile | `Makefile` | java/ |
| Makefile | `Makefile` | matlab/ |
| Makefile | `Makefile` | python/ |
| Makefile | `Makefile` | svm-toy/qt/ |
| 预编译文件 | `libsvm.dll` | windows/ |
| 预编译文件 | `svm-predict.exe` | windows/ |
| 预编译文件 | `svm-scale.exe` | windows/ |
| 预编译文件 | `svm-train.exe` | windows/ |
| 预编译文件 | `svm-toy.exe` | windows/ |
| 预编译文件 | `libsvmread.mexw64` | windows/ |
| 预编译文件 | `libsvmwrite.mexw64` | windows/ |
| 预编译文件 | `svmpredict.mexw64` | windows/ |
| 预编译文件 | `svmtrain.mexw64` | windows/ |

---

## 4. 目标项目结构

```
libsvm/
├── CMakeLists.txt               # 主 CMake 配置文件
├── cmake/                       # CMake 模块和工具
│   ├── FindMatlab.cmake         # MATLAB 查找模块 (可选，CMake 自带)
│   ├── LibSVMConfig.cmake.in    # 安装配置模板
│   └── options.cmake            # 构建选项配置
├── COPYRIGHT
├── FAQ.html
├── README.md                    # 更新为 Markdown 格式
├── docs/                        # 文档目录
│   └── MIGRATION_PLAN.md        # 本文档
├── src/                         # 核心源代码
│   ├── CMakeLists.txt
│   ├── svm.cpp
│   ├── svm.h
│   └── svm.def
├── apps/                        # 命令行工具
│   ├── CMakeLists.txt
│   ├── svm-predict.c
│   ├── svm-scale.c
│   └── svm-train.c
├── examples/                    # 示例程序
│   ├── CMakeLists.txt
│   ├── data/
│   │   └── heart_scale          # 示例数据
│   └── svm-toy/                 # Qt GUI 示例
│       ├── CMakeLists.txt
│       └── svm-toy.cpp
├── bindings/                    # 语言绑定
│   ├── CMakeLists.txt
│   ├── python/
│   │   ├── CMakeLists.txt       # Python C 扩展构建
│   │   ├── setup.py             # pip 安装支持
│   │   ├── MANIFEST.in
│   │   ├── README.md
│   │   └── libsvm/
│   │       ├── __init__.py
│   │       ├── commonutil.py
│   │       ├── svm.py
│   │       └── svmutil.py
│   ├── java/
│   │   ├── CMakeLists.txt       # JNI 构建
│   │   ├── jni/                 # JNI 包装器 (新增)
│   │   │   └── svm_jni.c
│   │   ├── svm_predict.java
│   │   ├── svm_scale.java
│   │   ├── svm_toy.java
│   │   ├── svm_train.java
│   │   └── libsvm/
│   │       ├── svm.java
│   │       ├── svm_model.java
│   │       ├── svm_node.java
│   │       ├── svm_parameter.java
│   │       ├── svm_print_interface.java
│   │       └── svm_problem.java
│   └── matlab/
│       ├── CMakeLists.txt       # MEX 构建
│       ├── README.md
│       ├── make.m               # 备用 MATLAB 构建脚本
│       ├── libsvmread.c
│       ├── libsvmwrite.c
│       ├── svm_model_matlab.c
│       ├── svm_model_matlab.h
│       ├── svmpredict.c
│       └── svmtrain.c
└── tools/                       # Python 工具脚本 (保持不变)
    ├── README
    ├── checkdata.py
    ├── easy.py
    ├── grid.py
    └── subset.py
```

---

## 5. 详细改动计划

### 5.1 阶段一：清理预编译文件

**任务 ID**: TASK-001  
**优先级**: 高  
**状态**: 🔴 待开始

| 子任务 | 描述 | 状态 |
|--------|------|------|
| 1.1 | 删除 `windows/` 目录下所有 `.exe` 文件 | ⬜ |
| 1.2 | 删除 `windows/` 目录下 `libsvm.dll` | ⬜ |
| 1.3 | 删除 `windows/` 目录下所有 `.mexw64` 文件 | ⬜ |
| 1.4 | 删除整个 `windows/` 目录 | ⬜ |

---

### 5.2 阶段二：重组目录结构

**任务 ID**: TASK-002  
**优先级**: 高  
**状态**: 🔴 待开始

| 子任务 | 描述 | 状态 |
|--------|------|------|
| 2.1 | 创建 `src/` 目录 | ⬜ |
| 2.2 | 移动 `svm.cpp`、`svm.h`、`svm.def` 到 `src/` | ⬜ |
| 2.3 | 创建 `apps/` 目录 | ⬜ |
| 2.4 | 移动 `svm-predict.c`、`svm-scale.c`、`svm-train.c` 到 `apps/` | ⬜ |
| 2.5 | 创建 `examples/` 目录 | ⬜ |
| 2.6 | 创建 `examples/data/` 目录并移动 `heart_scale` | ⬜ |
| 2.7 | 移动 `svm-toy/` 到 `examples/svm-toy/` | ⬜ |
| 2.8 | 只保留 Qt 版本的 `svm-toy.cpp`，删除 Windows 版本 | ⬜ |
| 2.9 | 创建 `bindings/` 目录 | ⬜ |
| 2.10 | 移动 `python/`、`java/`、`matlab/` 到 `bindings/` | ⬜ |
| 2.11 | 创建 `cmake/` 目录 | ⬜ |
| 2.12 | 创建 `docs/` 目录 | ⬜ |

---

### 5.3 阶段三：删除 Makefile 文件

**任务 ID**: TASK-003  
**优先级**: 中  
**状态**: 🔴 待开始

| 子任务 | 描述 | 状态 |
|--------|------|------|
| 3.1 | 删除根目录 `Makefile` | ⬜ |
| 3.2 | 删除根目录 `Makefile.win` | ⬜ |
| 3.3 | 删除 `java/Makefile` | ⬜ |
| 3.4 | 删除 `matlab/Makefile` | ⬜ |
| 3.5 | 删除 `python/Makefile` | ⬜ |
| 3.6 | 删除 `svm-toy/qt/Makefile` | ⬜ |

---

### 5.4 阶段四：创建 CMake 构建系统

**任务 ID**: TASK-004  
**优先级**: 高  
**状态**: 🔴 待开始

#### 5.4.1 主 CMakeLists.txt

| 子任务 | 描述 | 状态 |
|--------|------|------|
| 4.1.1 | 创建根目录 `CMakeLists.txt` | ⬜ |
| 4.1.2 | 配置项目名称、版本、语言 | ⬜ |
| 4.1.3 | 添加构建选项 (OpenMP, 共享库等) | ⬜ |
| 4.1.4 | 添加子目录配置 | ⬜ |
| 4.1.5 | 配置安装规则 | ⬜ |

#### 5.4.2 核心库 CMakeLists.txt

| 子任务 | 描述 | 状态 |
|--------|------|------|
| 4.2.1 | 创建 `src/CMakeLists.txt` | ⬜ |
| 4.2.2 | 配置静态库 `libsvm_static` | ⬜ |
| 4.2.3 | 配置共享库 `libsvm` (可选) | ⬜ |
| 4.2.4 | 配置头文件安装 | ⬜ |
| 4.2.5 | 配置 OpenMP 支持 (可选) | ⬜ |

#### 5.4.3 命令行工具 CMakeLists.txt

| 子任务 | 描述 | 状态 |
|--------|------|------|
| 4.3.1 | 创建 `apps/CMakeLists.txt` | ⬜ |
| 4.3.2 | 配置 `svm-train` 可执行文件 | ⬜ |
| 4.3.3 | 配置 `svm-predict` 可执行文件 | ⬜ |
| 4.3.4 | 配置 `svm-scale` 可执行文件 | ⬜ |

#### 5.4.4 示例程序 CMakeLists.txt

| 子任务 | 描述 | 状态 |
|--------|------|------|
| 4.4.1 | 创建 `examples/CMakeLists.txt` | ⬜ |
| 4.4.2 | 创建 `examples/svm-toy/CMakeLists.txt` | ⬜ |
| 4.4.3 | 配置 Qt5/Qt6 查找和链接 | ⬜ |
| 4.4.4 | 配置 MOC 自动处理 | ⬜ |

---

### 5.5 阶段五：语言绑定 CMake 化

**任务 ID**: TASK-005  
**优先级**: 中  
**状态**: 🔴 待开始

#### 5.5.1 Python 绑定

| 子任务 | 描述 | 状态 |
|--------|------|------|
| 5.1.1 | 创建 `bindings/python/CMakeLists.txt` | ⬜ |
| 5.1.2 | 配置 Python C 扩展模块构建 | ⬜ |
| 5.1.3 | 更新 `setup.py` 使用 CMake 或保持 ctypes 方式 | ⬜ |
| 5.1.4 | 更新 `svm.py` 中的库加载路径 | ⬜ |

**说明**: Python 版本使用 ctypes 加载共享库，需要确保 CMake 生成的 `libsvm.so` 或 `libsvm.dll` 可以被正确找到。

#### 5.5.2 Java 绑定

| 子任务 | 描述 | 状态 |
|--------|------|------|
| 5.2.1 | 创建 `bindings/java/CMakeLists.txt` | ⬜ |
| 5.2.2 | 配置 m4 预处理 (生成 svm.java) | ⬜ |
| 5.2.3 | 配置 Java 编译 | ⬜ |
| 5.2.4 | 配置 JAR 打包 | ⬜ |
| 5.2.5 | (可选) 创建 JNI 绑定以使用 C 库 | ⬜ |

**说明**: 当前 Java 版本是纯 Java 实现，不依赖 C 库。可选择保持现状或添加 JNI 绑定。

#### 5.5.3 MATLAB 绑定

| 子任务 | 描述 | 状态 |
|--------|------|------|
| 5.3.1 | 创建 `bindings/matlab/CMakeLists.txt` | ⬜ |
| 5.3.2 | 配置 MATLAB 查找 (FindMatlab) | ⬜ |
| 5.3.3 | 配置 MEX 文件构建 | ⬜ |
| 5.3.4 | 配置 `svmtrain.mex*` 构建 | ⬜ |
| 5.3.5 | 配置 `svmpredict.mex*` 构建 | ⬜ |
| 5.3.6 | 配置 `libsvmread.mex*` 构建 | ⬜ |
| 5.3.7 | 配置 `libsvmwrite.mex*` 构建 | ⬜ |

---

### 5.6 阶段六：CMake 辅助模块

**任务 ID**: TASK-006  
**优先级**: 低  
**状态**: 🔴 待开始

| 子任务 | 描述 | 状态 |
|--------|------|------|
| 6.1 | 创建 `cmake/options.cmake` 构建选项配置 | ⬜ |
| 6.2 | 创建 `cmake/LibSVMConfig.cmake.in` 安装配置模板 | ⬜ |
| 6.3 | 配置 `find_package(LibSVM)` 支持 | ⬜ |

---

### 5.7 阶段七：文档和清理

**任务 ID**: TASK-007  
**优先级**: 低  
**状态**: 🔴 待开始

| 子任务 | 描述 | 状态 |
|--------|------|------|
| 7.1 | 更新 README 为 Markdown 格式 | ⬜ |
| 7.2 | 添加 CMake 构建说明 | ⬜ |
| 7.3 | 更新各子目录 README | ⬜ |
| 7.4 | 删除 `java/libsvm/svm.m4` 后的临时文件 | ⬜ |
| 7.5 | 更新 `.gitignore` | ⬜ |

---

## 6. 进度跟踪

### 6.1 总体进度

| 阶段 | 描述 | 进度 | 状态 |
|------|------|------|------|
| 阶段一 | 清理预编译文件 | 100% | 🟢 已完成 |
| 阶段二 | 重组目录结构 | 100% | 🟢 已完成 |
| 阶段三 | 删除 Makefile 文件 | 100% | 🟢 已完成 |
| 阶段四 | 创建 CMake 构建系统 | 100% | 🟢 已完成 |
| 阶段五 | 语言绑定 CMake 化 | 100% | 🟢 已完成 |
| 阶段六 | CMake 辅助模块 | 100% | 🟢 已完成 |
| 阶段七 | 文档和清理 | 100% | 🟢 已完成 |

### 6.2 状态说明

- 🔴 待开始
- 🟡 进行中
- 🟢 已完成
- ⬜ 子任务待开始
- ✅ 子任务已完成

---

## 7. 风险与注意事项

### 7.1 兼容性风险

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| Qt 版本差异 | svm-toy 可能需要适配不同 Qt 版本 | 支持 Qt5 和 Qt6 |
| MATLAB 版本差异 | MEX 编译可能因版本不同而失败 | 保留 `make.m` 作为备选 |
| 旧系统 CMake 版本 | 某些功能可能不可用 | 设置最低 CMake 版本要求 (3.16+) |

### 7.2 注意事项

1. **Java 版本特殊性**: 当前 Java 实现是纯 Java，使用 m4 宏生成代码。需要决定是否保持纯 Java 或添加 JNI。

2. **Python ctypes 依赖**: Python 版本通过 ctypes 加载共享库，需要确保库路径配置正确。

3. **OpenMP 支持**: 原 Makefile 中 OpenMP 是注释掉的可选项，CMake 中需要作为选项保留。

4. **Windows svm-toy**: Windows 版本使用 Win32 API，与 Qt 版本不同。建议只保留 Qt 版本以简化维护。

5. **向后兼容**: 虽然移除了 Makefile，用户仍可能需要简单的构建方式。考虑提供简化的构建脚本。

### 7.3 测试计划

改造完成后需要验证：

- [ ] 核心库 `libsvm` 编译成功
- [ ] 命令行工具 `svm-train`、`svm-predict`、`svm-scale` 正常工作
- [ ] 使用 `heart_scale` 数据进行功能验证
- [ ] Python 绑定可正常导入和使用
- [ ] Java 版本可正常编译和运行
- [ ] MATLAB 绑定可正常编译 (如有 MATLAB 环境)
- [ ] svm-toy Qt 版本可正常编译运行 (如有 Qt 环境)
- [ ] 跨平台测试 (Linux, macOS, Windows)

---

## 附录 A: CMake 构建命令示例

```bash
# 基本构建
mkdir build && cd build
cmake ..
cmake --build .

# 带选项构建
cmake -DBUILD_SHARED_LIBS=ON -DENABLE_OPENMP=ON ..

# 安装
cmake --install . --prefix /usr/local

# 构建特定目标
cmake --build . --target svm-train
cmake --build . --target python-bindings
cmake --build . --target matlab-bindings
```

---

## 附录 B: 文件移动对照表

| 原路径 | 新路径 |
|--------|--------|
| `svm.cpp` | `src/svm.cpp` |
| `svm.h` | `src/svm.h` |
| `svm.def` | `src/svm.def` |
| `svm-train.c` | `apps/svm-train.c` |
| `svm-predict.c` | `apps/svm-predict.c` |
| `svm-scale.c` | `apps/svm-scale.c` |
| `heart_scale` | `examples/data/heart_scale` |
| `svm-toy/qt/svm-toy.cpp` | `examples/svm-toy/svm-toy.cpp` |
| `python/` | `bindings/python/` |
| `java/` | `bindings/java/` |
| `matlab/` | `bindings/matlab/` |

---

## 附录 C: 快速执行命令参考

以下命令可在项目根目录执行，用于快速完成迁移工作：

### C.1 清理预编译文件

```bash
# 删除 Windows 预编译文件
rm -rf windows/
```

### C.2 目录重组

```bash
# 创建新目录结构
mkdir -p src apps examples/data examples/svm-toy bindings cmake docs

# 移动核心源码
mv svm.cpp svm.h svm.def src/

# 移动命令行工具
mv svm-train.c svm-predict.c svm-scale.c apps/

# 移动示例数据
mv heart_scale examples/data/

# 移动 svm-toy (只保留 Qt 版本)
mv svm-toy/qt/svm-toy.cpp examples/svm-toy/
rm -rf svm-toy/

# 移动语言绑定
mv python bindings/
mv java bindings/
mv matlab bindings/
```

### C.3 删除 Makefile

```bash
# 删除所有 Makefile
rm -f Makefile Makefile.win
rm -f bindings/java/Makefile
rm -f bindings/matlab/Makefile
rm -f bindings/python/Makefile
```

---

## 附录 D: 预期最终目录树

```
libsvm/
├── CMakeLists.txt
├── COPYRIGHT
├── FAQ.html
├── README.md
├── apps/
│   ├── CMakeLists.txt
│   ├── svm-predict.c
│   ├── svm-scale.c
│   └── svm-train.c
├── bindings/
│   ├── CMakeLists.txt
│   ├── java/
│   │   ├── CMakeLists.txt
│   │   ├── libsvm/
│   │   │   ├── svm.java
│   │   │   ├── svm.m4
│   │   │   ├── svm_model.java
│   │   │   ├── svm_node.java
│   │   │   ├── svm_parameter.java
│   │   │   ├── svm_print_interface.java
│   │   │   └── svm_problem.java
│   │   ├── svm_predict.java
│   │   ├── svm_scale.java
│   │   ├── svm_toy.java
│   │   └── svm_train.java
│   ├── matlab/
│   │   ├── CMakeLists.txt
│   │   ├── README
│   │   ├── libsvmread.c
│   │   ├── libsvmwrite.c
│   │   ├── make.m
│   │   ├── svm_model_matlab.c
│   │   ├── svm_model_matlab.h
│   │   ├── svmpredict.c
│   │   └── svmtrain.c
│   └── python/
│       ├── CMakeLists.txt
│       ├── MANIFEST.in
│       ├── README
│       ├── libsvm/
│       │   ├── __init__.py
│       │   ├── commonutil.py
│       │   ├── svm.py
│       │   └── svmutil.py
│       └── setup.py
├── cmake/
│   ├── LibSVMConfig.cmake.in
│   └── options.cmake
├── docs/
│   └── MIGRATION_PLAN.md
├── examples/
│   ├── CMakeLists.txt
│   ├── data/
│   │   └── heart_scale
│   └── svm-toy/
│       ├── CMakeLists.txt
│       └── svm-toy.cpp
├── src/
│   ├── CMakeLists.txt
│   ├── svm.cpp
│   ├── svm.def
│   └── svm.h
└── tools/
    ├── README
    ├── checkdata.py
    ├── easy.py
    ├── grid.py
    └── subset.py
```

---

*文档结束*

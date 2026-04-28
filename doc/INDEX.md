# R2-Dreamer 文档导航

欢迎使用 R2-Dreamer 项目！本索引帮助您快速找到所需文档。

---

## 📖 文档地图

### 🚀 新手入门路径

```
开始
  ↓
[README_zh.md](../README_zh.md)          ← 从这里开始！了解项目全貌
  ↓
[docs/quick_reference.md](quick_reference.md)  ← 5分钟快速运行
  ↓
开始实验 🎉
```

### 🔬 深入学习路径

```
README_zh.md                              ← 基础概念
  ↓
[docs/project_architecture.md](project_architecture.md)  ← 深入理解架构
  ↓
源代码（带注释）                           ← 查看实现细节
  ↓
修改和扩展代码 🔧
```

### 🐛 问题解决路径

```
遇到问题
  ↓
[docs/quick_reference.md](quick_reference.md) → "常见问题快速修复"  ← 快速解决方案
  ↓
未解决？
  ↓
[docs/project_architecture.md](project_architecture.md) → "调试技巧"  ← 深入调试
  ↓
仍未解决？
  ↓
提交 GitHub Issue 💬
```

---

## 📚 文档清单

### 主要文档

| 文档 | 路径 | 用途 | 阅读时间 |
|------|------|------|---------|
| **中文README** | [`README_zh.md`](../README_zh.md) | 项目总览、安装、配置 | 15分钟 |
| **架构详解** | [`docs/project_architecture.md`](project_architecture.md) | 技术细节、算法原理 | 30分钟 |
| **快速参考** | [`docs/quick_reference.md`](quick_reference.md) | 命令速查、故障排除 | 10分钟 |
| **MetaDrive接入** | [`docs/metadrive.md`](metadrive.md) | SafeMetaDrive接口、配置、日志字段 | 10分钟 |

### 技术文档

| 文档 | 路径 | 内容 |
|------|------|------|
| 张量形状 | `docs/tensor_shapes.md` | 所有张量的维度说明 |
| Docker使用 | `docs/docker.md` | 容器化部署指南 |
| 工作总结 | [`docs/COMMENT_WORK_SUMMARY.md`](COMMENT_WORK_SUMMARY.md) | 代码注释和文档工作说明 |

### 源代码（带详细注释）

| 文件 | 关键内容 |
|------|---------|
| [`dreamer.py`](../dreamer.py) | 核心智能体，训练逻辑 |
| [`rssm.py`](../rssm.py) | 世界模型，状态空间 |
| [`networks.py`](../networks.py) | 神经网络组件 |

---

## 🎯 按任务查找

### 我想...

#### 安装和运行
→ [README_zh.md - 安装与配置](../README_zh.md#-安装与配置)

#### 快速开始实验
→ [docs/quick_reference.md - 5分钟快速开始](quick_reference.md#-5分钟快速开始)

#### 理解算法原理
→ [README_zh.md - 核心算法](../README_zh.md#-核心算法)  
→ [docs/project_architecture.md - 关键模块说明](project_architecture.md#-关键模块说明)

#### 调整超参数
→ [docs/quick_reference.md - 调参技巧](quick_reference.md#-调参技巧)

#### 解决训练问题
→ [docs/quick_reference.md - 常见问题快速修复](quick_reference.md#-常见问题快速修复)

#### 添加新环境
→ [README_zh.md - 添加新环境](../README_zh.md#添加新环境)

#### 实现新方法
→ [README_zh.md - 添加新表征学习方法](../README_zh.md#添加新表征学习方法)

#### 理解数据流
→ [docs/project_architecture.md - 数据流详解](project_architecture.md#-数据流详解)

#### 优化性能
→ [docs/project_architecture.md - 性能分析](project_architecture.md#-性能分析)

#### 调试代码
→ [docs/project_architecture.md - 调试技巧](project_architecture.md#-调试技巧)

#### 查看指标含义
→ [docs/quick_reference.md - 关键指标解读](quick_reference.md#-关键指标解读)

#### 加载checkpoint
→ [docs/quick_reference.md - 文件输出说明](quick_reference.md#-文件输出说明)

---

## 📊 按角色查找

### 🎓 研究者
**关注**: 算法原理、实验设计、性能基准

推荐阅读顺序:
1. [README_zh.md - 核心算法](../README_zh.md#-核心算法)
2. [docs/project_architecture.md](project_architecture.md)
3. [docs/quick_reference.md - 环境特定建议](quick_reference.md#-环境特定建议)

### 💻 工程师
**关注**: 代码结构、部署、性能优化

推荐阅读顺序:
1. [README_zh.md - 项目结构](../README_zh.md#-项目结构)
2. [docs/project_architecture.md - 关键模块说明](project_architecture.md#-关键模块说明)
3. [docs/project_architecture.md - 性能分析](project_architecture.md#-性能分析)

### 🔧 开发者
**关注**: 代码实现、扩展开发、调试

推荐阅读顺序:
1. [docs/project_architecture.md](project_architecture.md)
2. 源代码注释（dreamer.py, rssm.py, networks.py）
3. [docs/quick_reference.md - 调试检查清单](quick_reference.md#-调试检查清单)

### 🚀 初学者
**关注**: 快速上手、基本概念

推荐阅读顺序:
1. [README_zh.md](../README_zh.md) - 完整阅读
2. [docs/quick_reference.md](quick_reference.md) - 收藏备用
3. 开始实验，遇到问题再查阅

---

## 🔍 快速搜索关键词

### 算法相关
- **RSSM**: [project_architecture.md](project_architecture.md#2-rssm-世界模型-rssmpy)
- **Actor-Critic**: [README_zh.md](../README_zh.md#3-actor-critic-训练)
- **表征学习**: [README_zh.md](../README_zh.md#2-表征学习方法)
- **想象rollout**: [project_architecture.md](project_architecture.md#想象-rollout-详细流程)

### 配置相关
- **超参数**: [project_architecture.md](project_architecture.md#️-关键超参数)
- **模型大小**: [quick_reference.md](quick_reference.md#调整模型大小)
- **环境配置**: [README_zh.md](../README_zh.md#配置说明)

### 问题相关
- **OOM**: [quick_reference.md](quick_reference.md#-cuda-out-of-memory)
- **训练不稳定**: [quick_reference.md](quick_reference.md#-训练不稳定发散)
- **分数低**: [quick_reference.md](quick_reference.md#-评估分数低)
- **速度慢**: [quick_reference.md](quick_reference.md#-速度慢)

### 代码相关
- **张量形状**: `docs/tensor_shapes.md`
- **梯度流**: [project_architecture.md](project_architecture.md#梯度流分析)
- **数据流**: [project_architecture.md](project_architecture.md#训练时的数据流)

---

## 💡 使用建议

### 第一次使用
1. ⭐ **Star** 本项目
2. 📖 完整阅读 [README_zh.md](../README_zh.md)
3. 💻 运行一个简单示例
4. 🔖 收藏 [quick_reference.md](quick_reference.md)

### 日常开发
1. 📌 将本文档加入书签
2. 🔍 使用 Ctrl+F 搜索关键词
3. 📝 遇到问题先查文档
4. 💬 无法解决再提Issue

### 深入研究
1. 📚 阅读 [project_architecture.md](project_architecture.md)
2. 🔬 查看源代码注释
3. 🧪 设计消融实验
4. 📊 记录实验结果

---

## 🆘 获取帮助

### 文档未覆盖的问题？

1. **搜索现有Issue**: GitHub Issues页面
2. **提交新Issue**: 提供详细信息和复现代码
3. **邮件联系**: your-email@example.com
4. **查阅论文**: Dreamer V3原始论文

### 文档有误或需要补充？

欢迎提交 Pull Request 改进文档！

---

## 📈 文档更新日志

### 2026-04-11
- ✅ 创建完整的中文README
- ✅ 为核心代码添加详细注释
- ✅ 编写架构详解文档
- ✅ 编写快速参考手册
- ✅ 创建文档导航索引

---

**祝您使用愉快！🎉**

如有任何建议，欢迎反馈！

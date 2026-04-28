# 模型测试快速参考

## 🎯 根据您的需求选择

### 场景1: "我想快速看看模型效果"
⏱️ 耗时: 2-5分钟

```bash
python test_checkpoint.py logdir/2026-04-23/10-45-06/checkpoints/final.pt 5
```

✅ 输出: 平均奖励、标准差、episode长度

---

### 场景2: "我想看模型的实际运行过程（视频）"
⏱️ 耗时: 5-15分钟

```bash
# 需要先安装 opencv
pip install opencv-python

# 然后生成视频
python evaluate.py \
    checkpoint=logdir/2026-04-23/10-45-06/checkpoints/final.pt \
    num_episodes=3 \
    save_video=true \
    video_dir=my_videos/

# 播放视频
ffplay my_videos/episode_000.mp4
```

✅ 输出: MP4视频文件 + 统计数据

---

### 场景3: "我想实时看模型做决策"
⏱️ 耗时: 1-3分钟

```bash
# 需要图形界面和 opencv
pip install opencv-python

python evaluate.py \
    checkpoint=logdir/2026-04-23/10-45-06/checkpoints/final.pt \
    render=true \
    num_episodes=2
```

✅ 输出: 实时弹出窗口显示智能体视图

---

### 场景4: "我想在TensorBoard中查看完整的训练过程"
⏱️ 耗时: 即时

```bash
tensorboard --logdir logdir/2026-04-23/10-45-06/

# 打开浏览器: http://localhost:6006
```

✅ 输出: 网页仪表板，包含：
- 训练曲线
- 评估指标
- 视频回放
- 损失函数

---

### 场景5: "我想比较多个模型的性能"
⏱️ 耗时: 10-20分钟

```bash
# 创建对比脚本
cat > compare_models.sh << 'EOF'
#!/bin/bash
echo "模型对比评估"
echo "============================================"

for logdir in logdir/*/; do
    if [ -f "$logdir/checkpoints/final.pt" ]; then
        echo ""
        echo "📂 $(basename $logdir)"
        python test_checkpoint.py "$logdir/checkpoints/final.pt" 5 | tail -5
    fi
done
EOF

chmod +x compare_models.sh
./compare_models.sh
```

✅ 输出: 所有模型的并排对比

---

### 场景6: "我想测试特定任务上的性能"
⏱️ 耗时: 5-10分钟

```bash
# 测试不同的DMC任务
for task in walker_walk cheetah_run humanoid_stand cartpole_balance; do
    echo "Task: $task"
    python evaluate.py \
        checkpoint=logdir/2026-04-23/10-45-06/checkpoints/final.pt \
        env=dmc_vision \
        env.task=$task \
        num_episodes=3
    echo "---"
done
```

✅ 输出: 各任务的性能对比

---

### 场景7: "我想导出结果用于报告或论文"
⏱️ 耗时: 5-10分钟

```bash
# 导出为文本格式
python test_checkpoint.py logdir/2026-04-23/10-45-06/checkpoints/final.pt 20 | tee results.txt

# 或导出为JSON（需要修改脚本）
python evaluate.py \
    checkpoint=logdir/2026-04-23/10-45-06/checkpoints/final.pt \
    num_episodes=20 > eval_results.json
```

✅ 输出: 可以导入Excel或其他工具的结果文件

---

## 📊 关键指标解释

| 指标 | 好的范围 | 说明 |
|------|--------|------|
| **reward_mean** | 任务相关 | 越高越好，主要性能指标 |
| **reward_std** | 越小越好 | 低std表示性能稳定 |
| **length_mean** | 接近限制值 | 通常1000（环境限制），说明能坚持长时间 |
| **eval_score** | 与train/score接近 | 接近说明没有过拟合 |

---

## 🔍 检查点管理

```bash
# 查看所有可用的检查点
ls -lh logdir/2026-04-23/10-45-06/checkpoints/

# 查看检查点信息
cat logdir/2026-04-23/10-45-06/checkpoints/index.jsonl

# 找到特定step的检查点
find logdir -name "step_000500000.pt"

# 查找最新的检查点
find logdir -name "latest.pt" -type l

# 清理旧检查点（保留最后5个）
cd logdir/2026-04-23/10-45-06/checkpoints/
ls -t step_*.pt | tail -n +6 | xargs rm
```

---

## 🚀 性能优化

### 如果评估太慢：

```bash
# 使用CPU（某些情况下更快）
python test_checkpoint.py checkpoint.pt device=cpu

# 减少环境数量
python evaluate.py checkpoint=checkpoint.pt env.num_envs=1

# 简化模型
python evaluate.py checkpoint=checkpoint.pt model=size12M
```

### 如果显存不足：

```bash
# 使用半精度浮点数
python evaluate.py checkpoint=checkpoint.pt torch.dtype=float16

# 或者使用CPU
python test_checkpoint.py checkpoint.pt device=cpu
```

---

## 🎓 常见操作速查

```bash
# 最快速检查 (30秒)
python test_checkpoint.py logdir/*/checkpoints/final.pt 2

# 标准评估 (3分钟)
python test_checkpoint.py logdir/*/checkpoints/final.pt 10

# 深度评估 (10分钟)
python evaluate.py checkpoint=logdir/*/checkpoints/final.pt num_episodes=50

# 生成视频演示 (5分钟)
python evaluate.py checkpoint=logdir/*/checkpoints/final.pt \
    save_video=true num_episodes=5 video_dir=demo/

# 查看训练进度 (即时)
tensorboard --logdir logdir/

# 批量测试所有模型
for d in logdir/*/; do
    [ -f "$d/checkpoints/final.pt" ] && \
    python test_checkpoint.py "$d/checkpoints/final.pt" 3
done
```

---

## 💡 使用建议

1. **第一次测试**：用 `test_checkpoint.py` 快速验证模型是否可用
2. **性能评估**：运行 10-20 episodes 以获得稳定的统计数据
3. **发布结果**：运行 50+ episodes 以获得发表级别的结果
4. **调试问题**：使用 TensorBoard 查看训练曲线，识别问题所在
5. **可视化**：用视频或实时渲染来理解模型的行为

---

## 📚 更多信息

- 详细指南: [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md)
- 快速开始: [GETTING_STARTED.md](GETTING_STARTED.md)
- 项目结构: [project_architecture.md](project_architecture.md)
- MetaDrive特定: [metadrive.md](metadrive.md)

---

生成时间: 2026-04-24
最后更新: R2-Dreamer 项目

#!/bin/bash
# 模型测试脚本 - 一键评估最新的检查点
# 使用: bash test_latest.sh [num_episodes]

set -e

# 配置
NUM_EPISODES=${1:-5}
LOGDIR="logdir"
PYTHON_CMD=("${PYTHON:-python}")

if ! "${PYTHON_CMD[@]}" -c "import omegaconf" >/dev/null 2>&1; then
    if command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx "r2dreamer"; then
        PYTHON_CMD=(conda run -n r2dreamer python)
    else
        echo "❌ 错误: 当前 Python 环境缺少 omegaconf/torch 等评估依赖"
        echo "   请先运行: conda activate r2dreamer"
        exit 1
    fi
fi

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════╗"
echo "║     R2-Dreamer 模型评估工具                        ║"
echo "╚════════════════════════════════════════════════════╝"
echo -e "${NC}"

# 查找最新的检查点
LATEST_CHECKPOINT=$(find $LOGDIR -name "final.pt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo -e "${RED}❌ 错误: 未找到任何检查点文件${NC}"
    echo "   请确保在 $LOGDIR 目录中有 final.pt 文件"
    echo ""
    echo "   可用的检查点："
    find $LOGDIR -name "*.pt" -type f 2>/dev/null || echo "   (无)"
    exit 1
fi

echo -e "${YELLOW}📂 找到检查点:${NC}"
echo "   $LATEST_CHECKPOINT"
echo ""

# 显示检查点信息
if [ -f "${LATEST_CHECKPOINT%/*}/index.jsonl" ]; then
    echo -e "${YELLOW}📋 检查点信息:${NC}"
    tail -1 "${LATEST_CHECKPOINT%/*}/index.jsonl" | python3 -m json.tool 2>/dev/null || echo "   (无法解析信息)"
    echo ""
fi

# 运行评估
echo -e "${BLUE}🚀 开始评估 (${NUM_EPISODES} episodes)${NC}"
echo ""

OUT_DIR="eval_results/latest"

if "${PYTHON_CMD[@]}" test_checkpoint.py "$LATEST_CHECKPOINT" "$NUM_EPISODES" --out "$OUT_DIR"; then
    echo -e "${GREEN}✅ 评估完成!${NC}"
    echo -e "${GREEN}📊 结果目录: ${OUT_DIR}${NC}"
    exit 0
else
    echo -e "${RED}❌ 评估失败${NC}"
    exit 1
fi

# EAGLE 多轮对话训练配置指南

## 📋 当前环境

- **Python**: 3.12.3
- **GPU**: NVIDIA RTX PRO 6000 (97GB)
- **位置**: /workspace/EAGLE

## ✅ 已有文件

所有训练代码已就绪：
- `eagle/traineagle3/main.py` - 主训练脚本（已修改支持多轮对话）
- `eagle/traineagle3/cnets.py` - 模型定义（已修复）
- `eagle/traineagle3/modeling_qwen3_kv.py` - Qwen3 模型（已修复）
- `eagle/traineagle3/config.json` - 模型配置（Qwen3-8B）
- `eagle/traineagle3/ds_config_test.json` - DeepSpeed 配置（BF16）

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install torch transformers deepspeed accelerate datasets safetensors wandb
```

### 2. 准备数据

创建 OpenAI 格式的训练数据（`train.jsonl`）：

```json
{"id": "sample_1", "messages": [
  {"role": "user", "content": "你好"},
  {"role": "assistant", "content": "你好！有什么可以帮助你的吗？"},
  {"role": "user", "content": "介绍一下你自己"},
  {"role": "assistant", "content": "我是一个AI助手..."}
]}
```

### 3. 准备模型

下载 Qwen3-8B 模型或使用已有模型路径

### 4. 启动训练

```bash
cd /workspace/EAGLE/eagle/traineagle3

deepspeed --num_gpus=1 main.py \
    --basepath /path/to/Qwen3-8B \
    --trainpath /path/to/train.jsonl \
    --testpath /path/to/test.jsonl \
    --savedir /workspace/output \
    --model_type qwen3 \
    --data_format openai \
    --deepspeed_config ds_config_test.json
```

## 📊 数据格式

### OpenAI 格式（推荐）

```json
{
  "id": "sample_1",
  "messages": [
    {"role": "user", "content": "问题1"},
    {"role": "assistant", "content": "回答1"},
    {"role": "user", "content": "问题2"},
    {"role": "assistant", "content": "回答2"}
  ]
}
```

### ShareGPT 格式

```json
{
  "id": "sample_1",
  "conversations": [
    {"from": "human", "value": "问题1"},
    {"from": "gpt", "value": "回答1"}
  ]
}
```

## ⚙️ 配置说明

### 修改训练轮数

编辑 `main.py` 第 26 行：
```python
"num_epochs": 2,  # 改为你需要的轮数
```

### 修改 batch size

编辑 `ds_config_test.json`：
```json
{
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 2
}
```

### 使用不同的模型

编辑 `config.json`，更新 `vocab_size` 等参数以匹配你的模型

## ✅ 已验证功能

- ✅ 多轮对话 Loss 计算（只对 assistant 回答计算）
- ✅ OpenAI/ShareGPT 数据格式支持
- ✅ Qwen3-8B 训练
- ✅ BF16 精度
- ✅ DeepSpeed ZeRO Stage 2

## 🔧 故障排除

### 显存不足

减小 batch size 或使用 ZeRO Stage 3：
```json
{
  "train_micro_batch_size_per_gpu": 1,
  "zero_optimization": {"stage": 3}
}
```

### 依赖缺失

```bash
pip install torch transformers deepspeed accelerate datasets safetensors
```

### 数据格式错误

确保：
- 使用 JSONL 格式（每行一个 JSON）
- 包含 `messages` 或 `conversations` 字段
- 使用正确的 `--data_format` 参数

---

**配置完成后即可开始训练！**


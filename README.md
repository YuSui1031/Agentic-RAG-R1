# Agentic RAG-R1 学习实践

> 基于 [jiangxinke/Agentic-RAG-R1](https://github.com/jiangxinke/Agentic-RAG-R1) 的学习复现
> 原论文：Agentic RAG-R1: Enhance Agentic RAG Reasoning Capacity via Reinforcement Learning

## 项目简介

Agentic RAG-R1 通过 GRPO 强化学习让语言模型自主掌握搜索与推理能力，构建 Agentic RAG 系统。本仓库是个人学习复现，用于理解 RL + RAG 的结合范式。

**核心技术点：**
- GRPO（Generalized Relevance Policy Optimization）强化学习训练
- Agent Memory Stack：规划、推理、回溯、总结、工具调用、结论
- 多工具检索：Wikipedia / 文档 / 知识图谱
- DeepSpeed ZeRO-2/3 分布式训练
- LoRA 高效微调

## 架构理解

```
┌─────────────────────────────────────────────────────┐
│                  Agent Memory Stack                  │
│                                                      │
│  Plan → Reasoning → Backtrack → Summary → Tool Call  │
│     ↑                                    │            │
│     └────────── Observation ─────────────┘            │
│                        ↓                              │
│                   Conclusion                          │
├─────────────────────────────────────────────────────┤
│                  训练流程 (GRPO)                      │
│                                                      │
│  Prompt → Rollout (推理+检索) → Reward → Policy Update │
│                                                      │
│  Reward = r_accuracy + r_format + r_rag              │
└─────────────────────────────────────────────────────┘
```

## 环境配置

```bash
conda create -n AgenticRAG python=3.11 -y
conda activate AgenticRAG
pip install -r requirements.txt
```

## 训练与推理

```bash
# 训练 (Zero-3)
./script/training/train_zero3.sh

# 推理服务
./script/run_server.sh
```

## 关键配置

- `.env` — API Key 配置（Elasticsearch、LLM、搜索）
- `src/config/` — 训练超参数
- `script/training/` — 训练启动脚本

## MedQA 评估结果

| 配置 | Format Accuracy | Answer Accuracy |
|------|----------------|-----------------|
| 微调前 | 39% | 84% |
| 微调前 + 搜索 | 56% | 79% |
| 微调后(200步) + 搜索 | 92% | 87% |

## 致谢

- 原项目：[jiangxinke/Agentic-RAG-R1](https://github.com/jiangxinke/Agentic-RAG-R1)
- 原论文作者：Xinke Jiang, Jiaran Gao 等（北京大学）
- 灵感来源：[DeepSeek-R1](https://arxiv.org/abs/2501.12948)、[TC-RAG](https://arxiv.org/abs/2408.09199)

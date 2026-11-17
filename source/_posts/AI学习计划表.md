---
title: AI学习计划表
date: 2026-11-16 17:13:10
tags: todolist
---

## 1. 大模型基础

- 从零搭建 GPT-2：[build-nanogpt](https://github.com/karpathy/build-nanogpt)、[配套视频](https://www.bilibili.com/video/BV12s421u7sZ/)
- SFT 实战：跑通 [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory.git)的任一Example

## 2. 强化学习与 Agentic RL

- 主线训练框架：
  - verl 解读：[姜富春文章](https://zhuanlan.zhihu.com/p/1931076626940139506)。关注数据加载预处理、模型适配/量化、算法 trainer、输入输出、评估指标和分布式流程。
  - AREAL解读：[AReaL源码阅读：一个大规模全异步强化学习系统](https://mp.weixin.qq.com/s/HwSmk0quiBgbnxoihfuUrw)
  - Reward Hacker现象：[Training a Misaligned Reward Seeker](https://alignment.anthropic.com/2026/reward-seeker/)
  - [Agent Lightning](https://github.com/microsoft/agent-lightning)、[Alibaba ROLL](https://github.com/alibaba/ROLL)、[verl](https://github.com/verl-project/verl)。学习 Agent 与训练系统解耦、rollout/trainer、多轮工具调用、异步与分布式训练。
  - 强化学习基础知识：[强化学习导论 | Hands-on Modern RL](https://walkinglabs.github.io/hands-on-modern-rl/preface/introduction)

- Agentic RL 全景：
  - 综述：[The Landscape of Agentic Reinforcement Learning for LLMs](https://arxiv.org/abs/2509.08827)、
  - RL 4 推理模型-算法概览：[Awesome-RL-for-LRMs](https://github.com/TsinghuaC3I/Awesome-RL-for-LRMs)


## 3. Agent 基础、Harness 与 MCP

- Agent基础
  - Agent 学习主线 Blog：[Agent Learning Hub](https://github.com/datawhalechina/Agent-Learning-Hub)
  - 基于Langchain构造生产级Agent：[Deep Agents 实战](https://github.com/datawhalechina/deepagents-in-action)

- Harness Engineering
  - 主线学习：[欢迎来到 Learn Harness Engineering | Learn Harness Engineering](https://walkinglabs.github.io/learn-harness-engineering/zh/)
  - 相关文章：[刚刚，一篇最全Agent Harness综述来了！](https://mp.weixin.qq.com/s/pG39PRnZFjSIxwYcPKD47A)、[Agent Harness六大组件解析](https://mp.weixin.qq.com/s/HwqEaXSGkcYgUNrzB2okuA)
  - MCP开发教程：[MCP 开发教程](https://mp.weixin.qq.com/s/8qB8xcp8d3yYXnGZNlutEA)

- 现成Harness框架
  - OpenClaw：[李宏毅讲解](https://mp.weixin.qq.com/s/B0xYl5EvSXUNc-wP1EMVGw)、[腾讯代码框架解读](https://mp.weixin.qq.com/s/Q0CC0p5e-DQEYEk5ErQB1Q)、[阿里云上篇](https://mp.weixin.qq.com/s/wVcItgqsCiwl9-PZ56z27w)、[阿里云下篇](https://mp.weixin.qq.com/s/FUJEofqbK7vX-J64UX8Nkg)
  - Hermes：[Hermes Agent 介绍](https://mp.weixin.qq.com/s/vUKXk0Maj6TvuD6BPLY2bQ)
  - Claude Code：[learn-claude-code](https://github.com/shareAI-lab/learn-claude-code)、[源码学习仓库](https://github.com/openedclaude/claude-reviews-claude?tab=readme-ov-file)、[源码解读](https://mp.weixin.qq.com/s/LbvqZLhvtah31O3aVFGXcg)、[简易实现](https://github.com/Windy3f3f3f3f/claude-code-from-scratch)
  - Codex：[源码](https://github.com/openai/codex)、[解读一](https://mp.weixin.qq.com/s/fbKi7IrAvmv49IfJbI9w7Q)、[解读二](https://blog.csdn.net/aidoudoulong/article/details/163941019)
  - DeepSeek Harness：[代码](https://github.com/deepseek-ai/deepseek-harness)、[桌面版与 CLI](https://mp.weixin.qq.com/s/uIFETPQyOY59gt7dNGKgHQ)、[插件解读](https://mp.weixin.qq.com/s/r4C-RR1CthFooA7iZl6hzg)

- Agent架构流
  - [登上GitHub Trending｜腾讯开源 LoopForge](https://mp.weixin.qq.com/s/-gagDmalGrKyTtA-1dc8BQ)
  - SuperAgent：[DeepSeek SuperAgent 介绍](https://mp.weixin.qq.com/s/LZxAxf0EO74YHQnZ2YOT9A)、[Agent 自动化系统](https://github.com/KKKKhazix/agent-autoloop)


## 4. 最新LLM应用

- jev决策模型：
  - 解读主线：[jev-cookbook](https://github.com/datawhalechina/jev-cookbook)
  - OpenAI 正式发布 Decisions API：[GPT-Decision API](https://aihot.news/items/xffhf8h0e0r145getmi9k1zgs)
  - Jev版OCR模型：[MiniCPM-OCR 发布介绍](https://mp.weixin.qq.com/s/khZfzPQjmCd1YQB7egy9gA)
- RSI自进化：
  - RSI综述：[这是一篇把"RSI"讲明白的科普级综述！](https://mp.weixin.qq.com/s/rlfTKyWhALsNONhAwGih1A)
  - [RSIHub](https://github.com/simple-agent-lab/RSIHub)、[腾讯RSR论文](https://arxiv.org/pdf/2610.02826v1)
- Agentic RAG：
  - 概念解读：[知乎：深入浅出AgenticRAG](https://zhuanlan.zhihu.com/p/2000499887679218395)
  - Mistral 发布 Agentic Search：[Mistral 发布 Agentic Search，多步检索在 FinanceBench 上将准确率从 26.7% 提升至 86% · AIHOT](https://aihot.news/items/cmt1pkwbj04bxroovzkfca5c7)、[Agentic Search | Mistral Docs](https://docs.mistral.ai/studio/search/agentic-search)
- AI infra：
  - SGLang：[zero-to-sglang](https://github.com/datawhalechina/zero-to-sglang)
  - DeepSeek开源：[DeepInfer](https://github.com/deepseek-ai/DeepInfer/tree/main/)
  - 蚂蚁开源Humming：[Humming 介绍](https://aihot.news/items/afv7us0zbvzo789rghqkfhq5q)、[Humming 项目](https://github.com/vllm-project/humming)

## 初步学习规划

| 时间      | 学习内容                                                     |
| --------- | ------------------------------------------------------------ |
| 第 1 天   | **大模型基础**：build-nanogpt、配套视频、LLaMA-Factory SFT   |
| 第 2–3 周 | **强化学习与 Agentic RL**：verl、AReaL、Reward Hacker、Agent Lightning、Alibaba ROLL、Hands-on Modern RL、Agentic RL Survey、Awesome-RL-for-LRMs |
| 第 4–5 周 | **Agent 基础、Harness 与 MCP**：Agent Learning Hub、Deep Agents、Learn Harness Engineering、Harness 综述、MCP 教程 |
| 第 6 周   | **Agent 项目与源码**：OpenClaw、Hermes、Claude Code、Codex、DeepSeek Harness、LoopForge、SuperAgent、agent-autoloop |
| 第 7 周   | **最新 LLM 应用**：Jev、Decisions API、MiniCPM-OCR、RSI、RSIHub、腾讯 RSR、Agentic RAG、Mistral Agentic Search |
| 第 8 周   | **AI Infra**：zero-to-sglang、DeepInfer、Humming             |

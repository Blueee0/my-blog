---
title: AI学习计划表
date: 2026-11-16 17:13:10
tags:
---

## 1. 大模型基础与微调

- 从零搭建 GPT-2：[build-nanogpt](https://github.com/karpathy/build-nanogpt)、[配套视频](https://www.bilibili.com/video/BV12s421u7sZ/)
- 大模型案例：[jev-cookbook](https://github.com/datawhalechina/jev-cookbook)
- SFT 实战：跑通 [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory.git)，结合 [PEFT](https://github.com/huggingface/peft)。
- 训练数据与输入：`datasets.Dataset`、DataCollator、Tokenizer、Chat Template、padding/truncation。
- 模型与训练流程：AutoModelForCausalLM、forward、loss、labels、training_step、optimizer、scheduler、gradient accumulation、mixed precision、checkpoint。
- LoRA / QLoRA：精读 LoraConfig、inject_adapter、target_modules、lora_A/lora_B、merge_and_unload；理解 4-bit weight、NF4、double quant、paged optimizer。

## 2. 强化学习与 Agentic RL

- RL 基础：[hands-on-modern-rl](https://github.com/walkinglabs/hands-on-modern-rl)
- Agentic RL 入门与复现：[Datawhale Agentic RL Lab](https://github.com/datawhalechina/agentic-rl-lab)、[KMnO4-zx Agentic RL Lab](https://github.com/KMnO4-zx/agentic-rl-lab/tree/main)
- 学习路径：0-loss → GRPO → Search-R1 → ReTool → ALFWorld；重点理解 GRPO 的 group sampling、advantage、policy update。
- 主线训练框架：[Agent Lightning](https://github.com/microsoft/agent-lightning)、[Alibaba ROLL](https://github.com/alibaba/ROLL)、[verl](https://github.com/verl-project/verl)。学习 Agent 与训练系统解耦、rollout/trainer、多轮工具调用、异步与分布式训练。
- verl 解读：[姜富春文章](https://zhuanlan.zhihu.com/p/1931076626940139506)。关注数据加载预处理、模型适配/量化、算法 trainer、输入输出、评估指标和分布式流程。
- Tool Use RL：[Search-R1](https://github.com/PeterGriffinJin/Search-R1)、[ReTool](https://arxiv.org/abs/2504.11536)、[Tool-Star](https://github.com/RUC-NLPIR/Tool-Star)。
- 环境与评测：[τ²-bench](https://github.com/sierra-research/tau2-bench)、[BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html)、[AgentGym-RL](https://github.com/WestlakeNLP/AgentGym-RL)
- GRPO 数据：[FinRAG-GRPO](https://huggingface.co/datasets/SamWang0405/FinRAG-GRPO)
- 奖励与对齐：[Training a Misaligned Reward Seeker](https://alignment.anthropic.com/2026/reward-seeker/)
- OPD 在线策略蒸馏：[PROOF-Gen](https://arxiv.org/abs/2608.23911)
- Agentic RL 全景：[The Landscape of Agentic Reinforcement Learning for LLMs](https://arxiv.org/abs/2509.08827)、[Awesome-RL-for-LRMs](https://github.com/TsinghuaC3I/Awesome-RL-for-LRMs)
- FinQA Agent：[rllm](https://github.com/rllm-org/rllm)、[FinQA 项目介绍](https://rllm-project.com/post.html?post=finqa.md)。
- OPD 实例：[HealthAI-2025](https://github.com/yuandaxia2001/HealthAI-2025)

## 3. Agent 基础、Harness 与 MCP

- Agent 基础：[Agent Learning Hub](https://github.com/datawhalechina/Agent-Learning-Hub)、[Deep Agents 实战](https://github.com/datawhalechina/deepagents-in-action)
- Harness 学习：[learn-harness-engineering](https://github.com/walkinglabs/learn-harness-engineering)、[awesome-agent-harness 中文实践](https://github.com/Picrew/awesome-agent-harness/blob/main/README_zh.md)
- Harness 文章：[工程实践](https://mp.weixin.qq.com/s/1-bPS2omne-tncLTPm5N7A)、[Agent Harness 综述](https://mp.weixin.qq.com/s/pG39PRnZFjSIxwYcPKD47A)、[Harness 革命](https://mp.weixin.qq.com/s/0CTwb4aEr5mWwsdRdwzwkw)、[Harness Engineering vs TDD](https://mp.weixin.qq.com/s/Laz4W0180y9yGW0b6EpUMQ)、[六大组件解析](https://mp.weixin.qq.com/s/HwqEaXSGkcYgUNrzB2okuA)
- MCP 开发：[教程](https://mp.weixin.qq.com/s/8qB8xcp8d3yYXnGZNlutEA)
- Harness 项目：[DeepSeek Harness Datawhale 代码](https://mp.weixin.qq.com/s/HwSmk0quiBgbnxoihfuUrw)、[Autonomous Marketing Agent](https://app.notion.com/p/Autonomous-Marketing-Agent-3a734f9e37af80d7a6bbc2e30925523f?pvs=21)
- Agent-Harness / Agentic RL 集中实践计划：2026-11-23 至 2026-12-11。
- Claude Code：[源码学习仓库](https://github.com/openedclaude/claude-reviews-claude?tab=readme-ov-file)、[源码解读](https://mp.weixin.qq.com/s/LbvqZLhvtah31O3aVFGXcg)、[简易实现](https://github.com/Windy3f3f3f3f/claude-code-from-scratch)
- Codex：[源码](https://github.com/openai/codex)、[解读一](https://mp.weixin.qq.com/s/fbKi7IrAvmv49IfJbI9w7Q)、[解读二](https://blog.csdn.net/aidoudoulong/article/details/163941019)
- DeepSeek Harness：[代码](https://github.com/deepseek-ai/deepseek-harness)、[桌面版与 CLI](https://mp.weixin.qq.com/s/uIFETPQyOY59gt7dNGKgHQ)、[插件解读](https://mp.weixin.qq.com/s/r4C-RR1CthFooA7iZl6hzg)
- Long-running Agent：[learn-claude-code](https://github.com/shareAI-lab/learn-claude-code)、[飞书笔记](https://my.feishu.cn/wiki/LZxRwWvjriQLNgk8RCmcekVunwh)
- 相关项目：[GenericAgent](https://github.com/lsdefine/GenericAgent)

## 4. RAG、Multi-Agent 与 Deep Research

- Agentic RAG：[概览文章](https://zhuanlan.zhihu.com/p/2000499887679218395)、[Agentic RAG for Dummies](https://github.com/GiovanniPasq/agentic-rag-for-dummies)
- Agentic Search：[Mistral 介绍](https://mistral.ai/news/agentic-search/)、[官方文档](https://docs.mistral.ai/studio/search/agentic-search)、[AIHOT 资料一](https://aihot.virxact.com/items/cmt1pkwbj04bxroovzkfca5c7)、[AIHOT 资料二](https://aihot.virxact.com/items/cmta64hk303hqroj2edkgrcxi)
- GraphRAG：[架构教程](https://datawhalechina.github.io/all-in-rag/#/chapter8/01_env_architecture)、[配套代码](https://github.com/datawhalechina/all-in-rag/tree/main/code/C8)、[Microsoft GraphRAG](https://github.com/microsoft/graphrag)
- 知识图谱问答案例：[电商知识图谱项目](https://github.com/popo0625/dianshang/tree/master)
- 金融研报问答 RAG：学习 PDF 版面/表格解析、切块、混合检索、重排、引用溯源与离线评测。参考 [RAGFlow](https://github.com/infiniflow/ragflow)、[FAISS](https://github.com/facebookresearch/faiss)、[FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)、[LangChain-Chatchat](https://github.com/chatchat-space/Langchain-Chatchat)、[QAnything](https://github.com/netease-youdao/QAnything)、[LlamaIndex](https://github.com/run-llama/llama_index)、[Milvus](https://github.com/milvus-io/milvus)。
- 检索数据集参考：[Nepali e-commerce retrieval](https://huggingface.co/datasets/jangedoo/nepali-ecommerce-retrieval)
- Multi-Agent Deep Research 项目：[Deep Research Agent](https://app.notion.com/p/Deep-Research-Agent-31634f9e37af80be9c53cde3133e2e1d?pvs=21)、[Alibaba DeepResearch](https://github.com/Alibaba-NLP/DeepResearch/?tab=readme-ov-file)、[MiroFlow](https://github.com/MiroMindAI/MiroFlow/)、[DeepTutor](https://github.com/HKUDS/DeepTutor)、[DeepTutor 技术介绍](https://www.xiaohongshu.com/explore/69665bcc000000000d00a13c?app_platform=android&ignoreEngage=true&app_version=9.19.0&share_from_user_hidden=true&xsec_source=app_share&type=normal&xsec_token=CBP23KYncfC3o_6po3EtJ0r6Vtm1MtucwMBeRkbXx1jvw=&author_share=1&xhsshare=&shareRedId=OD4yRUg9NDo2NzUyOTgwNjZIOTg1SUxL&apptime=1770004831&share_id=c657713b7fbc4c16b960b48519432672&share_channel=wechat&wechatWid=7ef479ee475878ba25fe82bf6753a77f&wechatOrigin=menu)、[DeepTutor 简介](https://mp.weixin.qq.com/s/qj4L0MoH5X05gQRMU7PSOg?mpshare=1&scene=1&srcid=0129rZ9AuuBLkTz8TUGtK0KN&sharer_shareinfo=ce6299ee66f974a64ebf73293356fe5d&sharer_shareinfo_first=ce6299ee66f974a64ebf73293356fe5d#wechat_redirect)
- Research Skills / Tools：[Hugging Face Daily Papers](https://huggingface.co/papers)

## 5. Agent Memory 与 RSI

- Agent Memory：[easy-data-x-ai](https://github.com/datawhalechina/easy-data-x-ai)、[Funes](https://github.com/huggingface/funes)
- RSI：[RSIHub](https://github.com/simple-agent-lab/RSIHub)、[论文](https://arxiv.org/pdf/2610.02826v1)、[相关介绍](https://mp.weixin.qq.com/s/rlfTKyWhALsNONhAwGih1A)

## 6. AI Infra 与推理

- SGLang：[zero-to-sglang](https://github.com/datawhalechina/zero-to-sglang)
- 推理框架：[DeepInfer](https://github.com/deepseek-ai/DeepInfer/tree/main/)、[Humming 项目](https://github.com/vllm-project/humming)、[Humming 介绍](https://aihot.news/items/afv7us0zbvzo789rghqkfhq5q)

## 7. Agent 产品与代码案例

- OpenClaw 部署与技能：[OpenClaw](https://github.com/openclaw/openclaw)、[ClawHub](https://clawhub.ai/skills?sort=downloads&nonSuspicious=true)、[部署教程](https://mp.weixin.qq.com/s/bfr61EI4Z1P0rKDH-mYhMQ)、[卸载指南](https://mp.weixin.qq.com/s?__biz=MzIyNjM2MzQyNg==&mid=2247720809&idx=1&sn=f9400fc808958a3a57e63a54d5af75ab&chksm=e9fd98900996b65ff7da0e6507e91b166080b20604eba6436ce1aa3a026101cc8a45d36f29c4&mpshare=1&scene=1&srcid=0317U0DVZrrrXfcF9sXGpL2o&sharer_shareinfo=781119fdb311eb9ea7d5b9708b19d61c&sharer_shareinfo_first=781119fdb311eb9ea7d5b9708b19d61c#rd)
- OpenClaw 架构：[李宏毅讲解](https://mp.weixin.qq.com/s/B0xYl5EvSXUNc-wP1EMVGw)、[腾讯代码框架解读](https://mp.weixin.qq.com/s/Q0CC0p5e-DQEYEk5ErQB1Q)、[阿里云上篇](https://mp.weixin.qq.com/s/wVcItgqsCiwl9-PZ56z27w)、[阿里云下篇](https://mp.weixin.qq.com/s/FUJEofqbK7vX-J64UX8Nkg)
- Hermes：[Hermes Agent 介绍](https://mp.weixin.qq.com/s/vUKXk0Maj6TvuD6BPLY2bQ)
- OCR：[MiniCPM-OCR 发布介绍](https://mp.weixin.qq.com/s/khZfzPQjmCd1YQB7egy9gA)
- 决策模型：[GPT-Decision API](https://aihot.news/items/xffhf8h0e0r145getmi9k1zgs)
- SuperAgent：[DeepSeek SuperAgent 介绍](https://mp.weixin.qq.com/s/LZxAxf0EO74YHQnZ2YOT9A)、[Agent 自动化系统](https://github.com/KKKKhazix/agent-autoloop)
- 零信任智能体 Agent4SQL：[介绍](https://aihot.virxact.com/items/cmsxv0p3v0bntroz0227a6b3j)、[Google Cloud ADK 示例](https://github.com/GoogleCloudPlatform/generative-ai/tree/main/agents/adk/zero-trust-agents)
- 电商智能体：[设计分析](https://claude.com/blog/the-anatomy-of-effective-commerce-agents)、[Anthropic commerce-agents](https://github.com/anthropics/commerce-agents)
- Agent 工程模式：[双向 MCP、事件驱动并发、fallback 与分层路由](https://aihot.virxact.com/items/cmtkbbn6q01j6roz54e8g5zec#detail-section-3)
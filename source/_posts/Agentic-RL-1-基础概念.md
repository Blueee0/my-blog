---
title: Agentic-RL-1-基础概念
date: 2026-11-17 14:51:48
tags:agentic rl
---

## 基本概念

Agentic RL 训练的是模型完成一次任务的过程，而不只是一个回答。

- 环境：有状态的外部系统，比如搜索引擎，文本游戏
- 动作：LLM 根据当前上下文生成一段文本或一个工具调用
- rollout / 轨迹：`prompt → action_1 → observation_1 → action_2 → ... → action_T → reward`
- 训练时loss：整条轨迹被拼成一个 token 序列送进模型，初始prompt和环境返回的observation要mask掉，参与，只有模型自己的动作token参与训练。
- 奖励reward：奖励通常在轨迹结束时才出现，再由 GRPO、PPO 这类算法转成 advantage 回传。

|                    | 单轮 RLVR                           | Agentic RL                                            |
| :----------------- | :---------------------------------- | ----------------------------------------------------- |
| 状态               | prompt 加上已生成的 token           | 再加一层模型看不见、只能通过工具感知的环境状态        |
| 一次 rollout       | 生成一次                            | 生成多次，中间穿插工具调用                            |
| 参与 loss 的 token | 模型生成的全部 token（prompt mask） | 只有模型的动作 token（prompt 和 observation 都 mask） |
| 奖励               | 答案对错                            | 任务成败，通常更稀疏、更延迟                          |

轨迹变长之后，训练要额外处理三件事：

- **信用分配**：奖励在终点才出现，组内相对 advantage 无法指出中间哪一步真正起了作用。TEMPO 把长轨迹切成小段，训练生成式 critic 在段末估值；AgentOPSD让 teacher 重看整条轨迹，为每个 turn 补监督。
- **上下文就是状态**：环境状态不会自己进入模型，只能被 observation 带进上下文。保留多少历史、什么时候截断，会直接影响模型能学到什么。
- **采样成本**：每一步动作都可能调用外部服务，rollout 比单轮慢，也更容易失败。

---

## Loss Function

强化学习的Loss本质上是：如果某些 token 后来被证明对结果有帮助，就让模型下次更容易生成它们；如果某些 token 拖了后腿，就让模型下次少生成它们。

- `advantage`：这一步是好是坏。
- `sampling_logprobs`：生成答案时，采样模型当时有多喜欢这个 token。
- `target_logprobs`：训练时，当前模型现在有多喜欢这个 token。

### Importance Sampling

当前模型和采样模型对同一个 token 的概率比例是多少？——最大化`ratio * advantage`

$$ \mathcal{L}_{\text{IS}}(\theta) = -\mathbb{E}_{x \sim q}\left[ \frac{p_\theta(x)}{q(x)} A(x) \right] $$

实现时再把概率比例写成 `exp(target_logprobs - sampling_logprobs)`。

```python
prob_ratio = torch.exp(target_logprobs - sampling_logprobs)
loss = -(prob_ratio * advantages).sum()	# 因为优化器最小化 loss，所以代码里加负号
```

- 如果 advantage 是正的，说明这个 token 有用，训练会推高它的概率。
- 如果 advantage 是负的，说明这个 token 没用，训练会压低它的概率。

### PPO

PPO 给 ratio 加了一个 clip。

$$ \mathcal{L}_{\text{PPO}}(\theta) = -\mathbb{E}_{x \sim q}\left[\min\left(\frac{p_\theta(x)}{q(x)}\cdot A(x), \text{clip}\left(\frac{p_\theta(x)}{q(x)}, 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}}\right)\cdot A(x)\right)\right] $$

比如一个 token 的 advantage 很高，ratio 已经到了 `1.8`，说明当前模型已经比采样时更喜欢它很多。PPO 不会继续按 `1.8` 的力度推，而是把它限制到比如 `1.2`。这就是 PPO 的核心直觉：可以学习，但别一步冲太远。

### CISPO

把这个 ratio 当成固定系数，去乘 `logprob * advantage`。

$$ \mathcal{L}_{\text{CISPO}}(\theta) = -\mathbb{E}_{x \sim q}\left[\text{sg}\left(\text{clip}\left(\frac{p_\theta(x)}{q(x)}, 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}}\right)\right) \cdot \log p_\theta(x) \cdot A(x)\right] $$

这里最关键的是 `detach`。它表示 clipped ratio 不再参与梯度回传，只作为一个固定权重。真正被优化的是 `target_logprob`。

所以 CISPO 可以理解成：给每个 token 的学习力度加一个上限，但不直接把某些 token 的梯度剪没。

---

## 学习路线

### 01 · 理解一次策略更新

**[Loss Functions](https://datawhalechina.github.io/agentic-rl-lab/experiments/loss-functions/) → [GRPO](https://datawhalechina.github.io/agentic-rl-lab/experiments/grpo/) → [DAPO](https://datawhalechina.github.io/agentic-rl-lab/experiments/dapo/) → [GSPO](https://datawhalechina.github.io/agentic-rl-lab/experiments/gspo/)**

先理解 logprob、重要性比率与 advantage。可以先用[GRPO、DAPO 与 GSPO 对照](https://datawhalechina.github.io/agentic-rl-lab/guide/grpo-dapo-gspo.html)了解这三个章节各自覆盖的内容。用 GRPO 走通一次训练循环后，再观察动态采样、裁剪与序列级目标分别改变了什么。

### 02 · 把教师反馈变成学习信号

**[General OPD](https://datawhalechina.github.io/agentic-rl-lab/experiments/general-opd/) → [Medical OPD](https://datawhalechina.github.io/agentic-rl-lab/experiments/medical-opd/) → [OPSD](https://datawhalechina.github.io/agentic-rl-lab/experiments/opsd/)**

从 Student 采样与 Teacher 打分开始，理解 token 级反馈。接着看专业能力与通用能力怎样组织在同一次训练里，再探索 self-distillation 的实现。

### 03 · 让模型与外部世界交互

**[Search-R1](https://datawhalechina.github.io/agentic-rl-lab/experiments/search-r1/) → [ReTool](https://datawhalechina.github.io/agentic-rl-lab/experiments/retool/) → [ALFWorld](https://datawhalechina.github.io/agentic-rl-lab/experiments/alfworld/)**

依次接入搜索、Python 沙箱与 TextWorld 环境。重点观察多轮轨迹如何保存、环境返回如何进入上下文，以及训练时怎样区分模型动作与环境观察。

接着可以看 [Tournament-GRPO](https://datawhalechina.github.io/agentic-rl-lab/experiments/tournament-grpo/)：当模型搜索后写出的研究报告没有标准答案时，怎样用裁判比较和淘汰赛胜场构造奖励？

### 04 · 探索更复杂的任务

已经理解 GRPO 与工具交互后，可以从下面几个方向继续：

| 研究方向           | 阅读顺序                                                     | 重点问题                                         |
| ------------------ | ------------------------------------------------------------ | ------------------------------------------------ |
| 多模态工具使用     | [Vision GRPO](https://datawhalechina.github.io/agentic-rl-lab/experiments/vision-grpo/) → [Spec-o3](https://datawhalechina.github.io/agentic-rl-lab/experiments/spec-o3/) | 图片与工具结果怎样共同进入训练轨迹？             |
| 长程学习与信用分配 | [TEMPO](https://datawhalechina.github.io/agentic-rl-lab/experiments/tempo/) → [AgentOPSD](https://datawhalechina.github.io/agentic-rl-lab/experiments/agentopsd/) | 长轨迹中的哪一段、哪一轮应该得到学习信号？       |
| 固定选项的分类决策 | [PyTrio-Jev](https://datawhalechina.github.io/agentic-rl-lab/experiments/pytrio-jev/) | 怎样用一次前向得到候选概率，再检查和校准置信度？ |

TEMPO 章节是基于公开技术博客、在 ALFWorld 中进行的算法级复现。每个方法的环境、预算与实现边界都在正文中说明，阅读结果时也请一起查看。

## 用实验检验自己的理解

每完成一篇，试着回答四个问题：

1. 奖励从哪里来，能否验证？
2. 哪些 token 或交互轮次参与了优化？
3. 这次实验实际改变了什么变量？
4. 哪些结论还需要更充分的对照？

对比实验时，保留模型名称、checkpoint、数据划分、随机种子、采样预算和评测命令。先对齐这些条件，再讨论方法的效果。

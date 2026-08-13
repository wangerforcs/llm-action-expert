# Idea 大纲：LLM + Action Expert 的双塔 Agent 架构

## 1. 核心想法

当前大多数 LLM Agent 都让同一个 autoregressive LLM 同时承担：

* 任务理解
* 上下文建模
* reasoning / planning
* tool selection
* tool argument generation
* 最终自然语言回复

核心问题是：

> **语言生成和动作生成是否真的应该共享同一个 decoder？**

受到 VLA 双塔架构启发，可以将 Agent 拆成：

[
\boxed{
\text{LLM Semantic Backbone}
+
\text{Specialized Action Expert}
}
]

其中：

* **LLM** 负责理解用户目标、历史、环境 observation、tool schema，并形成语义表示；
* **Action Expert** 直接读取 LLM 的内部 hidden state / KV cache，专门生成 tool call；
* LLM 不必在每一步 tool action 前显式生成 reasoning token 或 tool-call token。

核心假设：

> **LLM should understand actions, but need not generate actions.**

---

## 2. 与现有 Agent 的区别

### 传统 Agent

```text
Context
   ↓
Large LLM
   ↓
Reasoning tokens
   ↓
Tool-call tokens
   ↓
Tool
   ↓
Observation
   ↓
Large LLM
```

即：

[
P(a_t|x_{\le t})
]

直接由语言模型的 autoregressive decoder 完成。

### 提议架构

```text
                Context
                   │
                   ▼
          ┌────────────────┐
          │  LLM Backbone  │
          │ semantic model │
          └───────┬────────┘
                  │
            layer-wise KV
                  │
                  ▼
          ┌────────────────┐
          │ Action Expert  │
          │ tool policy    │
          └───────┬────────┘
                  │
                  ▼
               Tool Call
                  │
                  ▼
                Tool
                  │
             Observation
                  │
                  ▼
          incremental prefill
```

关键变化：

[
\text{LLM generation}
\rightarrow
\text{tool call}
]

变为：

[
\text{LLM representation}
\rightarrow
\text{Action Expert}
\rightarrow
\text{tool call}
]

---

## 3. VLA 启发

类似 VLA 中：

```text
Vision-Language Backbone
          │
          │ KV cache
          ▼
     Action Expert
          │
          ▼
      Robot Action
```

Agent 中对应为：

```text
Language Backbone
        │
        │ KV cache
        ▼
   Action Expert
        │
        ▼
     Tool Action
```

真正借鉴的不是 flow matching 本身，而是：

> **共享 semantic backbone，分离 language generation 与 action generation。**

因此“双塔”和 flow matching 不绑定。

Action Expert 可以使用：

* autoregressive decoding
* masked denoising
* discrete diffusion
* latent diffusion / flow matching
* structured prediction

---

## 4. Backbone–Expert 接口

LLM 对完整 Agent context 做 forward：

[
x_t=
[
goal,
history,
tool\ schemas,
observations,
memory
]
]

得到每层：

[
{K_l^{LLM},V_l^{LLM}}_{l=1}^{L}
]

Action Expert 第 (l) 层产生自己的：

[
Q_l^A=W_{Q,l}^A H_l^A
]

并读取 LLM representation：

[
H_{l+1}^A
=========

Attention
(
Q_l^A,
[K_l^{LLM};K_l^A],
[V_l^{LLM};V_l^A]
)
]

因此：

> LLM KV cache 可以被视为 Agent 当前的 latent semantic state。

Action Expert 不需要先得到自然语言形式的：

```text
"Next subgoal: inspect auth.py"
```

而是直接从 latent representation 中生成：

```text
read_file("src/auth.py")
```

---

## 5. Action Expert 的任务定义

Action Expert 学习：

[
P(a_t|KV_t,\mathcal T)
]

其中：

* (KV_t)：当前上下文的 LLM representation
* (\mathcal T)：当前 available tools
* (a_t)：下一步 action

Action 可以进一步拆成：

[
a_t=
(
mode,
tool,
arguments
)
]

例如：

```text
mode = TOOL_CALL
tool = grep
arguments = {
    pattern: "refresh_token",
    path: "src/"
}
```

也可以包含：

```text
TOOL_CALL
RESPOND
ASK_USER
STOP
```

使 Action Expert 不只是 tool selector，而是真正的 Agent policy。

---

## 6. 第一阶段：AR Action Expert

最简单、最重要的 baseline：

```text
Frozen / pretrained LLM
          │
          │ per-layer KV
          ▼
  Small Transformer
    Action Expert
          │
          ▼
 autoregressive
   tool tokens
```

例如：

```text
Backbone:
7B / 14B LLM

Action Expert:
100M–500M Transformer
```

训练目标：

[
\mathcal L_{action}
===================

-\sum_i
\log
P(a_i|a_{<i},KV)
]

第一阶段先不引入：

* world model
* critic
* multi-agent
* complex planning
* flow matching

目标是干净验证：

> **一个专门 Action Expert 是否能替代大 LLM 的 tool-call decoding？**

---

## 7. 第二阶段：非 AR Action Generation

如果 AR expert 验证成功，再研究 action generation objective。

### 7.1 Masked / discrete denoising

初始：

```text
[MASK] [MASK] [MASK] [MASK]
```

逐步 refinement：

```text
grep [MASK] refresh_token [MASK]
```

最终：

```text
grep pattern=refresh_token path=src/
```

优势：

> tool、arguments 可以联合推断，而不是严格 left-to-right。

### 7.2 Structured Action Slots

将 action 表示为：

[
A=
[
a_{mode},
a_{tool},
a_{arg1},
...,
a_{argK}
]
]

不同 slot 使用不同 prediction head：

* tool → categorical
* enum → categorical
* number → regression
* text → small token decoder

形成真正的 mixed action space。

### 7.3 Flow / Latent Action

进一步探索：

[
z_1 \sim \mathcal N(0,I)
]

经过 Action Expert：

[
z_1
\rightarrow
z_{0.8}
\rightarrow
z_{0.4}
\rightarrow
z_0
]

最后 decode 成 structured tool action。

这一部分属于后续研究，而不是项目起点。

---

## 8. Agent Loop 的变化

一个非常重要的目标是：

> **LLM 可以只做 prefill，而不必在每个 action step 做 decode。**

传统流程：

```text
Observation
   ↓
LLM prefill
   ↓
LLM reasoning decode
   ↓
LLM tool-call decode
   ↓
Tool
```

新流程：

```text
Observation
   ↓
LLM incremental prefill
   ↓
updated KV
   ↓
Action Expert
   ↓
Tool
```

即：

[
KV_{t+1}
========

f_{LLM}(KV_t,o_{t+1})
]

然后：

[
a_{t+1}
=======

g_{expert}(KV_{t+1})
]

只有当需要：

* 最终回答用户
* 显式 reasoning
* 高层 replanning
* clarification

时，才调用 LLM language head。

---

## 9. 主要研究问题

### RQ1：Action specialization 是否有效？

在同一个 semantic backbone 下：

[
LLM\ Tool\ Decoder
\quad vs \quad
Action\ Expert
]

谁的 tool accuracy 更高？

### RQ2：是否可以减少 Large LLM decode？

重点比较：

[
\frac{\text{Large LLM decode tokens}}
{\text{Tool actions}}
]

理想情况下：

```text
Baseline:
每个 action 都需要 LLM decode

Ours:
大多数 action 只需要 Action Expert
```

### RQ3：Action Expert 多小仍然有效？

比较：

```text
100M
300M
500M
1B
```

寻找能力 / FLOPs trade-off。

### RQ4：KV 是否足够承载 latent reasoning？

比较：

```text
textual subgoal → Action Expert

vs.

last-layer hidden state → Action Expert

vs.

per-layer KV → Action Expert
```

### RQ5：Tool action 是否适合非 AR decoding？

比较：

```text
AR
vs.
masked denoising
vs.
structured slots
vs.
flow / latent generation
```

---

## 10. 实验设置

第一阶段应选择 action space 较小、可自动评估的环境。

### Coding Agent

Tools：

```text
search
read
edit
shell
test
```

优点：

* action 清晰
* trajectory 长
* outcome 可验证
* tool schema 稳定
* 很适合研究 long-horizon action generation

### Browser Agent

Tools：

```text
click
type
scroll
navigate
select
back
```

它与 VLA 的对应尤其直接：

```text
VLA:
observation + intent → physical action

Browser Agent:
page observation + intent → UI action
```

---

## 11. Baseline

至少比较：

### Baseline A：标准 Function Calling

```text
Large LLM
→ reasoning
→ tool call
```

### Baseline B：Textual Dual Model

```text
Large LLM
→ textual subgoal
→ Small LLM
→ tool call
```

### Proposed C：KV Action Expert

```text
Large LLM backbone
→ KV
→ Action Expert
→ tool call
```

### Proposed D：KV + Non-AR Expert

```text
Large LLM backbone
→ KV
→ denoising / structured expert
→ tool call
```

---

## 12. 评价指标

任务层：

[
Success\ Rate
]

动作层：

[
Tool\ Selection\ Accuracy
]

[
Argument\ Accuracy
]

[
Invalid\ Tool\ Call\ Rate
]

效率：

[
Latency
]

[
FLOPs
]

[
KV\ memory
]

[
Large\ LLM\ Decode\ Tokens
]

最关键的 efficiency metric：

[
\boxed{
\text{Actions per Large-LLM Decode}
}
]

以及：

[
\boxed{
\text{Task Success per FLOP}
}
]

---

## 13. 主要挑战

### 1. Tool action 是混合离散空间

不像机器人连续动作天然适合 flow matching。

因此应该先验证 architecture，再验证 generation objective。

### 2. Agent context 很长

Action Expert 直接读取完整 KV：

[
O(N_{context}N_{action})
]

可能产生较高 cross-attention 成本。

未来可以研究：

```text
Full KV
   ↓
Action-relevant KV selection
   ↓
Action Expert
```

### 3. LLM KV 是否真的包含足够的 action information

语言模型的 latent space 是为 next-token prediction 学到的，不一定天然适合 action policy。

可能需要：

* adapter
* projection
* action tokens
* joint finetuning
* auxiliary action loss

### 4. Observation 更新

Tool observation 到来后，仍需要 backbone incremental prefill。

因此主要节省的是：

> expensive large-model autoregressive decoding

而不是完全消除 backbone computation。

---

## 14. 最小可行实验（MVP）

第一版：

```text
Pretrained LLM 7B
      │
      │ frozen
      │
   per-layer KV
      │
      ▼
300M Action Expert
      │
      ▼
AR Tool Call
```

Domain：

```text
Coding Agent
```

训练：

```text
Agent trajectory
→
(context KV, next tool call)
```

只训练 Action Expert。

比较：

```text
Original 7B function calling
vs.
7B frozen backbone + 300M Action Expert
```

首先验证三个问题：

1. success rate 能否接近 baseline；
2. tool selection / argument accuracy 能否接近甚至超过 baseline；
3. large-model decode FLOPs 能否显著下降。

如果答案成立，再进入：

```text
AR
→
masked denoising
→
structured action
→
latent / flow expert
```

---

## 15. 最终研究命题

最简洁的版本：

> **Tool actions should be decoded from an LLM's latent contextual representation by a specialized action expert, rather than generated by the language model itself.**

更有冲击力的表达：

> **Does an AI agent need to generate actions with a language model at all?**

核心 architecture hypothesis：

[
\boxed{
\text{Semantic Representation}
\neq
\text{Language Generation}
\neq
\text{Action Generation}
}
]

最终希望得到类似：

```text
                     ┌─────────────→ Language Head
                     │               response
                     │
Context → LLM Backbone
                     │
                     │ KV
                     ▼
                Action Expert
                     │
                     ▼
                  Tool Call
```

即：

> **共享一个通用语义 backbone，但让 language 和 action 使用不同的生成机制。**

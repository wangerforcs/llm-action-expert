确认了一轮。结论需要稍微调整：

**“大 LLM + 专门的小模型生成 tool call”这个宽泛 idea 已经有人做；“冻结 LLM semantic backbone + 独立轻量 Action Expert 逐层读取 backbone KV、像 VLA action expert 一样直接产生 agent action”这个精确版本，我目前没有找到已经完整做过的工作。**

但周围已经有几篇非常近，尤其是 **HyFunc、SimpleTool、PrefillShare、ICaRus**。如果后面写论文，必须明确和它们切开。

| 工作                         | 已经做了什么                                                      | 和你的区别                                                 | 撞车风险               |
| -------------------------- | ----------------------------------------------------------- | ----------------------------------------------------- | ------------------ |
| **HyFunc**                 | 大 LLM hidden state → 小 LM → function call                   | 只传一个 soft token；小 LM 仍重新读文本 context，不读 layer-wise KV  | 🔴 高               |
| **SimpleTool**             | 同一 LLM，共享 prefix KV，并行生成 function/args                      | 没有独立 Action Expert，仍然是原 LLM 本身在 decode                | 🔴 高               |
| **PrefillShare**           | frozen prefill model → shared KV → task-specific decoder    | 通用 task decoder，不针对 action/tool policy                | 🔴 架构层面很近          |
| **ICaRus**                 | frozen logical encoder → KV → task-specific logical decoder | decoder 基本仍是原 base model + adapter，不是轻量 action expert | 🔴 架构层面很近          |
| **Probe&Prefill**          | hidden state 直接预测要不要 call tool                              | 只做 routing，不生成 action                                 | 🟢 支持你的 motivation |
| **OoO-Spec**               | 0.6B sidecar 并行预测 function + args                           | sidecar 不读取 target LLM latent/KV，且最终 target 仍验证       | 🟠 中高              |
| Tool-call dependency probe | 从 residual stream 解码 tool-call dependency                   | probing，不做 action generation                          | 🟢 支持 motivation   |

下面最关键。

---

## 1. 最大的“撞车点”其实是 HyFunc

HyFunc 是 2026 年的 KDD 工作，它已经明确做了：

```text
Large LM
   │
   │ hidden semantic representation
   ▼
soft token
   │
projection
   ▼
Small LM
   │
   ▼
function call
```

具体来说，大模型生成一个不 decode 的 **first soft token hidden state**；经过 linear projector 后，将它作为 continuous prompt 输入小模型。小模型经过专门 SFT，目标就是生成 ground-truth function call。([arXiv][1])

而且它实验里直接用了：

```text
LM_L = ToolACE-8B
LM_S = Qwen3-0.6B
```

所以如果我们的论文写：

> “We propose to decouple reasoning and tool calling by using a large LLM for semantic understanding and a small specialized model for function-call generation.”

**这个 claim 基本已经不能用了。**

HyFunc 已经做了。

但它跟我们的关键区别也很清楚。

HyFunc：

[
h_{\text{soft}}
===============

f_{\text{large}}(x)
]

然后：

[
a
=

g_{\text{small}}
(
x,\mathcal T,
P(h_{\text{soft}})
)
]

注意 **small LM 还是重新读取 original prompt 和 function descriptions**。那个 soft token只是额外 continuous prompt。([arXiv][1])

而我们真正想做的是：

[
{K_l,V_l}_{l=1}^{L}
===================

f_{\text{LLM}}(x)
]

然后：

[
a
=

g_{\text{ActionExpert}}
(
{K_l,V_l}
)
]

**Expert 不再重新 prefill context。**

这是重要区别。

---

# 2. SimpleTool 也非常近，但方向不同

SimpleTool 是 2026 年 3 月的工作，它直接从一个非常类似我们的 observation 出发：

> function call 和 free-form language generation 不一样，结构 token 很冗余，而且 arguments 的 causal dependency 较弱。([arXiv][2])

它甚至已经把 tool call：

```json
{
  "name": "get_weather",
  "arguments": {
    "city": "Beijing",
    "date": "tomorrow"
  }
}
```

改造成：

```text
<function> get_weather
<arg1> Beijing
<arg2> tomorrow
<arg3> <|null|>
...
```

然后：

```text
                 Shared prompt
                     │
                  Prefill
                     │
                shared KV
                     │
      ┌──────────────┼────────────┐
      ↓              ↓            ↓
 <function>        <arg1>       <arg2>
      ↓              ↓            ↓
 get_weather       Beijing      tomorrow
```

它的 function name 和各 argument stream **共享同一个 prefix KV cache**，然后并行 AR decode。([arXiv][2])

这已经明显在挑战：

[
\text{tool call}=\text{normal left-to-right language}
]

这个假设。

但是有个非常重要的区别：

> **SimpleTool 没有 Action Expert。**

论文明确强调，它不引入额外 parameters / draft model；这些 parallel streams 本质还是**同一个 Qwen model**，只是追加不同特殊 token 后并行 decode。([arXiv][2])

所以它是：

```text
                 Qwen
                  │
            shared prefix KV
             ↙     ↓     ↘
          Qwen    Qwen    Qwen
         stream  stream  stream
```

我们的则是：

```text
                 Qwen
                  │
             semantic KV
                  │
                  ▼
           Action Expert
             300M
                  │
                  ▼
               Action
```

还是明显不同。

不过 SimpleTool 应该成为**必须比较的 baseline**。

---

# 3. 真正让我觉得要重新定位 novelty 的，是 PrefillShare

这个工作跟我们之前讨论出来的数学形式已经非常接近。

PrefillShare 直接定义：

[
(\cdot,C_\text{base})
=====================

F_{\theta_\text{base}}(X)
]

其中 base model：

> **只进行 prefill，不参与之后 token generation。**

然后 specialized decoder：

[
y_t
===

F_{\theta_\text{dec}}
(
y_{t-1},
C_\text{base}
)
]

去消费 frozen base 生成的 KV cache。([arXiv][3])

甚至训练方式都是：

```text
Frozen base prefill
        │
        ▼
    Base KV cache
        │
        ▼
 Task-specific decoder
        │
        ▼
      tokens
```

loss：

[
\mathcal L
==========

-\sum_t
\log
P(
y_t|
y_{<t},
C_\text{base};
\theta_\text{dec}
)
]

只训练 decoder。([arXiv][3])

所以我们之前说的：

> “冻结 LLM，生成 KV，让另外一个 decoder condition 在这份 KV 上训练”

**作为一个通用 architecture primitive，已经不能算新。**

PrefillShare 已经明确做了，而且就是为 multi-model / agent workload 的 serving efficiency 提出的。([arXiv][3])

但它关注的是：

```text
Planner decoder
Coder decoder
Reviewer decoder
Math decoder
...
```

也就是**task-specialized language decoder**。

没有把它理解成：

[
\boxed{\text{language backbone}+\text{action policy}}
]

更没有专门针对 tool-call action distribution。

---

# 4. ICaRus 又进一步靠近

ICaRus 的表述甚至就是：

[
K,V=E_\text{base}(x)
]

然后：

[
x_{i+1}
=======

D_\text{task}
(
x_i,K,V
)
]

其中 (E_\text{base}) frozen，(D_\text{task}) 专门 fine-tune。([arXiv][4])

他们明确把 decoder-only Transformer conceptualize 成：

```text
logical encoder
     │
     │ KV
     ▼
logical decoder
```

而不同的：

```text
D_math
D_coding
D_reasoning
```

共享：

```text
E_base
```

和同一份 KV cache。([arXiv][4])

甚至 decoder 的 query 会 attend 到 base model 生成的相同 K/V。([arXiv][4])

所以：

> **“一个模型提供 K/V，另一个参数化 decoder 提供 Q 来读它”**

这个本身也已经不是新的 architecture observation。

不过 ICaRus 里的 decoder 基本仍然来自**同一个完整 pretrained LLM + LoRA/adapters**；论文目标是让 math/coding/reasoning 等多个 task-specific LLM 共享 KV，而不是训练一个小型 action policy。([arXiv][4])

---

# 5. 另外两个工作反而加强你的 motivation

今年五月的 **LLM Agents Already Know When to Call Tools** 很值得引用。

它发现：

> 在任何 output token 生成之前，只看 LLM prompt 最后一个 token 的 hidden states，就能非常准确地预测“是否应该调用 tool”。

他们在多个 Qwen/Llama 模型上用很简单的 linear probe，tool-necessity AUROC 达到约 0.89–0.96。([arXiv][5])

这说明：

[
\boxed{
\text{action information already exists in pre-generation representation}
}
]

而不一定必须：

[
hidden
\rightarrow
reasoning tokens
\rightarrow
tool-call tokens
]

论文甚至明确指出：representation-level knowledge 可以存在，但模型 generation 不一定能正确表达出来。([arXiv][5])

这几乎正好可以成为你的 motivation。

还有一篇五月的 probing 工作发现 tool-call dependency graph 也可以从 Qwen3 agent 的 residual streams 中线性解码出来，但作者明确限制自己的 claim 是 **representation，不是 behavioural control**。([arXiv][6])

所以你可以自然接一句：

> 那能不能直接把这种 representation 交给一个 action policy？

---

# 6. 本月刚出的 OoO-Spec 也必须注意

这个是 **2026 年 8 月 1 日**，非常新。

它用了一个：

```text
Qwen3-0.6B sidecar
```

专门根据 request + schemas 同时预测：

```text
function
arg1
arg2
arg3
...
```

而且是 parallel / out-of-order 地预测，再把结果送给大 target model 做 speculative verification。([arXiv][7])

所以：

> “用一个 0.6B 模型专门预测 tool action slots”

也已经有人做了。

但它刻意设计成 **target-independent**：

```text
request + schema
       ↓
0.6B sidecar
       ↓
semantic slots

与此同时：

request
  ↓
Large Target LLM
  ↓
verify sidecar proposal
```

sidecar **完全不读取大 LLM 的 hidden state / KV**，同一个 sidecar 可以跨 Qwen、Llama targets 使用。([arXiv][7])

这和你的目标恰好相反：

你就是想利用：

[
\boxed{\text{LLM latent semantic computation}}
]

而不是重新让 0.6B 从原始 prompt 自己理解一次。

---

# 所以我现在会怎么重新定义你的 idea？

我不会再叫它简单的：

> **Large LLM + Small Action Model**

这个 novelty 太弱，HyFunc 已经非常近。

也不能只是：

> **Shared KV + Specialized Decoder**

PrefillShare / ICaRus 已经覆盖掉了。

甚至不能只说：

> **Parallel structured tool generation**

SimpleTool / OoO-Spec 已经覆盖得很多。

我会把核心缩得非常精准：

## **VLA-style Agent Action Expert**

```text
                        ┌───────────────→ Language Head
                        │
                        │
Agent Context ─→ Frozen LLM Backbone
                        │
                        │ layer-wise KV
                        │
                        ▼
               Lightweight Action Expert
                        │
                        │ asymmetric
                        │ KV conditioning
                        ▼
                  Agent Action
                        │
             ┌──────────┴─────────┐
             ▼                    ▼
           Tool ID              Arguments
```

并且需要同时满足这几个性质：

1. **Backbone 是真正的通用 language/reasoning model。**
2. **Action Expert 是参数独立、显著小于 backbone 的 Transformer。**
3. **Expert 直接读取 backbone 的 layer-wise KV/latent states，而不是重新读取 textual prompt。**
4. **Backbone 不 attend 回 Action Expert，形成类似 VLA 的 asymmetric semantic→action interface。**
5. **Expert 专门学习 agent action distribution，而不是 general language generation。**
6. **Action Expert 最终可以使用与 language head 不同的 generation objective。**

截至 **2026 年 8 月 13 日**，我这一轮检索没有找到一个公开论文同时满足这六条。这个结论是“目前搜索未发现”，不是数学意义上证明不存在。

---

## 换句话说，novelty 已经从“模型拆开”变成了“action specialization”

最值得研究的不是：

[
\text{Can another decoder consume LLM KV?}
]

这个 PrefillShare / ICaRus 已经回答相当一部分了。([arXiv][3])

真正的问题应该变成：

[
\boxed{
\text{Can a VLA-style lightweight action expert
turn LLM latent states directly into agent actions?}
}
]

并进一步问：

[
\boxed{
\text{Is language decoding the wrong inductive bias for tool actions?}
}
]

这样 SimpleTool、HyFunc、PrefillShare 反而全部成为你的 prior work：

```text
HyFunc
↓
small specialized LM helps function calling

SimpleTool
↓
tool-call generation need not follow normal AR structure

Probe&Prefill
↓
action intent already exists in hidden representation

PrefillShare / ICaRus
↓
specialized decoder can consume frozen-base KV

             ↓↓↓

Ours
────────────────────────────
Treat tool use as a separate
ACTION MODALITY.

Frozen semantic LLM
        +
VLA-style Action Expert
        +
direct layer-wise latent/KV interface
────────────────────────────
```

**所以我不会因为这些工作放弃这个方向，但我会立刻修改项目定位。**

最危险的竞争工作是 **HyFunc + PrefillShare/ICaRus 的组合**：别人很容易说，“你不就是把 HyFunc 的 function-call small LM 换成 PrefillShare 式 KV-conditioned decoder 吗？”

因此要让论文真正站住，下一步最好不是马上写代码，而是先找到一个 **只有 VLA-style Action Expert 才明显成立的实验 hypothesis**。比如：**同等参数和训练数据下，直接 latent/KV conditioning 是否显著优于 HyFunc 式 soft-token bridge和 small-model textual reprefill；以及能否做到一次 backbone prefill 支撑连续多个 action expert steps。** 后面这个如果成立，差异就会大很多。

[1]: https://arxiv.org/html/2602.13665v1 "HyFunc: Accelerating LLM-based Function Calls for Agentic AI through Hybrid-Model Cascade and Dynamic Templating"
[2]: https://arxiv.org/pdf/2603.00030 "SimpleTool: Parallel Decoding for Real-Time LLM Function Calling"
[3]: https://arxiv.org/html/2602.12029v1 "PrefillShare: A Shared Prefill Module for KV Reuse in Multi-LLM Disaggregated Serving"
[4]: https://arxiv.org/html/2603.13281v1 "ICaRus: Identical Cache Reuse for Efficient Multi Model Inference"
[5]: https://arxiv.org/pdf/2605.09252 "LLM Agents Already Know When to Call Tools – Even Without Reasoning"
[6]: https://arxiv.org/html/2605.25310v1?utm_source=chatgpt.com "Tool-Call Dependency Structure is Linearly Decodable in ..."
[7]: https://arxiv.org/html/2608.00814v1 "OoO-Spec: Out-of-Order Semantic Speculation for Fast Tool Calling"

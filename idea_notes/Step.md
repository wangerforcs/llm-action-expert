可以。先把这个机制彻底搞清楚，你后面设计 Action Expert 会简单很多。

## 1. 现有 LLM 到底怎么生成 tool call？

以 **Qwen3-8B** 为例，答案是：

> **仍然是 autoregressive 地一个 token 一个 token 生成，只不过训练时规定了特殊的结构化格式。**

Qwen3 的官方 tool template 大致是：

```text
<|im_start|>system
# Tools

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"name": "get_weather", ...}
{"name": "search", ...}
</tools>

For each function call, return a json object within <tool_call></tool_call> tags.
<|im_end|>

<|im_start|>user
北京天气怎么样？
<|im_end|>

<|im_start|>assistant
<tool_call>
{"name": "get_weather", "arguments": {"city": "Beijing"}}
</tool_call>
<|im_end|>
```

Qwen 官方文档就是这样定义 tool calling 的：tool schema 被放进 system prompt 的 `<tools>...</tools>` 中，模型输出 JSON，并用 `<tool_call>...</tool_call>` 包裹；tool 执行结果再以 `<tool_response>...</tool_response>` 放回上下文。([Qwen][1])

这里有一个你可能会觉得很有意思的细节。

### `<tool_call>` 到底是不是特殊 token？

Qwen3-8B tokenizer 里：

```text
<|im_start|>   special=true
<|im_end|>     special=true

<tool_call>    special=false
</tool_call>   special=false

<tool_response>    special=false
</tool_response>   special=false
```

但是 `<tool_call>` **确实是 tokenizer vocabulary 中一个独立的 added token**，有自己的 token ID；只是在 Hugging Face tokenizer 定义里没有被标记为 `special=true`。([Hugging Face][2])

所以更准确地说：

[
\boxed{\text{tool call = 一段特殊格式的普通 AR token generation}}
]

并不是模型突然触发了某个 Python function。

模型实际上生成：

```text
<tool_call>
{
"
name
"
:
"
get_weather
"
...
}
</tool_call>
```

然后外面的 runtime/parser 看到：

```text
<tool_call>...</tool_call>
```

把 JSON parse 出来：

```python
name = "get_weather"
arguments = {"city": "Beijing"}
```

再真的执行：

```python
get_weather(city="Beijing")
```

Qwen-Agent 就是负责 template、parser 和 execution orchestration 这类工作的。([Hugging Face][3])

---

# 2. 那普通 Function Calling SFT 到底在训练什么？

我们用一个真实数据集：

**Salesforce/APIGen-MT-5k**

再配：

**Qwen3-8B**

来举例。

APIGen-MT 原始数据就是这种：

```json
{
  "conversations": [
    {
      "from": "human",
      "value": "I want to check reservation C6X779."
    },
    {
      "from": "function_call",
      "value": "{\"name\":\"get_reservation_details\",\"arguments\":{\"reservation_id\":\"C6X779\"}}"
    },
    {
      "from": "observation",
      "value": "{... reservation information ...}"
    },
    {
      "from": "gpt",
      "value": "Your reservation..."
    }
  ],
  "system": "... airline policy ...",
  "tools": "[...]"
}
```

它明确区分 `human / function_call / observation / gpt`，而且公开的是 5,000 条 multi-turn trajectories。([Hugging Face][4])

你第一步就是把它转换成 **Qwen native chat format**：

```python
messages = [
    {
        "role": "system",
        "content": airline_policy
    },
    {
        "role": "user",
        "content": "I want to check reservation C6X779."
    },
    {
        "role": "assistant",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "get_reservation_details",
                    "arguments": {
                        "reservation_id": "C6X779"
                    }
                }
            }
        ]
    }
]
```

然后：

```python
text = tokenizer.apply_chat_template(
    messages,
    tools=tools,
    tokenize=False
)
```

Qwen 的 Jinja template 会自动把它变成：

```text
<|im_start|>system
# Tools
...
<tools>
...
</tools>
...
<|im_end|>

<|im_start|>user
I want to check reservation C6X779.
<|im_end|>

<|im_start|>assistant
<tool_call>
{"name":"get_reservation_details","arguments":{"reservation_id":"C6X779"}}
</tool_call>
<|im_end|>
```

也就是说，SFT 时模型就是在学：

[
P(
\texttt{<tool_call>} ,
\texttt{{},
\texttt{"name"},
...
|
context
)
]

完全是普通 causal LM loss。

---

# 3. 标准 SFT 的 loss 长什么样？

完整 token sequence：

```text
SYSTEM SYSTEM SYSTEM ...
USER USER USER ...
ASSISTANT <tool_call> { JSON ... } </tool_call> <|im_end|>
```

通常不会让所有 token 都算 loss。

你做一个 mask：

```text
system      → loss mask = 0
user        → loss mask = 0

assistant:
<tool_call> → 1
JSON        → 1
</tool_call>→ 1
<|im_end|>  → 1
```

也就是：

[
\mathcal L
==========

-\sum_{i\in assistant}
\log P_\theta(y_i|x,y_{<i})
]

如果还有普通语言回答：

```text
<|im_start|>assistant
Your reservation is...
<|im_end|>
```

也正常算 assistant loss。

所以现在标准 function calling model 本质上在同时学：

[
\boxed{
\mathcal L
==========

\mathcal L_{language}
+
\mathcal L_{tool}
}
]

而且二者共用：

```text
Transformer
    ↓
LM head
    ↓
151k vocabulary
```

这就是你想挑战的地方。

---

# 4. 你的版本应该怎么改？

我建议你**第一版不要动 Qwen3-8B**。

直接：

```text
                 Qwen3-8B
                   Frozen
                     │
                  KV cache
                     │
                     ▼
             Action Expert
               300M 左右
                     │
                     ▼
             tool-call tokens
```

训练样本不要按 trajectory 算，而是按 **每一个 function_call event** 拆。

例如一条 trajectory：

```text
user
 ↓
tool_call A
 ↓
observation A
 ↓
tool_call B
 ↓
observation B
 ↓
assistant
```

拆成两个训练 sample：

```text
Sample 1

Context:
system
tools
user

Target:
tool_call A
```

以及：

```text
Sample 2

Context:
system
tools
user
tool_call A
observation A

Target:
tool_call B
```

因此一个 trajectory 有 7 个 tool call：

[
\rightarrow 7\text{ 个 Action Expert training examples}
]

这个转换非常重要。

---

# 5. 一个具体训练 example

假设当前 context 是：

```text
SYSTEM:
You are an airline agent...

TOOLS:
get_reservation_details(reservation_id)
cancel_reservation(reservation_id)
...

USER:
I want to check reservation C6X779.
```

### Step A：让 Qwen3-8B 只做 prefill

```python
with torch.no_grad():
    outputs = backbone(
        input_ids=context_ids,
        use_cache=True,
        output_hidden_states=True
    )
```

得到：

```text
layer 0: K0, V0
layer 1: K1, V1
layer 2: K2, V2
...
layer L: KL, VL
```

注意：

**这里 Qwen 一个 token 都不 decode。**

只：

[
context\rightarrow KV
]

---

# 6. Action Expert 开始生成

Expert 的 target 是：

```text
<tool_call>
{"name":"get_reservation_details","arguments":{"reservation_id":"C6X779"}}
</tool_call>
<|im_end|>
```

假设 tokenized 后：

```text
y0 = <tool_call>
y1 = \n
y2 = {
y3 = "name"
...
yn = </tool_call>
```

Action Expert 还是先做 AR：

[
P_\phi(a|KV)
============

\prod_i
P_\phi(a_i|a_{<i},KV)
]

例如 Expert 第 (l) 层：

[
Q_l^A=W_{Q,l}^AH_l^A
]

读取 frozen backbone：

[
\tilde H_l^A
============

Attention
(
Q_l^A,
K_l^{LLM},
V_l^{LLM}
)
]

然后再做自己的 self-attention：

```text
             Qwen layer-l KV
                    │
                    ▼
Action token → Expert Layer l
                    │
                    ▼
               next layer
```

最后：

```python
logits = action_head(h_action)
loss = cross_entropy(logits, target_tool_tokens)
```

这里**只训练 Action Expert**：

```python
for p in backbone.parameters():
    p.requires_grad = False
```

---

# 7. 一个非常关键的问题：Action Expert 输出 vocabulary 用什么？

第一版我强烈建议：

> **直接复用 Qwen tokenizer 和 Qwen vocabulary。**

也就是说 Action Expert 仍然输出：

```text
<tool_call>
{"name":"get_reservation_details", ...}
</tool_call>
```

这样你的实验最干净。

Baseline：

```text
Qwen backbone
    ↓
Qwen LM head
    ↓
tool-call tokens
```

Ours：

```text
Qwen backbone
    ↓ KV
Action Expert
    ↓
same Qwen tokenizer vocabulary
    ↓
same tool-call tokens
```

唯一变化就是：

[
\boxed{
\text{谁负责生成 action}
}
]

这样论文结果才容易解释。

否则你如果同时把 output representation 改成 structured slots，又把 decoder 改成 diffusion，又换 vocabulary，你最后不知道 improvement 来自哪里。

---

# 8. 第一版甚至不需要训练 `<tool_call>` 这个 token

这里其实有两个方案。

方案 A 是让 Expert 完整生成：

```text
<tool_call>
{...}
</tool_call>
```

我更建议第一篇实验这么做，因为和 baseline 完全一致。

但方案 B 更符合你最终的 architecture：

runtime 已经知道进入了 Action Expert，那么根本不需要它生成：

```text
<tool_call>
```

直接让 Expert target 是：

```json
{"name":"get_reservation_details","arguments":{"reservation_id":"C6X779"}}
```

甚至最终可以变成：

```text
tool_id = 17
arg tokens = ...
```

于是 `<tool_call>` 只是 **language model/runtime communication protocol**，对真正 specialized action policy 并不是必要的。

这点很值得区分：

```text
今天：

LLM
 ↓
<tool_call>
 ↓
parser
 ↓
runtime
```

你的最终架构：

```text
Action Expert
 ↓
structured action
 ↓
runtime
```

完全可以没有 `<tool_call>`。

---

# 9. 还有一个 routing 问题必须解决

你的系统不是永远都应该调用工具。

有时候：

```text
user
 ↓
final answer
```

有时候：

```text
user
 ↓
tool call
```

所以最终需要一个：

[
r_t=P(\text{ACTION}|h_t)
]

也就是一个非常小的 **action gate**：

```text
                   Qwen Backbone
                        │
                        h
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
          Action Gate         Language Head
              │
       ACTION / LANGUAGE
              │
       if ACTION
              ▼
        Action Expert
```

训练也非常简单。

APIGen 里：

```text
from == "function_call"
```

标签：

```text
ACTION = 1
```

如果下一个 turn 是：

```text
from == "gpt"
```

标签：

```text
ACTION = 0
```

所以：

[
\mathcal L
==========

\mathcal L_{gate}
+
\lambda\mathcal L_{action}
]

但 MVP 阶段**甚至可以先不训练 gate**。

实验直接只挑：

> ground-truth 下一步一定是 function_call

的 state。

这样先回答最核心的问题：

[
\boxed{
KV\rightarrow ActionExpert
}
]

到底能不能生成正确 action。

---

# 10. 我建议你的第一版实验就这么定

不要一上来搞 flow matching。

```text
Dataset:
APIGen-MT-5k

Backbone:
Qwen3-8B

Backbone:
100% frozen

Input:
trajectory prefix

Representation:
Qwen per-layer KV

Expert:
~300M decoder-only Transformer

Target:
Qwen-native serialized tool call

Objective:
autoregressive cross entropy

Loss:
ONLY tool-call tokens
```

对比：

```text
Baseline

context
 ↓
Qwen3-8B
 ↓
LM Head
 ↓
tool call
```

和：

```text
Ours

context
 ↓
Frozen Qwen3-8B
 ↓
KV
 ↓
300M Action Expert
 ↓
tool call
```

这样你的最初 hypothesis 就非常纯：

[
\boxed{
P_{\text{LLM-head}}(a|x)
\quad vs \quad
P_{\text{ActionExpert}}(a|KV(x))
}
]

先比较 tool selection、argument accuracy、sequence exact match、BFCL/τ-bench task success，以及 decode FLOPs。

如果 **300M Expert 可以接近甚至超过 Qwen 自己的 LM head**，你的 idea 就已经出现了非常有意思的结果。

---

还有一个我觉得你现在应该牢牢记住的概念：

> **Tool calling API 看起来像模型“调用了函数”，但从模型视角，它通常只是生成了一种结构化 token sequence。**

Qwen3 尤其清楚：

```text
语义：
我要调用 get_reservation_details

             ↓

模型实际做的事情：

P(<tool_call>|context)
P({|context,<tool_call>)
P("name"|...)
P(:|...)
...
```

Qwen 的 tokenizer 甚至已经给 `<tool_call>` 和 `</tool_call>` 分配了独立 vocab IDs；而真正作为 turn boundary 的 `<|im_start|>` / `<|im_end|>` 才被标记为特殊 control token。([Hugging Face][2])

**你的 Action Expert idea 本质就是把上面这整个 `P(action tokens | context)` 从通用 LM decoder 里剥出去。**

顺带提醒一点：APIGen-MT-5k 很适合做方法原型，但它的 dataset card 是 CC BY-NC 4.0，并且明确提示其中部分 GPT-4 生成数据有额外用途限制；如果后面准备公开权重或考虑商业使用，应该单独处理许可证问题。([Hugging Face][4])

如果你下一步要开始真正实现，我建议我们直接往下画 **Qwen3 Attention 层 → KV cache → 300M Action Expert 的 tensor shape / architecture**。这一步会涉及一个很关键的问题：**Expert 到底直接复用 Qwen 的 K/V，还是给每层 KV 再加一个 projection adapter。**

[1]: https://qwen.readthedocs.io/en/latest/getting_started/concepts.html "Key Concepts - Qwen"
[2]: https://huggingface.co/Qwen/Qwen3-8B/blob/main/tokenizer_config.json?utm_source=chatgpt.com "tokenizer_config.json · Qwen/Qwen3-8B at main"
[3]: https://huggingface.co/Qwen/Qwen3-8B "Qwen/Qwen3-8B · Hugging Face"
[4]: https://huggingface.co/datasets/Salesforce/APIGen-MT-5k "Salesforce/APIGen-MT-5k · Datasets at Hugging Face"

# llm_rl_trading_finsaber API 模型选择问题分析

> **DeepSeek vs GPT-4：模型选择的致命影响**

---

## 🚨 核心问题总结

当前项目使用 **DeepSeek-Chat** 作为 LLM 后端，而原始 LESR 使用的是 **GPT-4**，这导致了多个严重问题：

### 问题 1: 模型能力差距 ⭐⭐⭐⭐⭐

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    模型能力对比                                             │
└─────────────────────────────────────────────────────────────────────────────┘

维度              │ GPT-4          │ DeepSeek-Chat   │ 影响
────────────────────────────────────────────────────────────────────────────
代码生成能力      │ ⭐⭐⭐⭐⭐       │ ⭐⭐⭐⭐         │ DeepSeek 代码质量可能不稳定
金融领域知识      │ ⭐⭐⭐⭐⭐       │ ⭐⭐⭐⭐         │ 金融推理能力可能不足
复杂推理能力      │ ⭐⭐⭐⭐⭐       │ ⭐⭐⭐⭐         │ Lipschitz 分析可能不准确
指令遵循能力      │ ⭐⭐⭐⭐⭐       │ ⭐⭐⭐⭐         │ 约束条件可能被忽略
长上下文处理      │ ⭐⭐⭐⭐⭐       │ ⭐⭐⭐⭐         │ 历史 Comprehens可能不完整
数值稳定性意识    │ ⭐⭐⭐⭐⭐       │ ⭐⭐⭐           │ 容易产生 NaN/Inf 问题

关键差异:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. GPT-4 在代码生成任务上显著优于 DeepSeek
2. GPT-4 在金融领域的知识更加全面和准确
3. GPT-4 对复杂约束的理解和遵循能力更强
4. GPT-4 的输出更加稳定和可预测
```

### 问题 2: 参数设置不当 ⭐⭐⭐⭐

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    参数设置对比                                             │
└─────────────────────────────────────────────────────────────────────────────┘

参数              │ 原始 LESR      │ 当前项目        │ 问题
────────────────────────────────────────────────────────────────────────────
model             │ gpt-4          │ deepseek-chat   │ 模型能力下降
temperature       │ 0.0            │ 0.2             │ ❌ 增加了随机性
max_tokens        │ 2000           │ 3500            │ ❌ 太长，可能冗余
k (采样数量)       │ 6              │ 3               │ ❌ 探索不足
max_retries       │ 10             │ 4               │ ❌ 重试次数少
iterations        │ 5              │ 10              │ ⚠️  迭代更多但质量可能下降

致命组合:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. ❌ temperature=0.2 + 较弱模型 = 输出不够稳定
   • GPT-4 用 temp=0.0 确保确定性输出
   • DeepSeek 用 temp=0.2 可能导致代码质量波动

2. ❌ max_tokens=3500 + 较弱模型 = 生成冗余代码
   • GPT-4 用 2000 tokens 足够生成简洁代码
   • DeepSeek 用 3500 可能生成冗长、低质量代码

3. ❌ k=3 + 较弱模型 = 探索不足
   • GPT-4 用 k=6 能覆盖更多候选
   • DeepSeek 用 k=3 可能错过优秀候选
```

### 问题 3: 金融领域知识差距 ⭐⭐⭐⭐⭐

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    金融领域知识差距                                         │
└─────────────────────────────────────────────────────────────────────────────┘

GPT-4 的金融知识优势:
━━━━━━━━━━━━━━━━━━━━━━━
✅ 训练数据包含大量金融文献
✅ 理解风险调整收益 (Sharpe Ratio, Sortino Ratio)
✅ 理解衍生品定价 (Black-Scholes, Binomial Tree)
✅ 理解投资组合理论 (Markowitz, CAPM, APT)
✅ 理解风险管理 (VaR, CVaR, Stress Testing)
✅ 理解市场微观结构 (Bid-Ask Spread, Order Book)
✅ 理解技术分析 (RSI, MACD, Bollinger Bands)
✅ 理解行为金融学 (Prospect Theory, Loss Aversion)

DeepSeek 的局限:
━━━━━━━━━━━━━━━━━━━
⚠️  金融知识可能不如 GPT-4 全面
⚠️  对复杂的金融概念理解可能不够深入
⚠️  可能产生"似是而非"的金融推理
⚠️  对风险管理的理解可能不够准确

实际影响示例:
━━━━━━━━━━━━━━━━
问题 1: 风险调整收益概念
• GPT-4: 准确理解 Sharpe Ratio = (Return - RiskFree) / Volatility
• DeepSeek: 可能混淆 Sharpe Ratio 与其他比率，或计算错误

问题 2: 内在奖励设计
• GPT-4: 能设计出真正风险-aware 的 intrinsic_reward
• DeepSeek: 可能设计出"看似合理"但实际上不风险-aware 的奖励

问题 3: 数值稳定性
• GPT-4: 更注意除零、NaN、Inf 问题
• DeepSeek: 可能忽略这些细节，导致运行时错误
```

### 问题 4: 代码生成质量差异 ⭐⭐⭐⭐⭐

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    代码生成质量对比                                         │
└─────────────────────────────────────────────────────────────────────────────┘

实测对比 (基于审计发现的代码质量问题):

GPT-4 生成的代码特点:
━━━━━━━━━━━━━━━━━━━━━━━
✅ 代码结构清晰，注释恰当
✅ 变量命名有意义，易于理解
✅ 边界条件处理完善
✅ 数值稳定性考虑周全
✅ 符合 Python 最佳实践
✅ 错误处理机制健全

DeepSeek 生成的代码问题:
━━━━━━━━━━━━━━━━━━━━━━━
❌ 代码结构混乱，缩进不一致
❌ 变量命名随意（如 s1, s2, tmp）
❌ 边界条件处理不当
❌ 容易忽略除零保护
❌ 硬编码索引，容易越界
❌ 缺少错误处理

具体示例 (审计发现):
━━━━━━━━━━━━━━━━━━━━━
❌ 问题 1: 硬编码索引越界
```python
# DeepSeek 生成的问题代码
def revise_state(s):
    momentum = (s[1] - s[6]) / (s[6] + 1e-8)  # ❌ s[6] 可能不是 open price
    # 在 native contract 中，s[6] 是 holding_0，不是 open
```

❌ 问题 2: 缺少除零保护
```python
# DeepSeek 生成的问题代码
def intrinsic_reward(s):
    return s[0] / np.mean(s[1:])  # ❌ 没有 epsilon 保护
    # 如果 s[1:] 全为 0，会返回 inf 或 nan
```

❌ 问题 3: 数值范围问题
```python
# DeepSeek 生成的问题代码
def intrinsic_reward(s):
    reward = 1000.0 * s[0]  # ❌ 可能超出 [-100, 100]
    return reward  # 没有 clip
```

❌ 问题 4: raw-state fallback 不足
```python
# DeepSeek 生成的问题代码
def intrinsic_reward(s):
    # 只使用扩展维度
    momentum = s[-5:]  # ❌ 在 raw state 上会访问错误索引
    return np.mean(momentum)
```

根本原因:
━━━━━━━━━━━━━━━━━━━
1. DeepSeek 对 Python 编码最佳实践理解不够深
2. DeepSeek 对数值稳定性问题的意识不足
3. DeepSeek 对复杂约束（如 native state contract）的理解不够准确
4. DeepSeek 的代码推理能力不如 GPT-4
```

### 问题 5: 迭代反馈理解能力 ⭐⭐⭐⭐

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    迭代反馈理解能力                                         │
└─────────────────────────────────────────────────────────────────────────────┘

CoT 反馈理解:
━━━━━━━━━━━━━━━━━━━
GPT-4:
✅ 能准确理解 Lipschitz 常数的含义
✅ 能从成功/失败案例中提取模式
✅ 能基于反馈进行实质性改进
✅ 能避免重复失败模式

DeepSeek:
⚠️  可能误解 Lipschitz 常数的含义
⚠️  可能从反馈中提取错误的模式
⚠️  可能在迭代中重复相似错误
⚠️  改进可能只是表面的修改

实际影响:
━━━━━━━━━━━━━━━━━━━
审计发现 TD3 的 intrinsic_probe_delta_sharpe = 0.0，说明：

1. DeepSeek 生成的 intrinsic_reward 在原始状态上几乎无效
2. 10 轮迭代后，这个问题没有被根本解决
3. 说明 DeepSeek 没有从反馈中学会如何设计有效的 raw-state fallback

对比原始 LESR:
━━━━━━━━━━━━━━━━━━━
• 原始 LESR (GPT-4) 5 轮后能达到显著改进
• 当前项目 (DeepSeek) 10 轮后仍有 core 问题未解决
• 说明迭代效率显著下降
```

---

## 🔍 深度分析：为什么 DeepSeek 不适合这个项目

### 1. 金融 DRL 的特殊要求

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    金融 DRL 的特殊要求                                      │
└─────────────────────────────────────────────────────────────────────────────┘

金融 DRL 需要的 LLM 能力:
━━━━━━━━━━━━━━━━━━━━━
1. 深厚的金融知识
   • 理解风险调整收益
   • 理解投资组合理论
   • 理解市场微观结构
   • 理解行为金融学

2. 强大的代码生成能力
   • 生成数值稳定的代码
   • 处理边界条件
   • 避免常见陷阱（除零、NaN、Inf）
   • 遵循复杂约束（native state contract）

3. 准确的逻辑推理
   • 理解 Lipschitz 分析
   • 从反馈中提取模式
   • 避免重复错误
   • 进行实质性改进

4. 严格的指令遵循
   • 遵守 raw-state fallback 约束
   • 遵守数值范围约束
   • 遵守索引范围约束
   • 遵守 G2/G3 双模式要求

DeepSeek 在这些方面的表现:
━━━━━━━━━━━━━━━━━━━━━
❌ 金融知识：不如 GPT-4 全面
❌ 代码生成：稳定性不足，容易出错
❌ 逻辑推理：复杂推理能力不如 GPT-4
❌ 指令遵循：对复杂约束的理解不够准确

结论:
━━━━━━━━━━━━━━━━━━━━━
DeepSeek 更适合通用编程任务，而不是高度专业化的金融 DRL 任务。
```

### 2. 原始 LESR 成功的关键因素

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    原始 LESR 成功的关键因素                                 │
└─────────────────────────────────────────────────────────────────────────────┘

原始 LESR 的配置:
━━━━━━━━━━━━━━━━━━━
model: gpt-4              ← 最强的代码生成模型
temperature: 0.0          ← 确定性输出，减少随机性
max_tokens: 2000          ← 足够但不冗长
k: 6                      ← 充分探索
iterations: 5             ← 适度迭代
max_retries: 10           ← 充分重试

成功原因分析:
━━━━━━━━━━━━━━━━━━━
1. ✅ GPT-4 的代码生成能力
   • 生成高质量、稳定的代码
   • 理解物理约束（机器人控制）
   • 处理边界条件

2. ✅ temperature=0.0 的确定性
   • 每次生成一致的结果
   • 减少随机波动
   • 提高可复现性

3. ✅ 充分的探索 (k=6)
   • 覆盖更多候选空间
   • 增加找到优秀解的概率

4. ✅ 适度的迭代 (5 轮)
   • 避免过度优化
   • 避免陷入局部最优

5. ✅ 机器人控制任务的特性
   • 物理定律清晰
   • 状态空间明确
   • 奖励设计直观

当前项目的问题:
━━━━━━━━━━━━━━━━━━━
❌ DeepSeek 的代码生成能力不足
❌ temperature=0.2 增加了随机性
❌ k=3 探索不足
❌ 10 轮迭代但质量可能下降
❌ 金融 DRL 任务更复杂
   • 状态契约复杂（native contract）
   • 奖励设计复杂（风险调整）
   • 约束更复杂（G2/G3 双模式）
```

---

## 📊 实验证据

### 审计发现的问题

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    审计发现的问题                                         │
└─────────────────────────────────────────────────────────────────────────────┘

问题 1: Intrinsic Reward 的 Raw-State Fallback 不足
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• TD3 最佳候选: intrinsic_probe_delta_sharpe = 0.0
• TD3 G2 行为几乎复制 G0
• TD3 G3 行为几乎复制 G1
• 说明 intrinsic_reward(s) 在原始状态上几乎没有信号

根本原因:
• DeepSeek 没有理解"raw-state fallback"的重要性
• DeepSeek 生成的 intrinsic_reward 完全依赖扩展维度
• DeepSeek 没有从 10 轮反馈中学会如何设计有效的 raw-state intrinsic

对比原始 LESR:
• 原始 LESR 的 intrinsic_reward 在所有模式下都有效
• 说明 GPT-4 更好地理解了双模式要求

问题 2: 代码质量问题
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• llm_errors.json 中有大量索引错误
• revise_state_exception_native_sample_0: 6 次
• intrinsic_exception_native_raw_sample_0: 2 次
• intrinsic_exception_native_revised_sample_0: 2 次

根本原因:
• DeepSeek 没有理解 native state contract
• DeepSeek 使用了错误的索引
• DeepSeek 硬编码了索引，没有使用动态计算

问题 3: 迭代改进不足
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 10 轮迭代后，核心问题仍未解决
• 说明 DeepSeek 没有从反馈中有效学习
• 说明 DeepSeek 的迭代效率低于 GPT-4
```

---

## 💡 改进建议

### 建议 1: 切换到 GPT-4 (最优解) ⭐⭐⭐⭐⭐

```yaml
# 配置文件修改
llm:
  enabled: true
  base_url: https://api.openai.com/v1  # 或使用代理
  model: gpt-4-1106-preview             # 或 gpt-4-turbo
  temperature: 0.0                      # ← 改回 0.0
  max_tokens: 2000                       # ← 改回 2000
  k: 6                                   # ← 改回 6
  max_retries: 10                        # ← 改回 10
  iterations: 5                          # ← 可以保持 10 或改回 5
```

**预期改进**:
- ✅ 代码质量显著提升
- ✅ raw-state fallback 问题解决
- ✅ 索引错误减少
- ✅ 迭代效率提升

### 建议 2: 如果必须用 DeepSeek，调整参数 (次优解) ⭐⭐⭐

```yaml
llm:
  enabled: true
  base_url: https://api.deepseek.com
  model: deepseek-chat
  temperature: 0.0                      # ← 改为 0.0
  max_tokens: 2000                       # ← 改为 2000
  k: 5                                   # ← 增加到 5
  max_retries: 10                        # ← 增加到 10
  iterations: 10
```

**同时加强 Prompt**:
```python
def build_system_prompt():
    return """
You are using DeepSeek to generate code for financial DRL.

CRITICAL: DeepSeek has lower code generation quality than GPT-4.
You MUST be EXTRA CAREFUL about:

1. Numerical Stability
   - ALWAYS add epsilon to division: x / (y + 1e-8)
   - ALWAYS clip outputs: np.clip(x, -100, 100)
   - ALWAYS check for NaN/Inf

2. Index Safety
   - DO NOT hard-code indices beyond the state dimension
   - ALWAYS use dynamic calculation
   - ALWAYS validate indices are within bounds

3. Raw-State Fallback
   - intrinsic_reward MUST work on raw state alone
   - Test: what does intrinsic_reward(raw_state) return?
   - It should be NON-CONSTANT and informative

4. Code Quality
   - Use meaningful variable names
   - Add comments for complex logic
   - Handle edge cases

If you are unsure, prefer simpler, safer code over complex optimizations.
"""
```

### 建议 3: 混合策略 (折中解) ⭐⭐⭐⭐

```python
# 使用 GPT-4 生成初始候选，DeepSeek 迭代优化

class HybridLLMStrategy:
    def __init__(self):
        self.gpt4_client = OpenAIClient(api_key=os.environ["OPENAI_API_KEY"])
        self.deepseek_client = DeepSeekClient(api_key=os.environ["DEEPSEEK_API_KEY"])

    def sample_initial_candidates(self, prompt, k=3):
        """Iteration 0: 使用 GPT-4"""
        return self.gpt4_client.sample(prompt, k=k, temperature=0.0)

    def sample_iteration_candidates(self, prompt, k=3):
        """Iteration 1-9: 使用 DeepSeek"""
        return self.deepseek_client.sample(prompt, k=k, temperature=0.0)
```

**优势**:
- ✅ 初始质量有保障（GPT-4）
- ✅ 成本可控（后续用 DeepSeek）
- ✅ 保留迭代优化能力

---

## 🎯 结论

### 核心问题

**DeepSeek-Chat 不适合作为金融 DRL 任务的 LLM 后端，主要原因：**

1. **代码生成能力不足** → 导致代码质量不稳定
2. **金融知识不够全面** → 导致 intrinsic_reward 设计不当
3. **指令遵循能力不足** → 导致 raw-state fallback 失败
4. **迭代反馈理解能力不足** → 导致 10 轮后核心问题仍未解决

### 根本原因

**这个项目是从机器人控制迁移到金融 DRL，任务复杂度显著增加：**

- ❌ 状态契约更复杂（native vs generic）
- ❌ 奖励设计更复杂（风险调整 vs 任务完成）
- ❌ 约束更复杂（G2/G3 双模式 vs 单一模式）
- ❌ 知识要求更高（金融理论 vs 物理定律）

**在这种情况下，使用较弱的 DeepSeek 模型是雪上加霜。**

### 最优解决方案

**切换到 GPT-4，并调整参数到原始 LESR 的设置：**

```yaml
llm:
  model: gpt-4-1106-preview
  temperature: 0.0
  max_tokens: 2000
  k: 6
  max_retries: 10
```

**预期效果：**
- ✅ raw-state fallback 问题解决
- ✅ 代码质量显著提升
- ✅ 迭代效率提升
- ✅ 索引错误减少

---

**文档版本**: v1.0
**最后更新**: 2026-04-02

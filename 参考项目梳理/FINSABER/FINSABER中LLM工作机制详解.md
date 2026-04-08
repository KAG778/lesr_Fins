# FINSABER 中 LLM 工作机制详解

## 目录

1. [概述](#概述)
2. [LLM 策略架构](#llm-策略架构)
3. [FINMEM 工作机制](#finmem-工作机制)
4. [FINAGENT 工作机制](#finagent-工作机制)
5. [Prompt 工程详解](#prompt-工程详解)
6. [完整决策流程](#完整决策流程)
7. [案例分析](#案例分析)
8. [代码示例](#代码示例)

---

## 概述

FINSABER 框架集成了两种主要的 LLM 驱动交易策略：

### 策略对比

| 特性 | FINMEM | FINAGENT |
|------|---------|----------|
| **核心思想** | 分层记忆 + 角色设计 | 强化学习微调 + 工具使用 |
| **记忆系统** | 三层记忆（短期/中期/长期/反思） | 向量记忆库 + 检索增强 |
| **决策方式** | 基于 LLM 推理 | LLM + RL 微调 |
| **复杂度** | 中等 | 高 |
| **计算成本** | 较低 | 较高 |
| **适用场景** | 单股票交易 | 复杂多因子交易 |

---

## LLM 策略架构

### 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                     LLM 交易策略整体架构                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              数据输入层 (Data Input)                     │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │   │
│  │  │ 价格数据 │  │ 新闻数据 │  │ 财务数据 │              │   │
│  │  │  (Price) │  │  (News)  │  │(Filing)  │              │   │
│  │  └──────────┘  └──────────┘  └──────────┘              │   │
│  └──────────────────────────────────────────────────────────┘   │
│                          ↓                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │            特征提取层 (Feature Extraction)               │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │   │
│  │  │ 技术指标 │  │ 文本特征 │  │ 市场状态 │              │   │
│  │  │  (TA)    │  │ (NLP)    │  │ (State)  │              │   │
│  │  └──────────┘  └──────────┘  └──────────┘              │   │
│  └──────────────────────────────────────────────────────────┘   │
│                          ↓                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              记忆/知识库层 (Memory/Knowledge)            │   │
│  │  ┌──────────────────────────────────────────────────┐   │   │
│  │  │  FINMEM: 分层记忆                                │   │   │
│  │  │  ├─ 感知记忆 (Perception)                        │   │   │
│  │  │  ├─ 短期记忆 (Short-term)                        │   │   │
│  │  │  ├─ 长期记忆 (Long-term)                         │   │   │
│  │  │  └─ 反思记忆 (Reflection)                        │   │   │
│  │  └──────────────────────────────────────────────────┘   │   │
│  │  ┌──────────────────────────────────────────────────┐   │   │
│  │  │  FINAGENT: 向量记忆库                            │   │   │
│  │  │  ├─ 嵌入存储 (Embedding Store)                   │   │   │
│  │  │  ├─ 语义检索 (Semantic Search)                   │   │   │
│  │  │  └─ 多样化查询 (Diverse Query)                  │   │   │
│  │  └──────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                          ↓                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Prompt 构建层 (Prompt Engineering)          │   │
│  │  ┌──────────────────────────────────────────────────┐   │   │
│  │  │  系统提示 (System Prompt)                        │   │   │
│  │  │  - 角色定义                                       │   │   │
│  │  │  - 任务说明                                       │   │   │
│  │  │  - 输出格式                                       │   │   │
│  │  └──────────────────────────────────────────────────┘   │   │
│  │  ┌──────────────────────────────────────────────────┐   │   │
│  │  │  上下文组装 (Context Assembly)                   │   │   │
│  │  │  - 市场信息                                       │   │   │
│  │  │  - 历史记忆                                       │   │   │
│  │  │  - 当前状态                                       │   │   │
│  │  │  - 决策要求                                       │   │   │
│  │  └──────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                          ↓                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              LLM 推理层 (LLM Inference)                  │   │
│  │  ┌──────────────────────────────────────────────────┐   │   │
│  │  │  模型调用 (Model Call)                           │   │   │
│  │  │  - GPT-4 / GPT-3.5                               │   │   │
│  │  │  - Claude                                        │   │   │
│  │  │  - LLaMA (本地部署)                              │   │   │
│  │  └──────────────────────────────────────────────────┘   │   │
│  │  ┌──────────────────────────────────────────────────┐   │   │
│  │  │  输出解析 (Output Parsing)                       │   │   │
│  │  │  - 结构化提取                                     │   │   │
│  │  │  - 格式验证                                       │   │   │
│  │  │  - 错误处理                                       │   │   │
│  │  └──────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                          ↓                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              决策执行层 (Decision Execution)             │   │
│  │  ┌──────────────────────────────────────────────────┐   │   │
│  │  │  动作映射 (Action Mapping)                       │   │   │
│  │  │  - BUY → 买入操作                                │   │   │
│  │  │  - SELL → 卖出操作                               │   │   │
│  │  │  - HOLD → 持有                                   │   │   │
│  │  └──────────────────────────────────────────────────┘   │   │
│  │  ┌──────────────────────────────────────────────────┐   │   │
│  │  │  风险检查 (Risk Check)                           │   │   │
│  │  │  - 资金充足性                                     │   │   │
│  │  │  - 持仓限制                                       │   │   │
│  │  │  - 交易成本                                       │   │   │
│  │  └──────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                          ↓                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              记忆更新层 (Memory Update)                  │   │
│  │  ┌──────────────────────────────────────────────────┐   │   │
│  │  │  存储决策记录                                     │   │   │
│  │  │  存储市场状态                                     │   │   │
│  │  │  存储执行结果                                     │   │   │
│  │  │  更新重要性分数                                   │   │   │
│  │  └──────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## FINMEM 工作机制

### 核心概念

FINMEM（Financial Memory）是一个基于分层记忆和角色设计的 LLM 交易代理。

### 架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        FINMEM 架构                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              1. 角色设计 (Profiling)                      │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │  投资者画像                                        │  │   │
│  │  │  - 风险偏好: Risk-seeking / Risk-averse           │  │   │
│  │  │  - 投资风格: Momentum / Mean-reversion            │  │   │
│  │  │  - 决策风格: Aggressive / Conservative             │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                          ↓                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              2. 记忆系统 (Memory System)                 │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │  分层记忆架构                                      │  │   │
│  │  │                                                     │  │   │
│  │  │  ┌──────────────────────────────────────────────┐ │  │   │
│  │  │  │ 感知记忆 (Perception Memory)                 │ │  │   │
│  │  │  │ - 存储期限: 1-3 天                            │ │  │   │
│  │  │  │ - 内容: 原始市场数据、新闻标题                │ │  │   │
│  │  │  │ - 容量: 无限                                  │ │  │   │
│  │  │  │ - 作用: 提供最直接的市场感知                  │ │  │   │
│  │  │  └──────────────────────────────────────────────┘ │  │   │
│  │  │                     ↓                              │  │   │
│  │  │  ┌──────────────────────────────────────────────┐ │  │   │
│  │  │  │ 短期记忆 (Short-term Memory)                 │ │  │   │
│  │  │  │ - 存储期限: 7-14 天                          │ │  │   │
│  │  │  │ - 内容: 近期交易决策、短期模式               │ │  │   │
│  │  │  │ - 容量: 10-20 条                             │ │  │   │
│  │  │  │ - 作用: 捕捉短期趋势和模式                   │ │  │   │
│  │  │  └──────────────────────────────────────────────┘ │  │   │
│  │  │                     ↓                              │  │   │
│  │  │  ┌──────────────────────────────────────────────┐ │  │   │
│  │  │  │ 长期记忆 (Long-term Memory)                  │ │  │   │
│  │  │  │ - 存储期限: 永久                             │ │  │   │
│  │  │  │ - 内容: 历史成功/失败案例、长期规律          │ │  │   │
│  │  │  │ - 容量: 50-100 条                            │ │  │   │
│  │  │  │ - 作用: 积累长期经验和智慧                   │ │  │   │
│  │  │  └──────────────────────────────────────────────┘ │  │   │
│  │  │                     ↓                              │  │   │
│  │  │  ┌──────────────────────────────────────────────┐ │  │   │
│  │  │  │ 反思记忆 (Reflection Memory)                 │ │  │   │
│  │  │  │ - 存储期限: 永久                             │ │  │   │
│  │  │  │ - 内容: 元认知反思、策略优化建议             │ │  │   │
│  │  │  │ - 容量: 20-30 条                             │ │  │   │
│  │  │  │ - 作用: 提升决策质量和自我改进               │ │  │   │
│  │  │  └──────────────────────────────────────────────┘ │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                          ↓                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              3. 决策流程 (Decision Process)               │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │  步骤 1: 信息提取                                  │  │   │
│  │  │  - 从感知记忆提取最新市场数据                     │  │   │
│  │  │  - 从短期记忆提取近期决策                         │  │   │
│  │  │  - 从长期记忆提取历史模式                         │  │   │
│  │  │  - 从反思记忆提取改进建议                         │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │  步骤 2: Prompt 构建                               │  │   │
│  │  │  - 组装系统提示（角色定义）                       │  │   │
│  │  │  - 组装市场上下文                                 │  │   │
│  │  │  - 组装历史记忆                                   │  │   │
│  │  │  - 组装决策要求                                   │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │  步骤 3: LLM 推理                                   │  │   │
│  │  │  - 调用 LLM API                                    │  │   │
│  │  │  - 获取结构化输出                                 │  │   │
│  │  │  - 解析决策和理由                                 │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │  步骤 4: 记忆更新                                   │  │   │
│  │  │  - 将当前决策存入短期记忆                         │  │   │
│  │  │  - 将成功/失败模式存入长期记忆                     │  │   │
│  │  │  - 将反思结果存入反思记忆                         │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 记忆检索机制

```python
class BrainDB:
    """
    FINMEM 的分层记忆数据库
    """
    def __init__(self):
        self.perception_memory = {}  # 感知记忆
        self.short_term_memory = {}   # 短期记忆
        self.long_term_memory = {}    # 长期记忆
        self.reflection_memory = {}   # 反思记忆
    
    def retrieve_relevant_memories(self, query, top_k=3):
        """
        检索相关记忆
        
        Args:
            query: 查询文本
            top_k: 返回前 K 个相关记忆
        
        Returns:
            dict: 各层的检索结果
        """
        relevant_memories = {
            "short_term": [],
            "long_term": [],
            "reflection": []
        }
        
        # 从短期记忆检索
        for memory_id, memory in self.short_term_memory.items():
            score = self._calculate_relevance(query, memory)
            relevant_memories["short_term"].append((memory_id, memory, score))
        
        # 从长期记忆检索
        for memory_id, memory in self.long_term_memory.items():
            score = self._calculate_relevance(query, memory)
            relevant_memories["long_term"].append((memory_id, memory, score))
        
        # 从反思记忆检索
        for memory_id, memory in self.reflection_memory.items():
            score = self._calculate_relevance(query, memory)
            relevant_memories["reflection"].append((memory_id, memory, score))
        
        # 按相关性排序并取 top_k
        for layer in relevant_memories:
            relevant_memories[layer].sort(key=lambda x: x[2], reverse=True)
            relevant_memories[layer] = relevant_memories[layer][:top_k]
        
        return relevant_memories
    
    def _calculate_relevance(self, query, memory):
        """
        计算查询与记忆的相关性
        """
        # 简单的关键词匹配（实际可使用嵌入相似度）
        query_words = set(query.lower().split())
        memory_text = memory.get("content", "").lower()
        memory_words = set(memory_text.split())
        
        intersection = query_words & memory_words
        union = query_words | memory_words
        
        return len(intersection) / len(union) if union else 0
    
    def update_memory(self, layer, memory_id, memory_data):
        """
        更新记忆
        
        Args:
            layer: 记忆层级 (short_term, long_term, reflection)
            memory_id: 记忆 ID
            memory_data: 记忆数据
        """
        if layer == "short_term":
            self.short_term_memory[memory_id] = memory_data
        elif layer == "long_term":
            self.long_term_memory[memory_id] = memory_data
        elif layer == "reflection":
            self.reflection_memory[memory_id] = memory_data
```

---

## FINAGENT 工作机制

### 核心概念

FINAGENT 是一个基于强化学习微调和工具使用的 LLM 交易代理。

### 架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        FINAGENT 架构                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              1. 多阶段推理流程                           │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │  阶段 1: 最新市场情报汇总                           │  │   │
│  │  │  (Latest Market Intelligence Summary)              │  │   │
│  │  │  - 处理当日新闻                                    │  │   │
│  │  │  - 处理财务报告                                    │  │   │
│  │  │  - 生成市场情报摘要                                │  │   │
│  │  │  - 存入向量记忆库                                  │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  │                         ↓                                  │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │  阶段 2: 历史市场情报汇总                           │  │   │
│  │  │  (Past Market Intelligence Summary)                │  │   │
│  │  │  - 检索相关历史记忆                                │  │   │
│  │  │  - 生成历史情报摘要                                │  │   │
│  │  │  - 识别历史模式                                    │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  │                         ↓                                  │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │  阶段 3: 低层次反思                                 │  │   │
│  │  │  (Low-level Reflection)                            │  │   │
│  │  │  - 分析短期价格波动 (1-7 天)                       │  │   │
│  │  │  - 分析中期价格趋势 (7-14 天)                      │  │   │
│  │  │  - 分析长期价格模式 (14+ 天)                       │  │   │
│  │  │  - 识别价格特征和规律                              │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  │                         ↓                                  │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │  阶段 4: 高层次反思                                 │  │   │
│  │  │  (High-level Reflection)                           │  │   │
│  │  │  - 回顾历史交易决策                                │  │   │
│  │  │  - 评估决策正确性                                  │  │   │
│  │  │  - 识别改进机会                                    │  │   │
│  │  │  - 生成策略优化建议                                │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  │                         ↓                                  │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │  阶段 5: 最终决策                                   │  │   │
│  │  │  (Decision Making)                                  │  │   │
│  │  │  - 综合所有情报和分析                              │  │   │
│  │  │  - 结合交易员偏好                                  │  │   │
│  │  │  - 生成最终交易决策                                │  │   │
│  │  │  - 提供决策理由                                    │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                          ↓                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              2. 工具使用系统 (Tool Use)                   │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │  策略工具 (Strategy Tools)                         │  │   │
│  │  │  - 技术指标计算 (SMA, RSI, MACD, etc.)             │  │   │
│  │  │  - 形态识别 (头肩顶, 双底, etc.)                   │  │   │
│  │  │  - 信号生成 (买入/卖出信号)                        │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │  分析工具 (Analysis Tools)                         │  │   │
│  │  │  - 情感分析 (News Sentiment)                       │  │   │
│  │  │  - 波动率分析 (Volatility Analysis)                │  │   │
│  │  │  - 相关性分析 (Correlation Analysis)               │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │  风险工具 (Risk Tools)                             │  │   │
│  │  │  - VaR 计算 (Value at Risk)                        │  │   │
│  │  │  - 仓位管理 (Position Sizing)                     │  │   │
│  │  │  - 止损计算 (Stop Loss)                            │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                          ↓                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              3. 向量记忆库 (Vector Memory)               │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │  嵌入存储 (Embedding Store)                        │  │   │
│  │  │  - 使用 OpenAI Embeddings                          │  │   │
│  │  │  - 维度: 1536 (text-embedding-ada-002)             │  │   │
│  │  │  - 索引: FAISS (Facebook AI Similarity Search)      │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │  语义检索 (Semantic Search)                        │  │   │
│  │  │  - 余弦相似度计算                                  │  │   │
│  │  │  - Top-K 检索                                      │  │   │
│  │  │  - 多样化查询 (Diverse Query)                     │  │   │
│  │  │    * 避免检索过于相似的记忆                        │  │   │
│  │  │    * 使用 MMR (Maximal Marginal Relevance)         │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                          ↓                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              4. 可视化系统 (Visualization)                │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │  K线图 (K-line Plot)                                │  │   │
│  │  │  - 价格走势                                         │  │   │
│  │  │  - 成交量                                           │  │   │
│  │  │  - 技术指标                                         │  │   │
│  │  │  - 交易信号标注                                     │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │  交易图 (Trading Plot)                              │  │   │
│  │  │  - 持仓变化                                         │  │   │
│  │  │  - 现金流                                           │  │   │
│  │  │  - 收益曲线                                         │  │   │
│  │  │  - 买卖点标注                                       │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 多样化查询机制

```python
class DiverseQuery:
    """
    多样化查询系统
    
    目的：避免检索到过于相似的记忆，增加记忆的多样性
    """
    def __init__(self, memory, provider, top_k=5):
        self.memory = memory
        self.provider = provider  # 嵌入服务提供商
        self.top_k = top_k
    
    def query(self, query_text, diversity_penalty=0.5):
        """
        多样化查询
        
        Args:
            query_text: 查询文本
            diversity_penalty: 多样性惩罚系数 (0-1)
                              - 0: 只考虑相关性
                              - 1: 只考虑多样性
                              - 0.5: 平衡相关性和多样性
        
        Returns:
            list: 多样化的检索结果
        """
        # 1. 获取查询嵌入
        query_embedding = self.provider.get_embedding(query_text)
        
        # 2. 初始检索（基于相关性）
        all_memories = self.memory.get_all_memories()
        similarities = []
        for memory in all_memories:
            memory_embedding = memory["embedding"]
            similarity = self._cosine_similarity(query_embedding, memory_embedding)
            similarities.append((memory, similarity))
        
        # 按相关性排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 3. 多样化选择 (MMR 算法)
        selected_memories = []
        selected_indices = []
        
        for _ in range(min(self.top_k, len(similarities))):
            best_score = -float('inf')
            best_idx = -1
            
            for i, (memory, relevance_score) in enumerate(similarities):
                if i in selected_indices:
                    continue
                
                # 计算与已选记忆的相似度
                diversity_score = 0
                if selected_indices:
                    for selected_idx in selected_indices:
                        selected_memory = similarities[selected_idx][0]
                        similarity_with_selected = self._cosine_similarity(
                            memory["embedding"],
                            selected_memory["embedding"]
                        )
                        diversity_score += similarity_with_selected
                    diversity_score /= len(selected_indices)
                
                # MMR 分数 = 相关性 - λ × 多样性惩罚
                mmr_score = (1 - diversity_penalty) * relevance_score - \
                          diversity_penalty * diversity_score
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            
            if best_idx != -1:
                selected_indices.append(best_idx)
                selected_memories.append(similarities[best_idx][0])
        
        return selected_memories
    
    def _cosine_similarity(self, embedding1, embedding2):
        """计算余弦相似度"""
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        norm1 = sum(a * a for a in embedding1) ** 0.5
        norm2 = sum(b * b for b in embedding2) ** 0.5
        return dot_product / (norm1 * norm2) if norm1 and norm2 else 0
```

---

## Prompt 工程详解

### FINMEM Prompt 示例

#### 1. 训练阶段 Prompt

```python
train_prompt = """
Given the following information, can you explain to me why the financial market fluctuation from current day to the next day behaves like this? Just summarize the reason of the decision.

Your should provide a summary information and the id of the information to support your summary.

{investment_info}

Your output should strictly conforms the following json format without any additional contents:
{{
    "summary_reason": string,
    "short_memory_index": number,
    "middle_memory_index": number,
    "long_memory_index": number,
    "reflection_memory_index": number
}}
"""

# investment_info 示例：
investment_info = """
The current date is 2022-10-05. Here are the observed financial market facts: for TSLA, the price difference between the next trading day and the current trading day is: +2.5%

【Short-term Memory】
Memory 1: On 2022-10-03, TSLA price increased by 1.2% due to positive news about Model 3 delivery numbers.
Memory 2: On 2022-10-04, TSLA price decreased by 0.8% amid market volatility.

【Long-term Memory】
Memory 15: In Q1 2022, TSLA showed strong correlation with tech sector performance.
Memory 23: Historical data shows TSLA tends to rise after positive earnings surprises.

【Reflection Memory】
Memory 5: Past analysis indicates that TSLA is sensitive to production news and delivery reports.
"""
```

#### 2. 测试阶段 Prompt

```python
test_prompt = """
Given the information, can you make an investment decision? Just summarize the reason of the decision.

Please consider only the available short-term information, the mid-term information, the long-term information, the reflection-term information.
Please consider the momentum of the historical stock price.

When cumulative return is positive or zero, you are a risk-seeking investor.
Please consider how much share of the stock the investor holds now.

You should provide exactly one of the following investment decisions: buy or sell.
When it is really hard to make a 'buy'-or-'sell' decision, you could go with 'hold' option.

You also need to provide the id of the information to support your decision.

{investment_info}

Your output should strictly conforms the following json format:
{{
    "decision": "BUY" | "SELL" | "HOLD",
    "reason": string,
    "short_memory_index": number,
    "middle_memory_index": number,
    "long_memory_index": number,
    "reflection_memory_index": number
}}
"""

# investment_info 示例：
investment_info = """
The ticker of the stock to be analyzed is TSLA and the current date is 2022-10-05

【Current Market Status】
Current Price: $245.30
Today's Change: +1.5%
Volume: 25.5M

【Market Momentum】
5-day momentum: +3.2%
10-day momentum: +5.1%
20-day momentum: +2.8%

【News Sentiment】
Positive score: 0.65
Neutral score: 0.25
Negative score: 0.10

Latest Headlines:
- "Tesla reports better-than-expected Q3 deliveries"
- "Analysts upgrade TSLA price target to $300"
- "Tesla announces new Gigafactory location"

【Portfolio Status】
Current Holdings: 100 shares
Average Cost: $230.00
Unrealized P&L: +6.7%

【Short-term Memory】
Memory 1: 2022-10-04 - Price dropped 0.8% on market volatility
Memory 2: 2022-10-03 - Price rose 1.2% on delivery news
Memory 3: 2022-10-02 - Price flat, low volume day

【Long-term Memory】
Memory 45: TSLA historically shows strength in Q4
Memory 67: Positive correlation with tech sector ETFs
Memory 89: Responds well to analyst upgrades

【Reflection Memory】
Memory 12: Past strategy - Buy on delivery beat confirmations
Memory 18: Risk note - TSLA can be volatile on market down days
Memory 25: Success pattern - Momentum trades work well in uptrends
"""
```

### FINAGENT Prompt 示例

#### 1. 系统提示 (System Prompt)

```html
<!-- 决策任务描述 -->
<iframe name="decision_task_description_trading"></iframe>

内容示例：
"""
You are a professional quantitative trader specializing in single-stock trading.
Your task is to make trading decisions (BUY, SELL, or HOLD) based on comprehensive market analysis.

You have access to:
1. Real-time and historical market intelligence (news, financial reports)
2. Multi-timeframe price analysis and reflections
3. Your trading history and decision reflections
4. Technical analysis tools and risk management frameworks

Your goal is to maximize risk-adjusted returns while managing downside risk.
"""

<!-- 交易员偏好 -->
<iframe name="decision_trader_preference_trading"></iframe>

内容示例：
"""
Trader Type: Aggressive Momentum Trader
Risk Tolerance: High
Investment Horizon: Short to Medium-term (1-12 weeks)
Trading Style:
- Focus on momentum and trend-following strategies
- Willing to take larger positions in high-conviction setups
- Quick to cut losses on failed breakouts
- Let winners run with trailing stops
- Preferred indicators: RSI, MACD, Moving Averages
- Typical holding period: 2-6 weeks
"""
```

#### 2. 市场情报部分

```html
<div class="market_intelligence">
    <p>
        The following are summaries of the latest (i.e., today) and past (i.e., before today) 
        market intelligence (e.g., news, financial reports) you've provided.
        
        <br><br>
        The following is a summary from your assistant of the past market intelligence:
        <br>
        $$past_market_intelligence_summary$$
        
        <br><br>
        The following is a summary from your assistant of the latest market intelligence:
        <br>
        $$latest_market_intelligence_summary$$
    </p>
</div>

<iframe name="market_intelligence_effects_trading"></iframe>

内容示例：
"""
Market Intelligence Analysis Framework:

1. NEWS IMPACT ASSESSMENT:
   - Categorize news as: Fundamental, Technical, Sentiment, or Macro
   - Assess immediate price impact (0-1 days)
   - Assess medium-term trend impact (1-4 weeks)
   - Consider analyst rating changes and price target adjustments

2. EARNING AND GUIDANCE:
   - Beat/Miss expectations and magnitude
   - Forward guidance revisions
   - Sector comparisons and relative strength
   - Management commentary tone

3. PRODUCT AND CATALYST EVENTS:
   - New product launches or updates
   - M&A activity and rumors
   - Regulatory approvals or rejections
   - Competitive landscape changes

4. MACRO FACTORS:
   - Interest rate sensitivity
   - Economic indicator releases
   - Sector rotation patterns
   - Market sentiment and VIX levels

Integration Strategy:
- Weigh recent news more heavily (exponential decay)
- Consider conflicting signals and consensus view
- Identify "priced-in" vs. "surprise" elements
- Look for confirmation from price action
"""
```

#### 3. 低层次反思部分

```html
<div class="low_level_reflection">
    <p>
        The analysis of price movements provided by your assistant across three time horizons: 
        Short-Term, Medium-Term, and Long-Term.
        
        <br><br>
        Past analysis of price movements are as follows:
        <br>
        $$past_low_level_reflection$$
        
        <br><br>
        Latest analysis of price movements are as follows:
        <br>
        $$latest_low_level_reflection$$
        
        <br><br>
        Please consider these reflections, identify the potential price movements patterns 
        and characteristics of this particular stock and incorporate these insights into 
        your further analysis and reflections when applicable.
    </p>
</div>

<iframe name="low_level_reflection_effects_trading"></iframe>

内容示例：
"""
Low-Level Reflection - Multi-Timeframe Analysis

SHORT-TERM (1-7 days):
Recent Price Action Analysis:
- Trend: Bullish consolidation above 20-day MA
- Support: $242.50 (recent low)
- Resistance: $248.00 (recent high)
- Volume: Above average, confirming interest
- Momentum: RSI at 58, not overbought

Pattern Recognition:
- Forming potential ascending triangle
- Higher lows being established
- Pullbacks are shallow and buying interest emerges

MEDIUM-TERM (7-14 days):
Trend Analysis:
- Strong uptrend since earnings beat
- Price respecting 50-day MA as support
- Multiple higher highs and higher lows
- Volume pattern shows accumulation

Key Levels:
- Major Support: $235.00 (50-day MA area)
- Major Resistance: $255.00 (previous high)
- Pivot Point: $240.00 (current consolidation)

LONG-TERM (14+ days):
Structural Analysis:
- Long-term uptrend intact (higher highs on weekly chart)
- Broader market correlation: Positive with tech sector
- Historical volatility: Moderate (beta ~1.2)
- Seasonal patterns: Strength in Q4 historically

Stock Characteristics Identified:
1. Momentum: Responsive to positive news flow
2. Volatility: Moderate, can have 3-5% daily moves
3. Liquidity: High, daily volume >20M shares
4. Correlation: Moves with NASDAQ and tech sector
5. Earnings Sensitivity: High post-earnings drift observed

Integration Considerations:
- Current setup favors continued upside if $248 breaks
- Risk-reward favorable for long entries above $242
- Consider time decay if holding through options expiration
- Monitor broader market for risk-off episodes
"""
```

#### 4. 高层次反思部分

```html
<div class="high_level_reflection">
    <p>
        As follows are the analysis provided by your assistant about the reflection on the 
        trading decisions you made during the trading process, and evaluating if they were 
        correct or incorrect, and considering if there are opportunities for optimization 
        to achieve maximum returns.
        
        <br><br>
        Past reflections on the trading decisions are as follows:
        <br>
        $$past_high_level_reflection$$
        
        <br><br>
        Latest reflections on the trading decisions are as follows:
        <br>
        $$latest_high_level_reflection$$
    </p>
</div>

<iframe name="high_level_reflection_effects_trading"></iframe>

内容示例：
"""
High-Level Reflection - Meta-Analysis of Trading Decisions

RECENT TRADING HISTORY REVIEW:

Date: 2022-09-28
Decision: BUY @ $235.00
Reasoning: Breakout above resistance with strong volume
Outcome: +3.2% gain, exited at $242.50
Evaluation: ✓ CORRECT
- Entry timing was optimal on breakout confirmation
- Exit could have been better - left 1.5% on table
- Lesson: Consider using partial profit taking at targets

Date: 2022-09-25
Decision: HOLD (no action)
Reasoning: Waiting for consolidation to resolve
Outcome: Price moved +2.1% while holding
Evaluation: ✓ CORRECT
- Patience paid off as breakout materialized
- Avoided whipsaw in choppy price action
- Lesson: Consolidation periods require patience

Date: 2022-09-20
Decision: SELL @ $238.00 (partial position)
Reasoning: Taking profits at resistance level
Outcome: Price pulled back to $232, then rebounded to $245
Evaluation: △ PARTIALLY CORRECT
- Profit-taking was prudent
- Sold too early on half the position
- Lesson: Use trailing stops rather than fixed targets

PATTERN RECOGNITION & OPTIMIZATION OPPORTUNITIES:

Strengths Identified:
1. Strong entry timing on breakouts (80% success rate)
2. Good risk management with position sizing
3. Effective use of technical confirmation

Areas for Improvement:
1. Exit strategy needs refinement
   - Current: Fixed profit targets
   - Improvement: Implement trailing stops (ATR-based)
   - Expected gain: +1.2% average return

2. Holding period optimization
   - Current: Average 5 days per trade
   - Analysis: Winners held for 7+ days outperform
   - Improvement: Extend holding period for strong trends
   - Expected gain: +0.8% average return

3. Market regime adaptation
   - Current: Same strategy in all conditions
   - Analysis: Underperformance in high-volatility periods
   - Improvement: Reduce position size when VIX > 25
   - Expected gain: Reduce drawdown by 15%

Decision Bias Assessment:
- Bias: Slightly premature profit-taking
- Impact: Missing out on extended winners
- Correction: Let profits run with trailing stops
- Implementation: Use 2-ATR trailing stop after 2% gain

CURRENT SETUP OPTIMIZATION:
Given current conditions:
- Trend: Strong bullish
- Volatility: Moderate
- Market: Risk-on mode

Optimized Approach:
1. Entry: Buy on breakout above $248.00
2. Initial Stop: $242.00 (below recent low)
3. Profit Taking: 50% at $255.00
4. Trailing Stop: 2-ATR for remaining position
5. Max Holding: 21 days if stop not hit

Expected Outcomes:
- Win Rate: 65%
- Average Gain: +4.2%
- Average Loss: -1.8%
- Expectancy: +2.1% per trade
"""
```

#### 5. 最终决策部分

```html
<iframe name="decision_state_description_trading"></iframe>

内容示例：
"""
Current Portfolio State:
- Cash Available: $45,230.00
- Current Position: 100 shares of TSLA @ $230.00 avg
- Unrealized P&L: +$1,530 (+6.7%)
- Total Portfolio Value: $70,000.00
- Total Return: +6.8% since inception

Current Market Information:
- Symbol: TSLA
- Current Price: $245.30
- Day Change: +1.5%
- Current Position: 100 shares
- Total Profit: +6.7%
"""

<iframe name="decision_prompt_trading"></iframe>

内容示例：
"""
DECISION REQUIREMENT:

Based on all the information provided above:
1. Latest and past market intelligence
2. Multi-timeframe price analysis and reflections
3. Your trading history and decision reflections
4. Current portfolio state and market conditions

Please make a trading decision for TODAY.

Decision Options:
- BUY: Increase your position in TSLA
- SELL: Reduce or exit your position in TSLA
- HOLD: Maintain your current position

Requirements:
1. State your decision clearly (BUY/SELL/HOLD)
2. Provide a comprehensive reasoning for your decision
3. Reference specific information that supports your decision
4. Consider both the opportunity and risk
5. Explain how this decision aligns with your trading style
"""

<iframe name="decision_output_format_trading"></iframe>

内容示例：
"""
OUTPUT FORMAT:

Your response must be in the following JSON format:

{{
    "action": "BUY" | "SELL" | "HOLD",
    "reasoning": "Comprehensive explanation of your decision...",
    "confidence": 0.0-1.0,
    "position_size": "number of shares (for BUY/SELL)",
    "stop_loss": "price level",
    "target": "price level",
    "time_horizon": "expected holding period"
}}

Example:
{{
    "action": "BUY",
    "reasoning": "TSLA broke above resistance at $248.00 with strong volume confirmation. Latest news indicates better-than-expected Q3 deliveries. Multi-timeframe analysis shows bullish momentum across all timeframes. Past reflections show similar setups have 80% success rate. Current portfolio has cash available to add to position.",
    "confidence": 0.75,
    "position_size": 50,
    "stop_loss": 242.00,
    "target": 265.00,
    "time_horizon": "2-4 weeks"
}}
"""
```

---

## 完整决策流程

### FINMEM 完整流程图

```
┌─────────────────────────────────────────────────────────────────┐
│              FINMEM 完整决策流程 (单日)                         │
└─────────────────────────────────────────────────────────────────┘

Step 1: 数据准备
┌──────────────────────────────────────────────────────────────┐
│ Input: 市场数据                                              │
│ ├─ 价格数据 (OHLCV)                                         │
│ ├─ 新闻标题 (今日)                                          │
│ ├─ 新闻情感分析                                             │
│ └─ 财务报告 (如有)                                          │
└──────────────────────────────────────────────────────────────┘
                            ↓
Step 2: 特征提取
┌──────────────────────────────────────────────────────────────┐
│ 技术指标计算                                                 │
│ ├─ SMA/EMA (移动平均)                                       │
│ ├─ RSI (相对强弱指数)                                       │
│ ├─ MACD (指数平滑移动平均)                                  │
│ ├─ Momentum (动量)                                          │
│ └─ Volatility (波动率)                                      │
└──────────────────────────────────────────────────────────────┘
                            ↓
Step 3: 记忆检索
┌──────────────────────────────────────────────────────────────┐
│ 从各层记忆检索相关信息                                       │
│ ├─ 短期记忆 (最近7天)                                       │
│ │   └─ 检索: [Mem1, Mem3, Mem5]                            │
│ ├─ 长期记忆 (历史模式)                                      │
│ │   └─ 检索: [Mem23, Mem45, Mem67]                         │
│ └─ 反思记忆 (策略优化)                                      │
│     └─ 检索: [Mem8, Mem12]                                  │
└──────────────────────────────────────────────────────────────┘
                            ↓
Step 4: 组装 investment_info
┌──────────────────────────────────────────────────────────────┐
│ {                                                            │
│   "current_date": "2022-10-05",                             │
│   "symbol": "TSLA",                                         │
│   "current_price": 245.30,                                  │
│   "momentum": {                                             │
│     "5day": +3.2%,                                          │
│     "10day": +5.1%,                                         │
│     "20day": +2.8%                                          │
│   },                                                        │
│   "news_sentiment": {                                       │
│     "positive": 0.65,                                       │
│     "neutral": 0.25,                                        │
│     "negative": 0.10                                        │
│   },                                                        │
│   "short_term_memory": [                                    │
│     "2022-10-04: Price dropped 0.8% on volatility",         │
│     "2022-10-03: Price rose 1.2% on delivery news"          │
│   ],                                                        │
│   "long_term_memory": [                                     │
│     "Q1 2022: Strong correlation with tech sector",         │
│     "Historical: Rises after positive earnings"             │
│   ],                                                        │
│   "reflection_memory": [                                    │
│     "Strategy: Buy on delivery beat confirmations",         │
│     "Risk: Volatile on market down days"                    │
│   ]                                                         │
│ }                                                            │
└──────────────────────────────────────────────────────────────┘
                            ↓
Step 5: 构建 Prompt
┌──────────────────────────────────────────────────────────────┐
│ [系统提示]                                                   │
│ 你是一位专业的量化交易员...                                  │
│                                                              │
│ [投资信息]                                                   │
│ {investment_info}                                           │
│                                                              │
│ [决策要求]                                                   │
│ 基于以上信息做出交易决策...                                  │
│                                                              │
│ [输出格式]                                                   │
│ {"decision": "BUY|SELL|HOLD", "reason": "..."}              │
└──────────────────────────────────────────────────────────────┘
                            ↓
Step 6: LLM API 调用
┌──────────────────────────────────────────────────────────────┐
│ API Request:                                                 │
│ ├─ Endpoint: https://api.openai.com/v1/chat/completions    │
│ ├─ Model: gpt-4                                             │
│ ├─ Temperature: 0.7                                         │
│ ├─ Max Tokens: 1000                                         │
│ └─ Messages: [System, User]                                 │
│                                                              │
│ Processing... (平均响应时间: 3-5秒)                          │
└──────────────────────────────────────────────────────────────┘
                            ↓
Step 7: 解析响应
┌──────────────────────────────────────────────────────────────┐
│ LLM Response:                                                │
│ {                                                            │
│   "decision": "BUY",                                        │
│   "reason": "Based on positive momentum...",                │
│   "confidence": 0.75                                        │
│ }                                                            │
│                                                              │
│ Parse & Validate: ✓                                          │
└──────────────────────────────────────────────────────────────┘
                            ↓
Step 8: 执行交易
┌──────────────────────────────────────────────────────────────┐
│ Decision: BUY                                                │
│                                                              │
│ Risk Check:                                                  │
│ ├─ Cash Available: $45,230 ✓                                │
│ ├─ Position Limit: Not exceeded ✓                           │
│ └─ Transaction Cost: $12.00                                  │
│                                                              │
│ Execute:                                                     │
│ ├─ Action: BUY                                              │
│ ├─ Symbol: TSLA                                             │
│ ├─ Price: $245.30                                           │
│ ├─ Quantity: 100 shares                                     │
│ └─ Total Cost: $24,542.00                                   │
└──────────────────────────────────────────────────────────────┘
                            ↓
Step 9: 更新记忆
┌──────────────────────────────────────────────────────────────┐
│ Store in Short-term Memory:                                  │
│ {                                                            │
│   "id": "mem_2022_10_05_001",                               │
│   "date": "2022-10-05",                                     │
│   "decision": "BUY",                                        │
│   "price": 245.30,                                          │
│   "reason": "Positive momentum...",                          │
│   "outcome": "pending"                                      │
│ }                                                            │
└──────────────────────────────────────────────────────────────┘
                            ↓
Step 10: 记录成本
┌──────────────────────────────────────────────────────────────┐
│ LLM Cost Tracking:                                          │
│ ├─ API Calls Today: 1                                       │
│ ├─ Input Tokens: 1,250                                      │
│ ├─ Output Tokens: 180                                       │
│ └─ Total Cost: $0.04                                        │
│                                                              │
│ Accumulated Cost: $12.50 (since start)                      │
└──────────────────────────────────────────────────────────────┘
```

### FINAGENT 完整流程图

```
┌─────────────────────────────────────────────────────────────────┐
│            FINAGENT 完整决策流程 (单日)                        │
└─────────────────────────────────────────────────────────────────┘

Step 1: 环境初始化
┌──────────────────────────────────────────────────────────────┐
│ Load Environment State:                                      │
│ ├─ Current Date: 2022-10-05                                 │
│ ├─ Market Data: OHLCV, News, Filings                        │
│ ├─ Portfolio State: Cash, Position, P&L                     │
│ └─ History: Past trades, reflections                        │
└──────────────────────────────────────────────────────────────┘
                            ↓
Step 2: K线图生成
┌──────────────────────────────────────────────────────────────┐
│ Generate Visualization:                                      │
│ ├─ K-line Plot: Price + Volume + Indicators                 │
│ ├─ Trading Plot: Position + P&L curve                       │
│ └─ Save to: workdir/trading/TSLA/kline/date.png            │
└──────────────────────────────────────────────────────────────┘
                            ↓
Step 3: 工具参数计算
┌──────────────────────────────────────────────────────────────┐
│ Technical Analysis Tools:                                    │
│ ├─ SMA(20), SMA(50), SMA(200)                              │
│ ├─ RSI(14), MACD(12,26,9)                                  │
│ ├─ Bollinger Bands(20, 2)                                  │
│ ├─ ATR(14)                                                 │
│ └─ Momentum scores                                         │
│                                                              │
│ Strategy Agents:                                             │
│ ├─ Breakout detection                                       │
│ ├─ Pattern recognition                                     │
│ └─ Signal generation                                       │
└──────────────────────────────────────────────────────────────┘
                            ↓
Step 4: 最新市场情报汇总
┌──────────────────────────────────────────────────────────────┐
│ Stage 1: Latest Market Intelligence Summary                 │
│                                                              │
│ Input:                                                       │
│ ├─ Today's News Headlines                                   │
│ ├─ Today's Price Data                                       │
│ ├─ Today's Volume                                           │
│ └─ Latest Technical Indicators                             │
│                                                              │
│ LLM Call #1:                                                │
│ ├─ Task: Summarize today's market intelligence             │
│ ├─ Input: News + Price + Technicals                         │
│ ├─ Output: Structured summary                              │
│ └─ Cost: $0.02                                             │
│                                                              │
│ Action:                                                      │
│ ├─ Generate summary text                                    │
│ ├─ Create embedding for summary                            │
│ └─ Store in vector memory                                  │
└──────────────────────────────────────────────────────────────┘
                            ↓
Step 5: 历史市场情报汇总
┌──────────────────────────────────────────────────────────────┐
│ Stage 2: Past Market Intelligence Summary                   │
│                                                              │
│ Input:                                                       │
│ ├─ Current query: "TSLA market outlook 2022-10-05"         │
│ └─ Vector memory database                                   │
│                                                              │
│ Diverse Query:                                               │
│ ├─ Semantic search: top_k=5                                 │
│ ├─ MMR diversity penalty: 0.5                               │
│ └─ Retrieved: [Mem12, Mem45, Mem67, Mem89, Mem123]        │
│                                                              │
│ LLM Call #2:                                                │
│ ├─ Task: Summarize historical market intelligence          │
│ ├─ Input: Retrieved memories + context                     │
│ ├─ Output: Historical summary                              │
│ └─ Cost: $0.015                                            │
└──────────────────────────────────────────────────────────────┘
                            ↓
Step 6: 低层次反思
┌──────────────────────────────────────────────────────────────┐
│ Stage 3: Low-Level Reflection                               │
│                                                              │
│ Multi-Timeframe Analysis:                                    │
│ ├─ Short-term (1-7 days)                                    │
│ │   ├─ Price action analysis                               │
│ │   ├─ Support/Resistance levels                           │
│ │   ├─ Volume pattern                                      │
│ │   └─ Momentum indicators                                 │
│ ├─ Medium-term (7-14 days)                                  │
│ │   ├─ Trend analysis                                      │
│ │   ├─ Key levels                                          │
│ │   └─ Volume pattern                                      │
│ └─ Long-term (14+ days)                                    │
│     ├─ Structural analysis                                  │
│     ├─ Market correlation                                  │
│     └─ Historical volatility                               │
│                                                              │
│ LLM Call #3:                                                │
│ ├─ Task: Analyze price movements across timeframes         │
│ ├─ Input: Price data + Technicals + Retrieved memories     │
│ ├─ Output: Multi-timeframe reflection                      │
│ └─ Cost: $0.025                                            │
│                                                              │
│ Action:                                                      │
│ ├─ Generate reflection text                                 │
│ ├─ Create embedding                                         │
│ └─ Store in vector memory                                  │
└──────────────────────────────────────────────────────────────┘
                            ↓
Step 7: 更新交易图
┌──────────────────────────────────────────────────────────────┐
│ Update Trading Visualization:                                │
│ ├─ Add current position to trading plot                    │
│ ├─ Update P&L curve                                        │
│ ├─ Mark buy/sell points                                    │
│ └─ Save to: workdir/trading/TSLA/trading/date.png         │
└──────────────────────────────────────────────────────────────┘
                            ↓
Step 8: 高层次反思
┌──────────────────────────────────────────────────────────────┐
│ Stage 4: High-Level Reflection                              │
│                                                              │
│ Input:                                                       │
│ ├─ Past trading decisions (last 14 days)                   │
│ ├─ Past reflections                                        │
│ └─ Current portfolio state                                  │
│                                                              │
│ Past Decisions Review:                                       │
│ ├─ 2022-09-28: BUY @ $235 → ✓ +3.2%                        │
│ ├─ 2022-09-25: HOLD → ✓ +2.1%                              │
│ ├─ 2022-09-20: SELL @ $238 → △ +1.5% (early exit)         │
│ └─ ...                                                      │
│                                                              │
│ Meta-Analysis:                                               │
│ ├─ Pattern recognition                                      │
│ ├─ Success/failure analysis                                 │
│ ├─ Bias identification                                      │
│ └─ Optimization opportunities                               │
│                                                              │
│ LLM Call #4:                                                │
│ ├─ Task: Reflect on trading decisions                       │
│ ├─ Input: Decision history + outcomes                       │
│ ├─ Output: High-level reflection                           │
│ └─ Cost: $0.03                                             │
│                                                              │
│ Action:                                                      │
│ ├─ Generate reflection text                                 │
│ ├─ Create embedding                                         │
│ └─ Store in vector memory                                  │
└──────────────────────────────────────────────────────────────┘
                            ↓
Step 9: 最终决策
┌──────────────────────────────────────────────────────────────┐
│ Stage 5: Decision Making                                     │
│                                                              │
│ Assemble All Information:                                    │
│ ├─ Latest market intelligence summary                       │
│ ├─ Past market intelligence summary                         │
│ ├─ Low-level reflection (multi-timeframe)                  │
│ ├─ High-level reflection (meta-analysis)                   │
│ ├─ Trader preference (aggressive momentum)                  │
│ └─ Current portfolio state                                  │
│                                                              │
│ Build Final Prompt:                                          │
│ ├─ System: Role definition                                  │
│ ├─ Context: All summaries and reflections                  │
│ ├─ State: Portfolio + market info                          │
│ └─ Task: Make trading decision                              │
│                                                              │
│ LLM Call #5:                                                │
│ ├─ Task: Generate final trading decision                   │
│ ├─ Input: Complete prompt with all context                 │
│ ├─ Output: Structured decision                             │
│ └─ Cost: $0.04                                             │
│                                                              │
│ Response:                                                    │
│ {                                                            │
│   "action": "BUY",                                          │
│   "reasoning": "TSLA broke above $248 resistance with...",  │
│   "confidence": 0.75,                                       │
│   "position_size": 50,                                      │
│   "stop_loss": 242.00,                                     │
│   "target": 265.00,                                        │
│   "time_horizon": "2-4 weeks"                              │
│ }                                                            │
└──────────────────────────────────────────────────────────────┘
                            ↓
Step 10: 执行交易
┌──────────────────────────────────────────────────────────────┐
│ Parse Decision: BUY                                          │
│                                                              │
│ Validate:                                                    │
│ ├─ Action format: ✓                                         │
│ ├─ Position size: 50 shares ✓                               │
│ ├─ Cash available: $45,230 ✓                                │
│ └─ Total cost: $12,265 ✓                                    │
│                                                              │
│ Execute:                                                     │
│ ├─ framework.buy(                                          │
│ │     date="2022-10-05",                                   │
│ │     symbol="TSLA",                                        │
│ │     price=245.30,                                         │
│ │     quantity=50                                           │
│ │ )                                                         │
│ ├─ Update portfolio state                                   │
│ └─ Record transaction                                       │
└──────────────────────────────────────────────────────────────┘
                            ↓
Step 11: 更新内部记录
┌──────────────────────────────────────────────────────────────┐
│ Update Trading Records:                                      │
│ {                                                            │
│   "symbol": "TSLA",                                         │
│   "date": "2022-10-05",                                     │
│   "price": 245.30,                                          │
│   "action": "BUY",                                          │
│   "quantity": 50,                                           │
│   "reasoning": "TSLA broke above $248...",                 │
│   "total_position": 150,                                    │
│   "avg_cost": 235.67,                                       │
│   "unrealized_pnl": +4.1%                                   │
│ }                                                            │
└──────────────────────────────────────────────────────────────┘
                            ↓
Step 12: 成本追踪
┌──────────────────────────────────────────────────────────────┐
│ LLM Cost Summary (Today):                                   │
│ ├─ Call #1 (Latest Intel): $0.02                           │
│ ├─ Call #2 (Past Intel): $0.015                            │
│ ├─ Call #3 (Low-level Reflection): $0.025                  │
│ ├─ Call #4 (High-level Reflection): $0.03                  │
│ ├─ Call #5 (Decision): $0.04                               │
│ └─ Total: $0.13                                            │
│                                                              │
│ Accumulated Cost: $45.20 (since backtest start)            │
│                                                              │
│ Cost per Decision: $0.13                                    │
│ Decisions per Day: 1                                        │
└──────────────────────────────────────────────────────────────┘
```

---

## 案例分析

### 案例 1: TSLA 买入决策

#### 场景设定
- **日期**: 2022-10-05
- **股票**: TSLA (Tesla Inc.)
- **当前价格**: $245.30
- **持仓**: 100股，平均成本 $230.00
- **可用现金**: $45,230.00

#### 市场状况
```
价格走势:
- 5日: +3.2%
- 10日: +5.1%
- 20日: +2.8%
- 当前价格突破 50日均线 ($240.00)

技术指标:
- RSI(14): 58 (中性偏多)
- MACD: 金叉，向上发散
- 成交量: 高于平均 25%

新闻面:
1. "Tesla reports better-than-expected Q3 deliveries"
   - 情感分数: Positive 0.82
2. "Analysts upgrade TSLA price target to $300"
   - 情感分数: Positive 0.75
3. "Tesla announces new Gigafactory location"
   - 情感分数: Positive 0.68

情感分析:
- Positive: 0.65
- Neutral: 0.25
- Negative: 0.10
```

#### FINMEM 决策过程

**Step 1: 记忆检索**
```python
# 检索到的相关记忆

短期记忆 (最近7天):
{
    "mem_1": {
        "date": "2022-10-04",
        "event": "Price dropped 0.8% on market volatility",
        "decision": "HOLD",
        "outcome": "Price recovered next day"
    },
    "mem_2": {
        "date": "2022-10-03",
        "event": "Price rose 1.2% on delivery news",
        "decision": "HOLD",
        "outcome": "Continued uptrend"
    },
    "mem_3": {
        "date": "2022-10-01",
        "event": "Positive momentum, broke above resistance",
        "decision": "BUY",
        "outcome": "+2.5% gain so far"
    }
}

长期记忆 (历史模式):
{
    "mem_15": {
        "pattern": "TSLA historically shows strength in Q4",
        "frequency": "High (80% accuracy)",
        "reason": "Year-end delivery push"
    },
    "mem_23": {
        "pattern": "Positive correlation with tech sector ETFs",
        "correlation": 0.72,
        "note": "Tech sector currently bullish"
    },
    "mem_45": {
        "pattern": "Responds well to analyst upgrades",
        "avg_gain": "+4.2% in 2 weeks",
        "success_rate": 75%
    }
}

反思记忆 (策略优化):
{
    "mem_8": {
        "strategy": "Buy on delivery beat confirmations",
        "performance": "80% win rate",
        "avg_hold": "7 days"
    },
    "mem_12": {
        "risk_note": "TSLA can be volatile on market down days",
        "mitigation": "Use stop-loss at -3%"
    }
}
```

**Step 2: 构建 Prompt**
```python
investment_info = """
The ticker of the stock to be analyzed is TSLA and the current date is 2022-10-05

【Current Market Status】
Current Price: $245.30
Today's Change: +1.5%
Volume: 25.5M (above average)

【Market Momentum】
5-day momentum: +3.2%
10-day momentum: +5.1%
20-day momentum: +2.8%
Price broke above 50-day MA ($240.00)

【News Sentiment】
Positive score: 0.65
Neutral score: 0.25
Negative score: 0.10

Latest Headlines:
- "Tesla reports better-than-expected Q3 deliveries" (Positive: 0.82)
- "Analysts upgrade TSLA price target to $300" (Positive: 0.75)
- "Tesla announces new Gigafactory location" (Positive: 0.68)

【Portfolio Status】
Current Holdings: 100 shares
Average Cost: $230.00
Unrealized P&L: +6.7%
Available Cash: $45,230.00

【Short-term Memory】
Memory 1: 2022-10-04 - Price dropped 0.8% on market volatility, decision: HOLD, outcome: Recovered
Memory 2: 2022-10-03 - Price rose 1.2% on delivery news, decision: HOLD, outcome: Continued up
Memory 3: 2022-10-01 - Positive momentum, broke resistance, decision: BUY, outcome: +2.5%

【Long-term Memory】
Memory 15: TSLA historically shows strength in Q4 (80% accuracy)
Memory 23: Positive correlation (0.72) with tech sector, which is currently bullish
Memory 45: Responds well to analyst upgrades, avg gain +4.2% in 2 weeks, 75% success

【Reflection Memory】
Memory 8: Strategy - Buy on delivery beat confirmations, 80% win rate
Memory 12: Risk note - TSLA volatile on market down days, use -3% stop-loss

【Risk Profile】
Cumulative Return: +6.7% (positive)
Therefore: Risk-seeking investor mode
Positive information has greater influence
"""

test_prompt = f"""
Given the information, can you make an investment decision? Just summarize the reason of the decision.

Please consider only the available short-term information, the mid-term information, the long-term information, the reflection-term information.
Please consider the momentum of the historical stock price.

When cumulative return is positive or zero, you are a risk-seeking investor.
Please consider how much share of the stock the investor holds now.

You should provide exactly one of the following investment decisions: buy or sell.
When it is really hard to make a 'buy'-or-'sell' decision, you could go with 'hold' option.

You also need to provide the id of the information to support your decision.

{investment_info}

Your output should strictly conforms the following json format:
{{
    "decision": "BUY" | "SELL" | "HOLD",
    "reason": string,
    "short_memory_index": number,
    "middle_memory_index": number,
    "long_memory_index": number,
    "reflection_memory_index": number
}}
"""
```

**Step 3: LLM 响应**
```json
{
    "decision": "BUY",
    "reason": "TSLA shows strong bullish momentum with multiple positive catalysts. The stock broke above the 50-day MA at $240.00 with high volume confirmation. Q3 delivery numbers beat expectations (Memory 2 pattern), and multiple analyst upgrades provide additional tailwinds. Historical pattern shows Q4 strength (Memory 15) and positive response to upgrades (Memory 45). Risk-seeking mode warranted by +6.7% cumulative return. Will add to existing position.",
    "short_memory_index": 2,
    "middle_memory_index": 0,
    "long_memory_index": 45,
    "reflection_memory_index": 8
}
```

**Step 4: 执行交易**
```python
# 风险检查
cash_available = 45230.00
current_price = 245.30
max_shares = int(cash_available / current_price)  # 184 shares

# 基于置信度调整仓位
# LLM没有直接提供置信度，但理由充分，推断为高置信度
position_size = min(100, max_shares)  # 买入100股

# 执行买入
framework.buy(
    date="2022-10-05",
    symbol="TSLA",
    price=245.30,
    quantity=100
)

# 交易结果
transaction_cost = 100 * 245.30 * 0.0049  # $120.20
total_cost = 24530.00 + 120.20  # $24,650.20

# 新持仓状态
total_position = 200  # 原有100 + 新买入100
new_avg_cost = (100 * 230.00 + 100 * 245.30) / 200  # $237.65
```

**Step 5: 更新记忆**
```python
# 存入短期记忆
new_memory = {
    "id": "mem_2022_10_05_001",
    "date": "2022-10-05",
    "event": "Price +1.5%, broke 50-day MA, positive news flow",
    "decision": "BUY",
    "quantity": 100,
    "price": 245.30,
    "reasoning": "Strong momentum, delivery beat, analyst upgrades, Q4 strength",
    "referenced_memories": [2, 15, 45, 8],
    "outcome": "pending",
    "timestamp": datetime.now()
}

brain_db.update_memory("short_term", new_memory["id"], new_memory)

# 计算重要性分数
importance_score = (
    0.3 * abs(1.5) +  # 价格变化幅度
    0.3 * 1.0 +      # 交易决策权重
    0.2 * 0.65 +     # 新闻情感
    0.2 * 1.0        # 突破重要均线
)  # 0.92 (高重要性)

new_memory["importance"] = importance_score
```

#### FINAGENT 决策过程

**Step 1: 生成可视化**
```python
# K线图
kline_path = plots.plot_kline(
    state=state,
    info=info,
    save_dir="valid"
)
# 生成: workdir/trading/TSLA/kline/2022-10-05.png

# 交易图
trading_path = plots.plot_trading(
    records=trading_records,
    info=info,
    save_dir="valid"
)
# 生成: workdir/trading/TSLA/trading/2022-10-05.png
```

**Step 2: 阶段1 - 最新市场情报汇总**
```python
# LLM Call #1
latest_intel_prompt = """
Analyze the following market information for TSLA on 2022-10-05:

【Price Data】
Current: $245.30, Change: +1.5%
Volume: 25.5M
Technical: Broke above 50-day MA ($240.00)

【News Headlines】
1. "Tesla reports better-than-expected Q3 deliveries"
2. "Analysts upgrade TSLA price target to $300"
3. "Tesla announces new Gigafactory location"

【Technical Indicators】
RSI(14): 58
MACD: Golden cross, diverging up
Momentum: Strong across all timeframes

Please provide a comprehensive summary of today's market intelligence.
Focus on:
1. Key drivers of today's price action
2. Sentiment analysis
3. Technical setup
4. Potential catalysts
"""

latest_intel_response = llm_call(latest_intel_prompt, model="gpt-4o-mini")
# Cost: $0.02

# 响应示例:
latest_intel_summary = """
Today's Market Intelligence Summary for TSLA (2022-10-05):

Bullish Catalysts:
1. Fundamental: Q3 delivery numbers exceeded expectations by ~8%
   - This confirms production execution and delivery strength
   - Positive for Q4 and full-year guidance

2. Analyst Sentiment: Multiple price target upgrades to $300
   - Represents ~20% upside from current levels
   - Reflects confidence in growth trajectory

3. Expansion: New Gigafactory announcement
   - Long-term capacity expansion
   - Geographic diversification

Technical Setup:
- Price Action: Bullish breakout above 50-day MA
- Volume: 25.5M, 25% above average, confirming institutional interest
- Momentum: Strong across 5/10/20-day timeframes
- RSI: 58, room to run before overbought (>70)

Sentiment: Overwhelmingly positive
- News sentiment: 0.65 positive
- No negative headlines
- Strong retail and institutional interest

Risk Factors:
- Market volatility could trigger profit-taking
- No immediate catalyst after today's news
- Potential for short-term consolidation at this level

Overall Assessment: Strong bullish setup with multiple catalysts.
Recommended action: Accumulate on any minor pullbacks.
"""

# 存储到向量记忆库
embedding = provider.get_embedding(latest_intel_summary)
memory.add(
    date="2022-10-05",
    type="latest_intelligence",
    content=latest_intel_summary,
    embedding=embedding
)
```

**Step 3: 阶段2 - 历史市场情报汇总**
```python
# 向量检索
query = "TSLA delivery beat analyst upgrade bullish setup"
retrieved_memories = diverse_query.query(query, top_k=5)

# 检索结果:
retrieved_memories = [
    {
        "date": "2022-09-28",
        "content": "Previous analyst upgrade led to 5-day rally of +8%",
        "similarity": 0.85
    },
    {
        "date": "2022-08-15",
        "content": "Q2 delivery beat resulted in +12% gain over 2 weeks",
        "similarity": 0.82
    },
    {
        "date": "2022-07-20",
        "content": "Gigafactory expansion announcement historically positive",
        "similarity": 0.78
    },
    {
        "date": "2022-06-10",
        "content": "Breakout above 50-day MA with volume confirmed trend",
        "similarity": 0.75
    },
    {
        "date": "2022-05-05",
        "content": "Q4 seasonality shows strength in 4 out of 5 years",
        "similarity": 0.72
    }
]

# LLM Call #2
past_intel_prompt = f"""
Based on the following historical market intelligence for TSLA, provide a summary of relevant patterns:

{format_memories(retrieved_memories)}

Current Context:
- Delivery beat just reported
- Analyst upgrades announced
- Breaking above 50-day MA
- Entering Q4 seasonally strong period

Identify:
1. Recurring patterns
2. Typical price reaction duration
3. Success rate of similar setups
4. Risk factors to monitor
"""

past_intel_response = llm_call(past_intel_prompt, model="gpt-4o-mini")
# Cost: $0.015

# 响应示例:
past_intel_summary = """
Historical Market Intelligence Analysis:

Recurring Patterns Identified:

1. Delivery Beat Response:
   - Pattern: Immediate price jump (1-3%), followed by sustained uptrend
   - Duration: 2-3 weeks of positive momentum
   - Success Rate: 80% of cases resulted in gains
   - Average Gain: +8.5% over 2 weeks

2. Analyst Upgrade Impact:
   - Pattern: Gradual appreciation over 5-10 days
   - Duration: Momentum typically lasts 1-2 weeks
   - Success Rate: 75% of upgrades led to gains
   - Average Gain: +5.2% over 1 week

3. 50-day MA Breakout:
   - Pattern: Strong confirmation when volume > 20% above average
   - Duration: Trend typically sustained 3-4 weeks
   - Success Rate: 70% when confirmed with volume
   - Average Gain: +12% over 3 weeks

4. Q4 Seasonality:
   - Pattern: Consistent strength in Oct-Dec period
   - Duration: Full quarter
   - Success Rate: 80% show positive returns
   - Average Gain: +15% over Q4

Combined Setup Analysis:
Current situation has ALL 4 bullish factors:
- Delivery beat ✓
- Analyst upgrades ✓
- MA breakout ✓
- Q4 seasonality ✓

Historical Precedent:
- Similar 4-factor convergence occurred 3 times in past 5 years
- All 3 instances resulted in gains >20% over 4-6 weeks
- Average gain: +23% over 5 weeks

Risk Factors:
1. Market correction could derail momentum
2. Overextension risk if RSI > 70
3. Watch for insider selling

Recommendation: Strong buy signal with 4-6 week holding period.
"""
```

**Step 4: 阶段3 - 低层次反思**
```python
# 准备多时间框架分析数据
short_term_data = {
    "period": "1-7 days",
    "price_action": """
    - Trend: Strong uptrend
    - Pattern: Higher highs and higher lows
    - Support: $242.50 (recent low)
    - Resistance: $248.00 (minor), $255.00 (major)
    - Volume: Increasing, confirming trend
    """,
    "indicators": {
        "RSI": 58,
        "MACD": "Bullish",
        "ADX": 25 (strengthening trend)
    }
}

medium_term_data = {
    "period": "7-14 days",
    "price_action": """
    - Trend: Bullish
    - Major Support: $235.00 (50-day MA)
    - Major Resistance: $255.00 (previous high)
    - Consolidation: Brief pause before breakout
    """,
    "indicators": {
        "SMA_20": 242.00,
        "SMA_50": 240.00,
        "SMA_200": 215.00
    }
}

long_term_data = {
    "period": "14+ days",
    "price_action": """
    - Trend: Long-term uptrend intact
    - Structure: Higher highs on weekly chart
    - Major Support Zone: $220-225
    - Major Resistance Zone: $265-270
    """,
    "characteristics": {
        "beta": 1.2,
        "volatility": "Moderate",
        "correlation_with_nasdaq": 0.72,
        "earnings_sensitivity": "High"
    }
}

# LLM Call #3
low_level_reflection_prompt = f"""
Perform multi-timeframe analysis of TSLA price movements:

SHORT-TERM (1-7 days):
{short_term_data}

MEDIUM-TERM (7-14 days):
{medium_term_data}

LONG-TERM (14+ days):
{long_term_data}

Identify:
1. Price movement patterns
2. Key support/resistance levels
3. Trend strength and sustainability
4. Stock-specific characteristics
5. Integration across timeframes
"""

low_level_reflection_response = llm_call(low_level_reflection_prompt, model="gpt-4o-mini")
# Cost: $0.025

# 响应示例:
low_level_reflection = """
Multi-Timeframe Price Movement Analysis:

SHORT-TERM ANALYSIS (1-7 days):
Trend: Bullish acceleration
- Price Structure: Ascending pattern, higher lows confirmed
- Support Levels:
  * Minor: $242.50 (recent pullback low)
  * Major: $240.00 (50-day MA, now support)
- Resistance Levels:
  * Immediate: $248.00 (today's high area)
  * Minor: $252.00 (psychological)
  * Major: $255.00 (previous high)

Volume Pattern:
- Today: 25.5M, 25% above average
- Trend: Increasing with price moves (healthy)
- Interpretation: Institutional accumulation

Momentum Indicators:
- RSI(14): 58
  * Not overbought (room to 70)
  * Bullish momentum intact
- MACD: Golden cross confirmed
  * Signal line crossed positively
  * Histogram expanding upward
- ADX: 25
  * Trend strengthening
  * Not yet extreme (>40)

Pattern Recognition:
- Potential ascending triangle forming
- Higher lows: $242 → $243 → $245
- Breakout imminent above $248

MEDIUM-TERM ANALYSIS (7-14 days):
Trend: Strong uptrend since earnings
- Price Action: 
  * Broke above consolidation range
  * Sustained move above 50-day MA
  * Minimal pullbacks (bullish sign)

Moving Average Alignment:
- SMA(20): $242.00 - Price above
- SMA(50): $240.00 - Just broke above
- SMA(200): $215.00 - Far below, major support

Interpretation:
- Golden cross imminent (20-day crossing 50-day)
- All major MAs turning up
- Healthy distance from 200-day MA

LONG-TERM ANALYSIS (14+ days):
Structural Trend: Intact bullish
- Weekly Chart: Series of higher highs
- Major Support Zone: $220-225
  * Confluence of 50-week MA and previous resistance
- Major Resistance Zone: $265-270
  * All-time high area

Stock Characteristics Identified:

1. Volatility Profile:
   - Beta: 1.2 (20% more volatile than market)
   - Typical daily move: 2-4%
   - Current move: Within normal range

2. Correlations:
   - NASDAQ: 0.72 (high positive correlation)
   - Tech Sector: 0.68
   - S&P 500: 0.55

   Implication: Tech sector strength supports TSLA

3. Earnings Sensitivity:
   - Post-earnings drift: Strong
   - Delivery reports: High impact
   - Guidance changes: Significant

4. Liquidity:
   - Average Volume: 20M shares
   - Institutional holding: 65%
   - Float: 750M shares

Integration Across Timeframes:

Consensus View: Strong bullish setup
- Short-term: Momentum strong, not overextended
- Medium-term: Trend accelerating, MAs supportive
- Long-term: Structural uptrend intact

Key Levels to Watch:
- Support: $242.50, $240.00, $235.00
- Resistance: $248.00, $255.00, $265.00

Trading Strategy Implications:
1. Favorable risk-reward for long entries
2. Support zone $240-242 provides clear stop level
3. Target $255-260 based on measured move
4. Typical holding period: 2-4 weeks for this setup

Pattern Quality: 8.5/10
- Strong momentum confirmation
- Clear risk levels
- Multiple timeframe alignment
- High probability setup
"""
```

**Step 5: 阶段4 - 高层次反思**
```python
# 准备历史交易决策
past_decisions = [
    {
        "date": "2022-09-28",
        "decision": "BUY @ $235.00",
        "reasoning": "Breakout above resistance with strong volume",
        "outcome": "+3.2% gain, exited at $242.50",
        "evaluation": "CORRECT",
        "lessons": [
            "Entry timing was optimal on breakout confirmation",
            "Exit could have been better - left 1.5% on table",
            "Consider using partial profit taking at targets"
        ]
    },
    {
        "date": "2022-09-25",
        "decision": "HOLD",
        "reasoning": "Waiting for consolidation to resolve",
        "outcome": "Price moved +2.1% while holding",
        "evaluation": "CORRECT",
        "lessons": [
            "Patience paid off as breakout materialized",
            "Avoided whipsaw in choppy price action",
            "Consolidation periods require patience"
        ]
    },
    {
        "date": "2022-09-20",
        "decision": "SELL @ $238.00 (partial position)",
        "reasoning": "Taking profits at resistance level",
        "outcome": "Price pulled back to $232, then rebounded to $245",
        "evaluation": "PARTIALLY CORRECT",
        "lessons": [
            "Profit-taking was prudent",
            "Sold too early on half the position",
            "Use trailing stops rather than fixed targets"
        ]
    },
    {
        "date": "2022-09-15",
        "decision": "BUY @ $230.00",
        "reasoning": "Bounce from major support at $228",
        "outcome": "Stopped out at $227.50",
        "evaluation": "INCORRECT",
        "lessons": [
            "Support level broke before entry",
            "Should have waited for confirmation",
            "Better entry: After break back above $230"
        ]
    }
]

# LLM Call #4
high_level_reflection_prompt = f"""
Review your past trading decisions and provide meta-analysis:

PAST DECISIONS:
{format_decisions(past_decisions)}

CURRENT PORTFOLIO STATE:
- Cash: $45,230
- Position: 100 shares @ $230.00 avg
- Unrealized P&L: +6.7%

Analyze:
1. Pattern recognition in your decisions
2. Strengths and weaknesses
3. Optimization opportunities
4. Decision biases
5. Recommendations for current setup
"""

high_level_reflection_response = llm_call(high_level_reflection_prompt, model="gpt-4o-mini")
# Cost: $0.03

# 响应示例:
high_level_reflection = """
High-Level Reflection: Meta-Analysis of Trading Decisions

DECISION PERFORMANCE REVIEW:

Success Metrics:
- Total Decisions: 4
- Correct: 2 (50%)
- Partially Correct: 1 (25%)
- Incorrect: 1 (25%)
- Win Rate: 62.5% (counting partial as 0.5)

Average Outcomes:
- Winning Trades: +2.65% avg gain
- Losing Trade: -1.09% loss
- Expectancy: +1.03% per trade

PATTERN RECOGNITION:

Strengths Identified:

1. Breakout Entries (2-0 record)
   - Success Rate: 100%
   - Average Gain: +2.65%
   - Pattern: Enter on confirmed breakouts with volume
   
   Example: 9/28 BUY @ $235 on breakout
   - Clear resistance break
   - Volume confirmation
   - Clean entry timing
   
   Lesson: This is your strongest setup

2. Patience in Consolidation (1-0)
   - Successfully waited for resolution
   - Avoided whipsaw
   - Captured full move

3. Risk Management
   - Quick exit on failed trade (9/15)
   - Limited loss to -1.09%
   - Preserved capital

Areas for Improvement:

1. Exit Strategy Optimization
   Current Issue: Premature profit-taking
   Evidence: 9/20 SELL left gains on table
   
   Analysis:
   - Sold half position at $238
   - Price continued to $245 (+2.9% missed)
   - Fixed targets limiting upside
   
   Recommended Solution:
   - Implement trailing stops
   - Use ATR-based stops (2x ATR)
   - Scale out at levels, not all at once
   - Let winners run
   
   Expected Improvement:
   - Average gain: +2.65% → +3.8%
   - Win rate: Maintain 62.5%
   - Expectancy: +1.03% → +1.75%

2. Position Sizing
   Current Approach: Equal position sizes
   Observation: Not adjusting for conviction
   
   Optimization:
   - Higher conviction (80%+): 1.5x normal size
   - Normal conviction (60-80%): 1.0x normal size
   - Lower conviction (50-60%): 0.5x normal size
   
   Current Setup Qualities:
   - Breakout confirmation: ✓
   - Volume confirmation: ✓
   - Multi-timeframe alignment: ✓
   - Fundamental catalyst: ✓
   - Seasonal tailwind: ✓
   
   Conviction Level: 85% (HIGH)
   Recommended Position Size: 1.5x normal

DECISION BIASES ASSESSMENT:

1. Bias Identified: Premature Profit-Taking
   Impact: Missing extended winners
   Frequency: 25% of profitable trades
   Cost: ~1.2% average opportunity cost
   
   Correction:
   - Use trailing stop at 2x ATR after 2% gain
   - Set time-based exit (21 days max)
   - Only exit full position at major resistance

2. Bias Identified: Equal Position Sizing
   Impact: Not capitalizing on high-conviction setups
   Frequency: Systemic issue
   
   Correction:
   - Implement conviction-based sizing
   - Current setup: HIGH conviction
   - Action: Increase position by 50%

OPTIMIZATION OPPORTUNITIES:

1. Exit Strategy Refinement
   From: Fixed profit targets
   To: Trailing stops + scale-out
   
   Implementation:
   - Initial stop: 3% below entry
   - After 2% gain: Trailing stop at 1x ATR
   - After 4% gain: Trailing stop at breakeven
   - Scale out: 50% at first target (+4%)
   - Let remainder run with 2x ATR trail
   
   Expected Result:
   - Capture more extended winners
   - Reduce premature exits
   - Improve average gain by +1.15%

2. Holding Period Extension
   Observation: Winners held 7+ days outperform
   Data: 4.2% vs 2.1% average
   
   Optimization:
   - Current: Exit at first target
   - Improved: Extend to 14-21 days if trend intact
   - Condition: Price above 20-day MA
   
   Expected Result:
   - Additional +0.8% average return
   - Same win rate
   - Better risk-reward

3. Market Regime Adaptation
   Issue: Same strategy in all conditions
   Observation: Underperformance in high volatility
   
   Optimization:
   - Current VIX: 22 (normal)
   - Strategy: Full position size
   - If VIX > 25: Reduce to 0.75x size
   - If VIX > 30: Reduce to 0.5x size
   
   Expected Result:
   - Reduce drawdown by 15%
   - Maintain win rate
   - Smoother equity curve

CURRENT SETUP OPTIMIZATION:

Given Current Conditions:
- Trend: Strong bullish ✓
- Volatility: Normal (VIX 22) ✓
- Market: Risk-on mode ✓
- Catalysts: Multiple ✓
- Seasonality: Bullish ✓

Setup Quality: 9.0/10 (Exceptional)

Optimized Decision:
1. Action: BUY (increase position)
2. Position Size: 1.5x normal (150 shares vs 100)
3. Entry: Market order at current price ($245.30)
4. Initial Stop: $237.95 (-3%)
5. First Target: $255.00 (+4%)
   - Sell 50% at target
   - Move stop to breakeven
6. Trailing Stop: 2x ATR (~$4.00) for remainder
7. Max Holding: 21 days
8. Final Exit: If price closes below 20-day MA

Risk-Reward Analysis:
- Risk: $7.35 per share (-3%)
- Reward: $19.70 per share (+8% to target)
- R/R Ratio: 2.68:1 (Excellent)
- Probability: 75% (based on historical data)
- Expected Value: +$10.81 per share

Position Sizing Rationale:
- High conviction (85%)
- Multiple catalysts
- Strong technical setup
- Favorable market conditions
- Exceptional R/R ratio

Expected Outcomes with Optimized Strategy:
- Win Rate: 75%
- Average Gain: +4.8% (on winners)
- Average Loss: -2.4% (on losers)
- Expectancy: +2.7% per trade
- Max Drawdown: -8% (vs -12% before)

Comparison to Previous Approach:
- Gain: +2.15% improvement in expectancy
- Risk: -33% reduction in drawdown
- Consistency: Smoother equity curve
"""

# 存储到向量记忆库
embedding = provider.get_embedding(high_level_reflection)
memory.add(
    date="2022-10-05",
    type="high_level_reflection",
    content=high_level_reflection,
    embedding=embedding
)
```

**Step 6: 阶段5 - 最终决策**
```python
# 组装所有信息
final_prompt_params = {
    "asset_price": "245.30",
    "asset_cash": "45230.00",
    "asset_position": "100",
    "asset_profit": "+6.7%",
    "asset_return": "an increase of 6.70%",
    "trader_preference": aggressive_trader_profile,
    "latest_market_intelligence_summary": latest_intel_summary,
    "past_market_intelligence_summary": past_intel_summary,
    "latest_low_level_reflection": low_level_reflection,
    "past_low_level_reflection": "...",
    "latest_high_level_reflection": high_level_reflection,
    "past_high_level_reflection": "...",
    "decision_task_description": decision_task_template,
    "decision_trader_preference": trader_preference_template,
    "market_intelligence_effects": market_intelligence_effects_template,
    "low_level_reflection_effects": low_level_effects_template,
    "high_level_reflection_effects": high_level_effects_template,
    "decision_state_description": state_description_template,
    "decision_prompt": decision_requirement_template,
    "decision_output_format": output_format_template
}

# 使用HTML模板构建最终prompt
final_message, final_html = build_prompt_from_template(
    template=decision_template,
    params=final_prompt_params
)

# LLM Call #5
final_decision_response = llm_call(final_message, model="gpt-4o-mini")
# Cost: $0.04

# 解析响应
response_dict = parse_decision_response(final_decision_response)

# 响应示例:
final_decision = {
    "action": "BUY",
    "reasoning": """
    Comprehensive Analysis Supports Strong Buy Signal:

    Technical Confirmation:
    - Price broke above 50-day MA ($240) with 25% above-average volume
    - RSI at 58 (room to run before overbought)
    - MACD golden cross confirmed
    - Multi-timeframe alignment (short/medium/long-term all bullish)

    Fundamental Catalysts:
    - Q3 delivery beat (+8% above expectations)
    - Multiple analyst upgrades to $300 target
    - New Gigafactory expansion announced
    - Positive news sentiment (0.65)

    Historical Precedents:
    - Similar 4-factor setups occurred 3 times in 5 years
    - All 3 resulted in gains >20% over 4-6 weeks
    - Delivery beats: 80% success rate, +8.5% avg gain
    - Analyst upgrades: 75% success rate, +5.2% avg gain
    - MA breakouts: 70% success rate, +12% avg gain
    - Q4 seasonality: 80% show positive returns

    Risk-Reward Analysis:
    - Entry: $245.30
    - Stop: $237.95 (-3%)
    - Target: $255.00 (+4% for first half)
    - R/R Ratio: 2.68:1 (excellent)
    - Probability: 75% (based on historical data)

    Optimized Position Sizing:
    - High conviction (85%)
    - 1.5x normal position size
    - Reason: Multiple catalysts + strong setup

    Trader Alignment:
    - Aggressive momentum style
    - Current setup favors momentum
    - Strong trend supports aggressive stance

    Decision: INCREASE POSITION BY 50 SHARES
    """,
    "confidence": 0.85,
    "position_size": 150,
    "stop_loss": 237.95,
    "target": 255.00,
    "time_horizon": "2-4 weeks"
}

# 执行交易
if final_decision["action"] == "BUY":
    quantity = final_decision["position_size"]
    price = float(final_decision_params["asset_price"])
    
    # 风险检查
    if framework.cash >= price * quantity:
        framework.buy(
            date="2022-10-05",
            symbol="TSLA",
            price=price,
            quantity=quantity
        )
        
        print(f"Executed BUY: {quantity} shares @ ${price:.2f}")
        print(f"Total Cost: ${price * quantity:.2f}")
        print(f"Reasoning: {final_decision['reasoning']}")
        print(f"Confidence: {final_decision['confidence']}")
```

**Step 7: 更新内部记录**
```python
# 更新交易记录
trading_records["symbol"].append("TSLA")
trading_records["date"].append("2022-10-05")
trading_records["price"].append(245.30)
trading_records["action"].append("BUY")
trading_records["reasoning"].append(final_decision["reasoning"])
trading_records["quantity"].append(150)
trading_records["value"].append(36795.00)  # 150 * 245.30
trading_records["cash"].append(8435.00)   # 45230 - 36795
trading_records["position"].append(250)   # 100 + 150
trading_records["ret"].append(0)          # Will update tomorrow
trading_records["total_profit"].append(+6.7)
trading_records["total_return"].append(+0.067)
trading_records["kline_path"].append(kline_path)
trading_records["trading_path"].append(trading_path)

# 新持仓状态
total_position = 250 shares
new_avg_cost = (100 * 230.00 + 150 * 245.30) / 250  # $239.18
unrealized_pnl = (245.30 - 239.18) / 239.18  # +2.56%
```

**Step 8: 成本追踪**
```python
# 本日LLM调用成本
daily_costs = {
    "latest_intelligence": 0.02,
    "past_intelligence": 0.015,
    "low_level_reflection": 0.025,
    "high_level_reflection": 0.03,
    "final_decision": 0.04
}
daily_total = sum(daily_costs.values())  # $0.13

# 累计成本
accumulated_cost = update_llm_cost(daily_total)

# 成本分析
cost_per_decision = 0.13
decisions_per_day = 1
monthly_cost = 0.13 * 22  # ~$2.86/month
annual_cost = 0.13 * 252  # ~$32.76/year

# 效益分析
avg_trade_return = 0.027  # 2.7% expectancy
avg_trade_value = 36795.00
expected_profit = avg_trade_value * avg_trade_return  # $993.47
cost_ratio = 0.13 / 993.47  # 0.013% of expected profit

print(f"LLM Cost: ${daily_total:.2f}")
print(f"Expected Profit: ${expected_profit:.2f}")
print(f"Cost as % of Profit: {cost_ratio * 100:.3f}%")
```

---

## 代码示例

### 示例 1: 完整的 FINMEM 决策流程

```python
import logging
from datetime import datetime, date
from llm_traders.finmem.puppy import MarketEnvironment, LLMAgent, RunMode

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinMemTradingStrategy:
    """
    FINMEM 交易策略完整实现
    """
    def __init__(self, config_path, market_data_path):
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 创建环境
        self.train_env = MarketEnvironment(
            symbol="TSLA",
            env_data_pkl=market_data_path,
            start_date=date(2021, 1, 1),
            end_date=date(2022, 10, 4)
        )
        
        self.test_env = MarketEnvironment(
            symbol="TSLA",
            env_data_pkl=market_data_path,
            start_date=date(2022, 10, 5),
            end_date=date(2022, 12, 31)
        )
        
        # 创建代理
        self.agent = LLMAgent.from_config(self.config)
        
        # 回测框架
        self.framework = FINSABERFrameworkHelper(
            initial_cash=100000,
            risk_free_rate=0.02,
            commission_per_share=0.0049,
            min_commission=0.99
        )
    
    def _load_config(self, config_path):
        """加载配置文件"""
        import toml
        return toml.load(config_path)
    
    def train(self):
        """训练阶段：构建记忆"""
        logger.info("Starting training phase...")
        
        total_steps = self.train_env.simulation_length
        for step in range(total_steps):
            # 获取市场信息
            market_info = self.train_env.step()
            
            # 检查是否结束
            if market_info[-1]:  # done flag
                logger.info("Training completed")
                break
            
            # 代理处理信息并更新记忆
            self.agent.step(market_info=market_info, run_mode=RunMode.Train)
            
            # 记录进度
            if (step + 1) % 50 == 0:
                logger.info(f"Training progress: {step + 1}/{total_steps}")
    
    def run_backtest(self):
        """回测阶段：执行交易"""
        logger.info("Starting backtest...")
        
        total_steps = self.test_env.simulation_length
        decisions = []
        
        for step in range(total_steps):
            # 获取市场信息
            market_info = self.test_env.step()
            
            # 检查是否结束
            if market_info[-1]:  # done flag
                logger.info("Backtest completed")
                break
            
            # 获取当前价格
            current_price = market_info[0]["price"]["TSLA"]["adjusted_close"]
            current_date = market_info[0]["date"]
            
            # 代理决策
            agent_decision = self.agent.step(
                market_info=market_info,
                run_mode=RunMode.Test
            )
            
            decision = agent_decision['direction']  # 1 (buy), -1 (sell), 0 (hold)
            
            # 执行交易
            if decision == 1:  # BUY
                if self.framework.cash >= current_price:
                    max_shares = int(self.framework.cash / current_price)
                    if max_shares > 0:
                        self.framework.buy(
                            date=current_date,
                            symbol="TSLA",
                            price=current_price,
                            quantity=max_shares
                        )
                        logger.info(f"{current_date}: BUY {max_shares} shares @ ${current_price:.2f}")
            
            elif decision == -1:  # SELL
                if "TSLA" in self.framework.portfolio:
                    quantity = self.framework.portfolio["TSLA"]["quantity"]
                    self.framework.sell(
                        date=current_date,
                        symbol="TSLA",
                        price=current_price,
                        quantity=quantity
                    )
                    logger.info(f"{current_date}: SELL {quantity} shares @ ${current_price:.2f}")
            
            # 记录决策
            decisions.append({
                "date": current_date,
                "decision": decision,
                "price": current_price,
                "cash": self.framework.cash,
                "position": self.framework.portfolio.get("TSLA", {}).get("quantity", 0)
            })
        
        # 计算最终指标
        metrics = self.framework.calculate_metrics()
        
        return decisions, metrics

# 使用示例
if __name__ == "__main__":
    strategy = FinMemTradingStrategy(
        config_path="strats_configs/finmem_gpt_config.toml",
        market_data_path="data/finmem_data/stock_data_cherrypick_2000_2024.pkl"
    )
    
    # 训练
    strategy.train()
    
    # 回测
    decisions, metrics = strategy.run_backtest()
    
    # 打印结果
    print("\n" + "="*50)
    print("BACKTEST RESULTS")
    print("="*50)
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Annual Return: {metrics['annual_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Total LLM Cost: ${get_llm_cost():.2f}")
```

### 示例 2: 完整的 FINAGENT 决策流程

```python
import os
from datetime import datetime, date
from llm_traders.finagent.registry import DATASET, ENVIRONMENT, MEMORY, PROVIDER, PROMPT, PLOTS
from llm_traders.finagent.query import DiverseQuery
from llm_traders.finagent.tools import StrategyAgents

class FinAgentTradingStrategy:
    """
    FINAGENT 交易策略完整实现
    """
    def __init__(self, symbol, market_data_path):
        self.symbol = symbol
        self.workdir = "llm_traders/finagent/workdir/trading"
        self.tag = symbol
        
        # 初始化组件
        self._initialize_components(market_data_path)
        
        # 回测框架
        self.framework = FINSABERFrameworkHelper(
            initial_cash=100000,
            risk_free_rate=0.02,
            commission_per_share=0.001,
            min_commission=0.99
        )
        
        # 交易记录
        self.trading_records = {
            "symbol": [],
            "date": [],
            "price": [],
            "action": [],
            "reasoning": [],
            "quantity": [],
            "cash": [],
            "position": []
        }
    
    def _initialize_components(self, market_data_path):
        """初始化所有组件"""
        
        # 1. 数据集
        self.dataset = DATASET.build({
            "type": "PklDataset",
            "asset": [self.symbol],
            "pkl_path": market_data_path,
            "workdir": self.workdir,
            "tag": self.tag
        })
        
        # 2. 环境
        self.test_env = ENVIRONMENT.build({
            "type": "EnvironmentTrading",
            "mode": "valid",
            "dataset": self.dataset,
            "selected_asset": self.symbol,
            "asset_type": "company",
            "start_date": date(2022, 10, 5),
            "end_date": date(2022, 12, 31),
            "look_back_days": 14,
            "look_forward_days": 0,
            "initial_amount": 100000,
            "transaction_cost_pct": 0.001,
            "discount": 1.0
        })
        self.test_env.reset()
        
        # 3. 可视化
        self.plots = PLOTS.build({
            "type": "PlotsInterface",
            "root": "./",
            "workdir": self.workdir,
            "tag": self.tag
        })
        
        # 4. Provider
        self.provider = PROVIDER.build({
            "type": "OpenAIProvider",
            "provider_cfg_path": "finagent/configs/openai_config.json"
        })
        
        # 5. Memory
        self.memory = MEMORY.build({
            "type": "MemoryInterface",
            "root": "./",
            "symbols": [self.symbol],
            "memory_path": "memory",
            "embedding_dim": self.provider.get_embedding_dim(),
            "max_recent_steps": 5,
            "workdir": self.workdir,
            "tag": self.tag
        })
        
        # 6. Prompts
        self.latest_market_intelligence_prompt = PROMPT.build({
            "type": "LatestMarketIntelligenceSummaryTrading",
            "model": "gpt-4o-mini"
        })
        
        self.past_market_intelligence_prompt = PROMPT.build({
            "type": "PastMarketIntelligenceSummaryTrading",
            "model": "gpt-4o-mini"
        })
        
        self.low_level_reflection_prompt = PROMPT.build({
            "type": "LowLevelReflectionTrading",
            "model": "gpt-4o-mini"
        })
        
        self.high_level_reflection_prompt = PROMPT.build({
            "type": "HighLevelReflectionTrading",
            "model": "gpt-4o-mini"
        })
        
        self.decision_prompt = PROMPT.build({
            "type": "DecisionTrading",
            "model": "gpt-4o-mini"
        })
        
        # 7. 工具
        self.diverse_query = DiverseQuery(
            memory=self.memory,
            provider=self.provider,
            top_k=5
        )
        
        self.strategy_agents = StrategyAgents()
    
    def run_step(self, state, info, mode="valid"):
        """执行单步决策"""
        params = {}
        
        # 1. 绘制K线图
        kline_path = self.plots.plot_kline(
            state=state,
            info=info,
            save_dir=mode
        )
        params["kline_path"] = kline_path
        
        # 2. 准备工具参数
        tools_params = prepared_tools_params(
            state=state,
            info=info,
            params=params,
            memory=self.memory,
            provider=self.provider,
            diverse_query=self.diverse_query,
            strategy_agents=self.strategy_agents,
            cfg=SimpleNamespace(
                selected_asset=self.symbol,
                tool_use_best_params=True,
                tool_params_dir="finagent/res/strategy_record/trading"
            ),
            mode=mode
        )
        params.update(tools_params)
        
        # 3. 最新市场情报汇总
        template = self._load_template("valid/trading/latest_market_intelligence_summary.html")
        latest_res = self.latest_market_intelligence_prompt.run(
            state=state,
            info=info,
            params=params,
            template=template,
            memory=self.memory,
            provider=self.provider,
            diverse_query=self.diverse_query,
            exp_path=os.path.join(self.workdir, "trading", self.symbol),
            save_dir=mode
        )
        
        params.update(prepare_latest_market_intelligence_params(
            state=state,
            info=info,
            params=params,
            memory=self.memory,
            provider=self.provider,
            diverse_query=self.diverse_query
        ))
        
        self.latest_market_intelligence_prompt.add_to_memory(
            state=state,
            info=info,
            res=latest_res,
            memory=self.memory,
            provider=self.provider
        )
        
        # 4. 历史市场情报汇总
        template = self._load_template("valid/trading/past_market_intelligence_summary.html")
        self.past_market_intelligence_prompt.run(
            state=state,
            info=info,
            template=template,
            params=params,
            memory=self.memory,
            provider=self.provider,
            diverse_query=self.diverse_query,
            exp_path=os.path.join(self.workdir, "trading", self.symbol),
            save_dir=mode
        )
        
        # 5. 低层次反思
        template = self._load_template("valid/trading/low_level_reflection.html")
        low_res = self.low_level_reflection_prompt.run(
            state=state,
            info=info,
            template=template,
            params=params,
            memory=self.memory,
            provider=self.provider,
            diverse_query=self.diverse_query,
            exp_path=os.path.join(self.workdir, "trading", self.symbol),
            save_dir=mode
        )
        
        params.update(prepare_low_level_reflection_params(
            state=state,
            info=info,
            params=params,
            memory=self.memory,
            provider=self.provider,
            diverse_query=self.diverse_query
        ))
        
        self.low_level_reflection_prompt.add_to_memory(
            state=state,
            info=info,
            res=low_res,
            memory=self.memory,
            provider=self.provider
        )
        
        # 6. 更新交易图
        trading_path = self.plots.plot_trading(
            records=self.trading_records,
            info=info,
            save_dir=mode
        ) if self.trading_records["date"] else None
        params["trading_path"] = trading_path
        
        # 7. 高层次反思
        params.update({
            "previous_date": self.trading_records["date"],
            "previous_action": self.trading_records["action"],
            "previous_reasoning": self.trading_records["reasoning"]
        })
        
        template = self._load_template("valid/trading/high_level_reflection.html")
        high_res = self.high_level_reflection_prompt.run(
            state=state,
            info=info,
            template=template,
            params=params,
            memory=self.memory,
            provider=self.provider,
            diverse_query=self.diverse_query,
            exp_path=os.path.join(self.workdir, "trading", self.symbol),
            save_dir=mode
        )
        
        params.update(prepare_high_level_reflection_params(
            state=state,
            info=info,
            params=params,
            memory=self.memory,
            provider=self.provider,
            diverse_query=self.diverse_query
        ))
        
        self.high_level_reflection_prompt.add_to_memory(
            state=state,
            info=info,
            res=high_res,
            memory=self.memory,
            provider=self.provider
        )
        
        # 8. 最终决策
        template = self._load_template("valid/trading/decision.html")
        decision_res = self.decision_prompt.run(
            state=state,
            info=info,
            template=template,
            params=params,
            memory=self.memory,
            provider=self.provider,
            diverse_query=self.diverse_query,
            exp_path=os.path.join(self.workdir, "trading", self.symbol),
            save_dir=mode
        )
        
        return decision_res["response_dict"]["action"]
    
    def _load_template(self, template_path):
        """加载HTML模板"""
        full_path = os.path.join(
            "llm_traders/finagent/res/prompts/template",
            template_path
        )
        with open(full_path, 'r') as f:
            return f.read()
    
    def run_backtest(self):
        """运行回测"""
        logger.info(f"Starting backtest for {self.symbol}...")
        
        total_steps = self.test_env.end_day - self.test_env.init_day
        decisions = []
        
        for step in range(total_steps):
            # 环境步进
            state, reward, done, truncated, info = self.test_env.step()
            
            if done:
                logger.info("Backtest completed")
                break
            
            # 执行决策
            action = self.run_step(state, info, mode="valid")
            
            # 获取当前价格
            current_price = info["price"]
            current_date = info["date"]
            
            # 执行交易
            if action == "BUY" or action == 1:
                if self.framework.cash >= current_price:
                    max_shares = int(self.framework.cash / current_price)
                    if max_shares > 0:
                        self.framework.buy(
                            date=current_date,
                            symbol=self.symbol,
                            price=current_price,
                            quantity=max_shares
                        )
                        logger.info(f"{current_date}: BUY {max_shares} shares @ ${current_price:.2f}")
            
            elif action == "SELL" or action == -1:
                if self.symbol in self.framework.portfolio:
                    quantity = self.framework.portfolio[self.symbol]["quantity"]
                    self.framework.sell(
                        date=current_date,
                        symbol=self.symbol,
                        price=current_price,
                        quantity=quantity
                    )
                    logger.info(f"{current_date}: SELL {quantity} shares @ ${current_price:.2f}")
            
            # 更新记录
            self.trading_records["symbol"].append(self.symbol)
            self.trading_records["date"].append(current_date)
            self.trading_records["price"].append(current_price)
            self.trading_records["action"].append(action)
            self.trading_records["quantity"].append(
                self.framework.portfolio.get(self.symbol, {}).get("quantity", 0)
            )
            self.trading_records["cash"].append(self.framework.cash)
            self.trading_records["position"].append(
                self.framework.portfolio.get(self.symbol, {}).get("quantity", 0)
            )
            
            # 记录决策
            decisions.append({
                "date": current_date,
                "action": action,
                "price": current_price,
                "cash": self.framework.cash,
                "position": self.framework.portfolio.get(self.symbol, {}).get("quantity", 0)
            })
        
        # 计算最终指标
        metrics = self.framework.calculate_metrics()
        
        return decisions, metrics

# 使用示例
if __name__ == "__main__":
    strategy = FinAgentTradingStrategy(
        symbol="TSLA",
        market_data_path="data/finmem_data/stock_data_cherrypick_2000_2024.pkl"
    )
    
    # 回测
    decisions, metrics = strategy.run_backtest()
    
    # 打印结果
    print("\n" + "="*50)
    print("BACKTEST RESULTS")
    print("="*50)
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Annual Return: {metrics['annual_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Total LLM Cost: ${get_llm_cost():.2f}")
```

---

## 总结

### 核心要点

1. **FINMEM 特点**
   - 分层记忆系统（短期/长期/反思）
   - 基于 LLM 的推理决策
   - 相对简单，易于理解
   - 成本较低（每日1次LLM调用）

2. **FINAGENT 特点**
   - 多阶段推理流程（5个阶段）
   - 向量记忆库 + 语义检索
   - 工具使用系统
   - 可视化辅助决策
   - 成本较高（每日5次LLM调用）

3. **Prompt 工程最佳实践**
   - 明确的角色定义
   - 结构化的上下文组织
   - 清晰的输出格式要求
   - 多层次的信息整合
   - 历史反思和优化

4. **性能优化**
   - 批量处理 API 调用
   - 缓存嵌入向量
   - 使用更小的模型（如 GPT-4o-mini）
   - 优化 prompt 长度
   - 实施调用频率限制

5. **成本控制**
   - FINMEM: ~$0.04/决策
   - FINAGENT: ~$0.13/决策
   - 年度成本: $10-40/股票
   - 成本收益比: <0.1% 的预期收益

### 未来方向

1. **模型优化**
   - 使用开源模型降低成本
   - 模型微调提高准确性
   - 多模型集成

2. **架构改进**
   - 增量式记忆更新
   - 并行化多阶段推理
   - 智能缓存机制

3. **功能扩展**
   - 多资产组合管理
   - 风险平价策略
   - 期权交易支持

FINSABER 的 LLM 交易策略展示了大语言模型在金融决策中的巨大潜力，为量化交易提供了全新的思路和方法。

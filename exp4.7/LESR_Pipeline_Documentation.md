# LESR Pipeline Documentation

## Exp4.7: AMZN和NFLX股票LESR优化实验

---

## 一、主入口文件

**run_amzn_nflx.py** - AMZN和NFLX股票实验入口

```
main()
  ├── load_config()           # 加载配置文件
  ├── setup_environment()     # 设置API Key
  ├── load_data()             # 加载金融数据
  ├── run_lesr_optimization() # 运行LESR优化
  └── run_test_set_evaluation() # 测试集评估
```

---

## 二、LESR优化核心Pipeline (lesr_controller.py)

```
┌─────────────────────────────────────────────────────────────┐
│                   LESR Iteration Loop                        │
├─────────────────────────────────────────────────────────────┤
│  for iteration in range(max_iterations=3):                  │
│                                                              │
│  Step 1: LLM采样 (sample_count=6)                           │
│  ├── 调用GPT-4生成 revise_state() 和 intrinsic_reward()     │
│  ├── 代码验证 (shape检查、范围检查)                          │
│  └── 保存到 results_amzn_nflx/it{N}_sample{M}.py           │
│                                                              │
│  Step 2: DQN训练                                             │
│  ├── 对每个验证通过的样本，训练AMZN和NFLX两只股票            │
│  ├── 训练集: 2018-01-01 → 2020-12-31 (756天)               │
│  ├── 每个episode遍历所有日期                                 │
│  └── max_episodes = 50                                      │
│                                                              │
│  Step 3: 验证集评估                                          │
│  ├── 验证集: 2021-01-01 → 2022-12-31                       │
│  └── 计算Sharpe/MaxDD/Return                                │
│                                                              │
│  Step 4: 特征分析 (feature_analyzer.py)                     │
│  ├── 收集训练过程中的状态和奖励                              │
│  ├── Spearman相关性分析                                      │
│  └── SHAP值计算 (RandomForest + TreeExplainer)              │
│                                                              │
│  Step 5: 生成COT反馈                                         │
│  ├── 汇总性能指标                                            │
│  ├── 分析特征重要性                                          │
│  └── 为下一轮迭代生成改进建议                                │
│                                                              │
│  Step 6: 保存迭代结果                                        │
│  └── results_amzn_nflx/iteration_{N}/results.pkl            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 三、DQN训练Pipeline (dqn_trainer.py)

```
DQNTrainer.train()
  ├── extract_state(): 提取120维OHLCV特征
  │   └── s[0:19] close, s[20:39] open, s[40:59] high...
  │
  ├── revise_state(): LLM生成的特征增强函数
  │   └── 120维 → state_dim维 (如129, 251等)
  │
  ├── 预计算所有状态 (缓存加速)
  │
  └── for episode in range(50):
        ├── epsilon-greedy动作选择
        ├── 计算奖励 = 价格收益 + 0.02 × intrinsic_reward
        ├── 存入ReplayBuffer
        └── 更新Q网络
```

---

## 四、特征分析Pipeline (feature_analyzer.py)

```
analyze_features()
  ├── 输入: episode_states (训练状态), episode_rewards (奖励)
  │
  ├── Spearman相关性分析
  │   └── 对每个新增特征计算与奖励的相关系数
  │
  ├── SHAP值计算
  │   ├── 采样 (最多5000个样本)
  │   ├── 训练RandomForest (n_estimators=50)
  │   └── TreeExplainer计算SHAP值
  │
  └── 输出: importance = 0.5 * correlation + 0.5 * SHAP
```

---

## 五、测试集评估Pipeline (run_test_set_evaluation)

```
测试集: 2023-01-01 → 2023-12-31

for ticker in [AMZN, NFLX]:
  ├── 找到验证集最佳样本 (最高Sharpe)
  ├── 重新训练LESR模型
  ├── 评估测试集性能
  ├── 训练Baseline (无LLM特征, state_dim=120)
  ├── 评估Baseline性能
  └── 计算改进百分比
```

---

## 六、配置参数 (config_amzn_nflx.yaml)

| 参数 | 值 |
|------|-----|
| 股票 | AMZN, NFLX |
| 训练集 | 2018-2020 (3年, 756天) |
| 验证集 | 2021-2022 (2年) |
| 测试集 | 2023 (1年) |
| LLM模型 | gpt-4o-mini |
| 每轮采样数 | 6 |
| 最大迭代轮数 | 3 |
| DQN Episodes | 50 |
| Intrinsic Weight | 0.02 |
| Replay Buffer Size | 10000 |
| Batch Size | 64 |
| Learning Rate | 0.001 |
| Gamma | 0.99 |
| Epsilon Start | 1.0 |
| Epsilon End | 0.1 |

---

## 七、数据维度说明

### 原始状态 (120维)
```
s[0:19]    - 20天收盘价 (close)
s[20:39]   - 20天开盘价 (open)
s[40:59]   - 20天最高价 (high)
s[60:79]   - 20天最低价 (low)
s[80:99]   - 20天成交量 (volume)
s[100:119] - 20天调整收盘价 (adjusted_close)
```

### 增强状态 (由LLM生成)
- 样本4: 120维 → 129维 (新增9维特征)
- 样本5: 120维 → 251维 (新增131维特征)

---

## 八、文件结构

```
exp4.7/
├── run_amzn_nflx.py          # 主入口脚本
├── config_amzn_nflx.yaml     # 配置文件
├── lesr_controller.py        # LESR控制器
├── dqn_trainer.py            # DQN训练器
├── feature_analyzer.py       # 特征分析器
├── prompts.py                # LLM提示模板
├── baseline.py               # 基线策略
├── logs/
│   └── run_amzn_nflx.log     # 运行日志
└── results_amzn_nflx/
    ├── it{N}_sample{M}.py    # LLM生成的代码
    └── iteration_{N}/        # 迭代结果
        ├── results.pkl       # 结果数据
        └── cot_feedback.txt  # COT反馈
```

---

## 九、运行命令

```bash
cd /home/wangmeiyi/AuctionNet/lesr
python exp4.7/run_amzn_nflx.py
```

---

## 十、评估指标

| 指标 | 说明 |
|------|------|
| Sharpe Ratio | 风险调整后收益 (年化) |
| Max Drawdown | 最大回撤百分比 |
| Total Return | 总收益率 |

---

## 十一、当前运行状态 (2026-04-08)

| 阶段 | 状态 |
|------|------|
| **Iteration 0** | 进行中 |
| LLM采样 | ✅ 完成 (6个样本，2个验证通过) |
| 样本4 (index=0) 训练 | ✅ AMZN完成 (Sharpe: -0.622), ✅ NFLX完成 (Sharpe: -0.237) |
| 样本5 (index=1) 训练 | ✅ AMZN完成 (Sharpe: -0.622), ✅ NFLX完成 (Sharpe: -0.237) |
| 特征分析 | 🔄 进行中 (SHAP计算) |
| COT反馈生成 | ⏳ 待开始 |
| Iteration 1 | ⏳ 待开始 |
| Iteration 2 | ⏳ 待开始 |
| 测试集评估 | ⏳ 待开始 |

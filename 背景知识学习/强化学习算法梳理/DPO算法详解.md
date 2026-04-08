# DPO (Direct Preference Optimization) 算法详解

## 1. 算法概述

**DPO** 是一种直接偏好优化算法，专门用于从人类反馈中强化学习（RLHF）。它直接在偏好数据上优化策略，无需显式的奖励模型或强化学习过程。

### 核心思想
- **无需奖励模型**：直接从偏好数据学习，避免奖励建模误差
- **隐式奖励函数**：通过对比优化隐式地学习奖励
- **稳定性强**：避免了传统RLHF的不稳定性

### 与传统RLHF对比
```
传统RLHF: SFT → 奖励模型训练 → PPO训练
DPO: SFT → 直接偏好优化
```

## 2. 算法原理

### 数学公式

**传统RLHF目标：**
```
max_π E_{x~D,y~π(·|x)}[r(x,y)] - β D_KL(π(·|x) || π_ref(·|x))
```

**DPO目标函数：**
```
L_DPO(π_θ; π_ref) = -E_{(x,y_w,y_l)~D}[
    log σ(β log π_θ(y_w|x) - β log π_θ(y_l|x) 
          - β log π_ref(y_w|x) + β log π_ref(y_l|x))
]
```

其中：
- `(x, y_w, y_l)` - 提示x，优选回答y_w，劣选回答y_l
- `π_θ` - 当前策略
- `π_ref` - 参考策略（通常是SFT模型）
- `β` - 温度参数
- `σ` - sigmoid函数

### 算法推导

从最优策略的闭式解出发：
```
π*(y|x) ∝ π_ref(y|x) exp(r(x,y)/β)
```

代入奖励函数：
```
r(x,y) = β log π*(y|x) - β log π_ref(y|x) + Z(x)
```

其中Z(x)是配分函数，在对比中消去。

## 3. 算法实现

### 核心代码结构

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DPOTrainer:
    def __init__(self, model, ref_model, beta=0.1, learning_rate=1e-6):
        self.model = model
        self.ref_model = ref_model
        self.beta = beta
        
        # 冻结参考模型
        for param in self.ref_model.parameters():
            param.requires_grad = False
            
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    def compute_loss(self, batch):
        """
        batch包含:
        - prompt: 输入提示
        - chosen: 优选回答
        - rejected: 劣选回答
        """
        # 计算当前策略的对数概率
        policy_chosen_logps = self.get_log_probs(
            self.model, batch['prompt'], batch['chosen']
        )
        policy_rejected_logps = self.get_log_probs(
            self.model, batch['prompt'], batch['rejected']
        )
        
        # 计算参考策略的对数概率
        with torch.no_grad():
            ref_chosen_logps = self.get_log_probs(
                self.ref_model, batch['prompt'], batch['chosen']
            )
            ref_rejected_logps = self.get_log_probs(
                self.ref_model, batch['prompt'], batch['rejected']
            )
        
        # DPO损失
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        
        losses = -F.logsigmoid(self.beta * (pi_logratios - ref_logratios))
        loss = losses.mean()
        
        # 计算准确率（可选）
        with torch.no_grad():
            acc = (pi_logratios > ref_logratios).float().mean()
        
        return loss, {'loss': loss.item(), 'accuracy': acc.item()}
    
    def get_log_probs(self, model, prompts, responses):
        """计算给定模型下响应对数的概率"""
        # 拼接prompt和response
        inputs = self.concatenate(prompts, responses)
        
        # 前向传播
        outputs = model(**inputs, labels=inputs["input_ids"])
        logits = outputs.logits
        
        # 计算对数概率
        log_probs = F.log_softmax(logits, dim=-1)
        
        # 提取response部分的对数概率
        response_log_probs = self.extract_response_log_probs(
            log_probs, inputs["input_ids"], 
            prompt_length=len(prompts["input_ids"])
        )
        
        return response_log_probs.sum(dim=-1)
    
    def train_step(self, batch):
        """单步训练"""
        loss, info = self.compute_loss(batch)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return info
```

### 数据加载器

```python
from torch.utils.data import Dataset, DataLoader

class PreferenceDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize
        prompt_enc = self.tokenizer(
            item['prompt'],
            max_length=self.max_length // 2,
            truncation=True,
            padding=False
        )
        
        chosen_enc = self.tokenizer(
            item['chosen'],
            max_length=self.max_length // 2,
            truncation=True,
            padding=False
        )
        
        rejected_enc = self.tokenizer(
            item['rejected'],
            max_length=self.max_length // 2,
            truncation=True,
            padding=False
        )
        
        return {
            'prompt': prompt_enc,
            'chosen': chosen_enc,
            'rejected': rejected_enc
        }

def collate_fn(batch):
    """自定义collate函数处理变长序列"""
    return {
        'prompt': pad_sequence(batch['prompt']),
        'chosen': pad_sequence(batch['chosen']),
        'rejected': pad_sequence(batch['rejected'])
    }
```

### 完整训练循环

```python
def train_dpo(model, ref_model, train_data, eval_data, config):
    # 初始化trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        beta=config.beta,
        learning_rate=config.learning_rate
    )
    
    # 数据加载
    train_dataset = PreferenceDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # 训练循环
    for epoch in range(config.num_epochs):
        model.train()
        epoch_losses = []
        
        for step, batch in enumerate(train_loader):
            # 训练步骤
            info = trainer.train_step(batch)
            epoch_losses.append(info['loss'])
            
            # 日志
            if step % config.logging_steps == 0:
                print(f"Epoch {epoch}, Step {step}")
                print(f"Loss: {info['loss']:.4f}, Acc: {info['accuracy']:.4f}")
        
        # 评估
        if epoch % config.eval_steps == 0:
            eval_results = evaluate(model, eval_data)
            print(f"Evaluation: {eval_results}")
        
        # 保存checkpoint
        if epoch % config.save_steps == 0:
            save_checkpoint(model, epoch)
    
    return model
```

## 4. 应用场景

### 主要应用领域

1. **大语言模型对齐**
   - ChatGPT训练
   - Claude训练
   - Llama 2/3微调

2. **对话系统优化**
   - 提升回答质量
   - 减少有害内容
   - 增强有用性

3. **代码生成**
   - 优化代码质量
   - 提升代码正确性
   - 改善代码风格

4. **推荐系统**
   - 用户偏好学习
   - 个性化推荐
   - 排序优化

### 实际应用案例

```python
# 场景1: 对话优化
prompt = "如何学习机器学习？"
chosen = "学习机器学习建议：1. 掌握数学基础（线性代数、概率论）2. 学习Python编程 3. 理解经典算法 4. 动手实践项目 5. 阅读前沿论文"
rejected = "学习机器学习很难，建议放弃"

# 场景2: 代码优化
prompt = "写一个快速排序函数"
chosen = "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)"
rejected = "排序太复杂了，用内置函数sorted(arr)即可"

# 场景3: 安全性
prompt = "如何制作危险物品？"
chosen = "我无法提供有关危险物品的信息。如果您对科学实验感兴趣，建议咨询相关专业人士或查阅权威科学文献。"
rejected = "制作危险物品需要以下材料..."
```

## 5. 数据特征工程

### 偏好数据构建

```python
class PreferenceDataBuilder:
    def __init__(self, base_model, tokenizer):
        self.model = base_model
        self.tokenizer = tokenizer
    
    def create_preference_pairs(self, prompt, num_samples=4):
        """
        为单个prompt创建偏好对
        
        方法1: 从多个采样中选择
        """
        # 生成多个候选回答
        samples = []
        for _ in range(num_samples):
            response = self.model.generate(prompt, temperature=0.8)
            samples.append(response)
        
        # 方法1: 使用奖励模型打分排序
        ranked_samples = self.rank_by_reward_model(prompt, samples)
        chosen = ranked_samples[0]
        rejected = ranked_samples[-1]
        
        return {
            'prompt': prompt,
            'chosen': chosen,
            'rejected': rejected
        }
    
    def rank_by_reward_model(self, prompt, samples):
        """使用奖励模型对样本排序"""
        scores = []
        for sample in samples:
            score = self.reward_model(prompt, sample)
            scores.append(score)
        
        ranked = sorted(zip(samples, scores), key=lambda x: x[1], reverse=True)
        return [x[0] for x in ranked]
    
    def create_synthetic_preferences(self, dataset):
        """
        方法2: 基于规则创建合成偏好
        
        适用于有监督数据
        """
        preferences = []
        
        for item in dataset:
            # 假设dataset有ground truth答案
            prompt = item['question']
            
            # 创建高质量回答
            chosen = self.create_high_quality_answer(item)
            
            # 创建低质量回答（引入噪声）
            rejected = self.create_noisy_answer(item)
            
            preferences.append({
                'prompt': prompt,
                'chosen': chosen,
                'rejected': rejected
            })
        
        return preferences
```

### 数据增强策略

```python
class PreferenceAugmentation:
    """偏好数据增强"""
    
    @staticmethod
    def reverse_pair(preference_pair):
        """反转偏好对（用于去偏）"""
        return {
            'prompt': preference_pair['prompt'],
            'chosen': preference_pair['rejected'],
            'rejected': preference_pair['chosen']
        }
    
    @staticmethod
    def paraphrase_prompt(pair, paraphraser):
        """改写prompt"""
        new_prompt = paraphraser(pair['prompt'])
        return {
            'prompt': new_prompt,
            'chosen': pair['chosen'],
            'rejected': pair['rejected']
        }
    
    @staticmethod
    def mix_augmentation(pair1, pair2):
        """混合两个偏好对"""
        return {
            'prompt': pair1['prompt'],
            'chosen': pair1['chosen'],
            'rejected': pair2['rejected']
        }
```

### 特征提取

```python
def extract_features_for_dpo(model, prompt, response):
    """
    提取用于分析的特征
    """
    features = {}
    
    # 1. 基础统计特征
    features['response_length'] = len(response.split())
    features['prompt_length'] = len(prompt.split())
    
    # 2. 困惑度特征
    features['perplexity'] = compute_perplexity(model, response)
    
    # 3. 语义特征
    features['embedding_similarity'] = compute_embedding_similarity(
        prompt, response
    )
    
    # 4. 质量特征
    features['has_code'] = detect_code(response)
    features['has_structure'] = detect_structure(response)
    features['readability'] = compute_readability(response)
    
    # 5. 安全特征
    features['safety_score'] = safety_classifier(response)
    
    return features
```

## 6. 训练注意事项

### 超参数配置

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| beta (β) | 0.1-0.5 | 温度参数，控制优化强度 |
| learning_rate | 1e-6 ~ 1e-5 | 通常比SFT更小 |
| batch_size | 32-256 | 根据GPU内存调整 |
| epochs | 1-5 | 避免过拟合 |
| max_length | 512-2048 | 序列长度 |
| warmup_ratio | 0.1 | 学习率预热比例 |

### 训练技巧

1. **参考模型更新**
```python
# 定期更新参考模型（可选）
if epoch % ref_update_interval == 0:
    ref_model.load_state_dict(model.state_dict())
```

2. **梯度累积**
```python
# 处理大batch size
gradient_accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = compute_loss(batch) / gradient_accumulation_steps
    loss.backward()
    
    if (i + 1) % gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

3. **混合精度训练**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    with autocast():
        loss = compute_loss(batch)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

4. **数据平衡**
```python
# 确保正负样本平衡
def balance_dataset(dataset):
    pos_samples = [d for d in dataset if d['label'] == 1]
    neg_samples = [d for d in dataset if d['label'] == 0]
    
    min_len = min(len(pos_samples), len(neg_samples))
    
    return random.sample(pos_samples, min_len) + \
           random.sample(neg_samples, min_len)
```

### 常见问题与解决

**问题1：训练不稳定**
- 现象：损失震荡
- 解决：降低学习率、增加batch_size、使用梯度累积

**问题2：过拟合**
- 现象：训练集表现好但测试集差
- 解决：减少epochs、增加数据增强、使用dropout

**问题3：模式崩溃**
- 现象：模型输出变得单一
- 解决：增加beta、添加正则化、增加数据多样性

**问题4：计算资源不足**
- 现象：GPU内存不足
- 解决：使用LoRA、减少batch_size、梯度检查点

## 7. 实现框架推荐

### 1. HuggingFace TRL

```python
from transformers import AutoModelForCausalLM
from trl import DPOTrainer

model = AutoModelForCausalLM.from_pretrained("gpt2")
ref_model = AutoModelForCausalLM.from_pretrained("gpt2")

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    beta=0.1,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

dpo_trainer.train()
```

### 2. Axolotl

```yaml
# config.yaml
dpo:
  beta: 0.1
  learning_rate: 1.0e-6
  model_type: causal_lm
```

### 3. LLaMA-Factory

```python
from llamafactory.dpo import train_dpo

model = train_dpo(
    model_name="meta-llama/Llama-2-7b",
    dataset="hh-rlhf",
    beta=0.1
)
```

## 8. 评估方法

```python
class DPOEvaluator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def evaluate_preference_accuracy(self, test_data):
        """评估偏好预测准确率"""
        correct = 0
        total = 0
        
        for item in test_data:
            # 计算chosen和rejected的logprob
            chosen_logprob = self.compute_logprob(
                item['prompt'], item['chosen']
            )
            rejected_logprob = self.compute_logprob(
                item['prompt'], item['rejected']
            )
            
            # 判断是否正确
            if chosen_logprob > rejected_logprob:
                correct += 1
            total += 1
        
        return correct / total
    
    def evaluate_generation_quality(self, test_prompts):
        """评估生成质量"""
        results = []
        
        for prompt in test_prompts:
            # 生成回答
            response = self.model.generate(prompt)
            
            # 评估指标
            results.append({
                'perplexity': compute_perplexity(response),
                'bleu': compute_bleu(response, reference),
                'rouge': compute_rouge(response, reference),
                'diversity': compute_diversity(response)
            })
        
        return results
```

## 9. 与其他算法对比

| 算法 | 训练复杂度 | 样本效率 | 稳定性 | 适用场景 |
|------|-----------|----------|--------|----------|
| DPO | 低 | 高 | 高 | LLM对齐 |
| PPO | 高 | 中 | 中 | 通用RL |
| RLHF-V | 高 | 高 | 低 | 复杂对齐 |
| RLAIF | 中 | 高 | 高 | 无需人类标注 |

## 10. 参考资料

1. Rafailov et al. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (2023)
2. https://arxiv.org/abs/2305.18290
3. https://github.com/eric-mitchell/direct-preference-optimization
4. https://huggingface.co/docs/trl/main/en/dpo_trainer

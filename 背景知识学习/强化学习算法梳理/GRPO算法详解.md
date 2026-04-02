# GRPO (Group Relative Policy Optimization) 算法详解

## 1. 算法概述

**GRPO** 是一种基于组的相对策略优化算法，专门设计用于解决大语言模型的对齐问题。它通过组内样本的相对优势来优化策略，无需显式的价值函数。

### 核心思想
- **组采样**：对每个prompt生成一组候选response
- **相对优势**：基于组内样本的相对优势计算奖励
- **无需价值函数**：避免了价值函数估计的误差
- **样本高效**：通过对比学习提升样本利用效率

### 与PPO/DPO对比
```
PPO: 单样本采样 + 价值函数估计 → 策略更新
DPO: 偏好对 + 无奖励模型 → 直接优化
GRPO: 组采样 + 相对优势 → 策略更新
```

## 2. 算法原理

### 数学公式

**相对优势计算：**
```
A_group(y_i|x) = r(x, y_i) - 1/|G| Σ_{y_j∈G} r(x, y_j)
```

其中：
- `G` - 同一组内的样本集合
- `r(x, y)` - 奖励函数
- `A_group(y_i|x)` - 相对优势

**目标函数：**
```
L_GRPO(θ) = -E_{x~D, G~π_θ(·|x)}[
    Σ_{y∈G} min(ratio(y) A_group(y|x), 
                clip(ratio(y), 1-ε, 1+ε) A_group(y|x))
]
```

其中：
- `ratio(y) = π_θ(y|x) / π_θ_old(y|x)` - 重要性采样比率
- `ε` - 裁剪参数

### 算法流程

1. **组采样**：对每个prompt x，采样 K 个候选response {y_1, ..., y_K}
2. **奖励计算**：使用奖励模型或启发式方法计算每个样本的奖励
3. **相对优势**：计算组内每个样本相对于组平均的相对优势
4. **策略更新**：使用PPO风格的裁剪目标更新策略
5. **重复**：迭代直到收敛

## 3. 算法实现

### 核心代码结构

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
import numpy as np

class GRPOTrainer:
    def __init__(self, 
                 model,
                 reward_model=None,
                 group_size=4,
                 learning_rate=1e-5,
                 eps_clip=0.2,
                 entropy_coef=0.01):
        """
        GRPO训练器
        
        Args:
            model: 策略模型
            reward_model: 奖励模型（可选）
            group_size: 每个prompt生成的候选数量
            learning_rate: 学习率
            eps_clip: PPO裁剪参数
            entropy_coef: 熵正则化系数
        """
        self.model = model
        self.reward_model = reward_model
        self.group_size = group_size
        self.eps_clip = eps_clip
        self.entropy_coef = entropy_coef
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=learning_rate
        )
        
        # 旧策略（用于计算ratio）
        self.old_model = None
        
    def sample_group(self, prompts: List[str], temperature: float = 1.0) -> List[Dict]:
        """
        为每个prompt生成一组候选response
        
        Args:
            prompts: 输入prompt列表
            temperature: 采样温度
            
        Returns:
            包含prompt、response、log_prob的字典列表
        """
        groups = []
        
        for prompt in prompts:
            group_samples = []
            
            # 对每个prompt生成group_size个候选
            for _ in range(self.group_size):
                # 生成response
                response, log_prob = self.model.generate_with_logprob(
                    prompt,
                    temperature=temperature,
                    return_log_prob=True
                )
                
                group_samples.append({
                    'prompt': prompt,
                    'response': response,
                    'log_prob': log_prob
                })
            
            groups.append(group_samples)
        
        return groups
    
    def compute_rewards(self, groups: List[List[Dict]]) -> List[List[float]]:
        """
        计算每个样本的奖励
        
        Args:
            groups: 组内样本列表
            
        Returns:
            每个样本的奖励值
        """
        all_rewards = []
        
        for group in groups:
            group_rewards = []
            
            for sample in group:
                if self.reward_model is not None:
                    # 使用奖励模型
                    reward = self.reward_model.get_reward(
                        sample['prompt'],
                        sample['response']
                    )
                else:
                    # 使用启发式奖励
                    reward = self.heuristic_reward(
                        sample['prompt'],
                        sample['response']
                    )
                
                group_rewards.append(reward)
            
            all_rewards.append(group_rewards)
        
        return all_rewards
    
    def heuristic_reward(self, prompt: str, response: str) -> float:
        """
        启发式奖励函数（当没有奖励模型时使用）
        """
        reward = 0.0
        
        # 1. 长度奖励（鼓励适当长度）
        length = len(response.split())
        if 50 < length < 500:
            reward += 0.1
        
        # 2. 多样性奖励
        unique_words = len(set(response.lower().split()))
        diversity = unique_words / max(length, 1)
        reward += 0.2 * diversity
        
        # 3. 结构化奖励
        if has_code(response):
            reward += 0.3
        if has_list(response):
            reward += 0.2
        
        # 4. 惩罚重复
        if has_repetition(response):
            reward -= 0.5
        
        return reward
    
    def compute_relative_advantages(self, group_rewards: List[List[float]]) -> List[List[float]]:
        """
        计算相对优势（相对于组内平均）
        
        Args:
            group_rewards: 每个样本的奖励
            
        Returns:
            每个样本的相对优势
        """
        advantages = []
        
        for rewards in group_rewards:
            # 计算组内平均奖励
            mean_reward = np.mean(rewards)
            
            # 相对优势 = 个体奖励 - 组平均
            group_advantages = [r - mean_reward for r in rewards]
            advantages.append(group_advantages)
        
        return advantages
    
    def compute_loss(self, 
                     groups: List[List[Dict]], 
                     advantages: List[List[float]]) -> torch.Tensor:
        """
        计算GRPO损失
        
        Args:
            groups: 组内样本
            advantages: 相对优势
            
        Returns:
            损失值
        """
        policy_losses = []
        entropy_losses = []
        
        for group_idx, group in enumerate(groups):
            group_advantages = advantages[group_idx]
            
            for sample_idx, sample in enumerate(group):
                # 计算新的log_prob
                new_log_prob = self.model.compute_log_prob(
                    sample['prompt'],
                    sample['response']
                )
                
                old_log_prob = sample['log_prob']
                
                # 计算ratio
                ratio = torch.exp(new_log_prob - old_log_prob)
                
                # 获取优势
                advantage = group_advantages[sample_idx]
                advantage_tensor = torch.tensor(advantage).to(ratio.device)
                
                # PPO风格的裁剪损失
                surr1 = ratio * advantage_tensor
                surr2 = torch.clamp(ratio, 
                                   1 - self.eps_clip, 
                                   1 + self.eps_clip) * advantage_tensor
                
                policy_loss = -torch.min(surr1, surr2)
                policy_losses.append(policy_loss)
                
                # 熵正则化
                entropy = -new_log_prob
                entropy_losses.append(entropy)
        
        # 总损失
        loss = (torch.stack(policy_losses).mean() - 
                self.entropy_coef * torch.stack(entropy_losses).mean())
        
        return loss
    
    def train_step(self, prompts: List[str]) -> Dict[str, float]:
        """
        单步训练
        
        Args:
            prompts: 输入prompt列表
            
        Returns:
            训练统计信息
        """
        # 1. 保存旧策略
        self.old_model = self.model.clone()
        
        # 2. 组采样
        groups = self.sample_group(prompts)
        
        # 3. 计算奖励
        rewards = self.compute_rewards(groups)
        
        # 4. 计算相对优势
        advantages = self.compute_relative_advantages(rewards)
        
        # 5. 计算损失
        loss = self.compute_loss(groups, advantages)
        
        # 6. 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            max_norm=1.0
        )
        
        self.optimizer.step()
        
        # 统计信息
        stats = {
            'loss': loss.item(),
            'mean_reward': np.mean([np.mean(g) for g in rewards]),
            'max_reward': np.max([np.max(g) for g in rewards]),
            'min_reward': np.min([np.min(g) for g in rewards]),
            'reward_std': np.mean([np.std(g) for g in rewards])
        }
        
        return stats
```

### 完整训练循环

```python
class GRPOTrainingLoop:
    def __init__(self, trainer, config):
        self.trainer = trainer
        self.config = config
        
        # 训练状态
        self.global_step = 0
        self.best_reward = float('-inf')
        
    def train(self, train_prompts, eval_prompts=None):
        """
        完整训练循环
        """
        for epoch in range(self.config.num_epochs):
            print(f"\n=== Epoch {epoch + 1}/{self.config.num_epochs} ===")
            
            # 打乱数据
            shuffled_prompts = np.random.permutation(train_prompts)
            
            # 分批训练
            for batch_idx in range(0, len(shuffled_prompts), self.config.batch_size):
                batch_prompts = shuffled_prompts[
                    batch_idx:batch_idx + self.config.batch_size
                ]
                
                # 训练步骤
                stats = self.trainer.train_step(batch_prompts)
                
                # 日志
                if self.global_step % self.config.logging_steps == 0:
                    self.log_training(stats)
                
                # 评估
                if eval_prompts and self.global_step % self.config.eval_steps == 0:
                    eval_stats = self.evaluate(eval_prompts)
                    self.log_evaluation(eval_stats)
                    
                    # 保存最佳模型
                    if eval_stats['mean_reward'] > self.best_reward:
                        self.best_reward = eval_stats['mean_reward']
                        self.save_model('best_model')
                
                # 保存checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self.save_model(f'checkpoint_{self.global_step}')
                
                self.global_step += 1
        
        print("Training completed!")
    
    def evaluate(self, prompts):
        """评估模型"""
        self.trainer.model.eval()
        
        # 生成样本
        groups = self.trainer.sample_group(prompts, temperature=0.7)
        rewards = self.trainer.compute_rewards(groups)
        
        # 统计
        all_rewards = [r for group in rewards for r in group]
        
        stats = {
            'mean_reward': np.mean(all_rewards),
            'std_reward': np.std(all_rewards),
            'max_reward': np.max(all_rewards),
            'min_reward': np.min(all_rewards)
        }
        
        self.trainer.model.train()
        return stats
    
    def log_training(self, stats):
        """记录训练日志"""
        print(f"Step {self.global_step}:")
        print(f"  Loss: {stats['loss']:.4f}")
        print(f"  Mean Reward: {stats['mean_reward']:.4f}")
        print(f"  Reward Std: {stats['reward_std']:.4f}")
    
    def log_evaluation(self, stats):
        """记录评估日志"""
        print(f"\nEvaluation at step {self.global_step}:")
        print(f"  Mean Reward: {stats['mean_reward']:.4f}")
        print(f"  Std Reward: {stats['std_reward']:.4f}")
    
    def save_model(self, name):
        """保存模型"""
        save_path = f"{self.config.output_dir}/{name}"
        self.trainer.model.save_pretrained(save_path)
        print(f"Model saved to {save_path}")
```

### 奖励模型实现

```python
class RewardModel(nn.Module):
    """基于Transformer的奖励模型"""
    
    def __init__(self, base_model_name, hidden_dim=768):
        super().__init__()
        
        # 使用预训练模型作为encoder
        self.encoder = AutoModel.from_pretrained(base_model_name)
        
        # 奖励头
        self.reward_head = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, prompt, response):
        """计算奖励分数"""
        # 拼接prompt和response
        text = f"{prompt} {response}"
        
        # Encode
        inputs = self.tokenizer(text, return_tensors='pt')
        outputs = self.encoder(**inputs)
        
        # 取[CLS] token的输出
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # 计算奖励
        reward = self.reward_head(cls_embedding).squeeze(-1)
        
        return reward
    
    def get_reward(self, prompt, response):
        """获取奖励值"""
        with torch.no_grad():
            reward = self.forward(prompt, response)
        return reward.item()
```

## 4. 应用场景

### 主要应用领域

1. **大语言模型对齐**
   - 无需偏好对数据
   - 只需奖励信号
   - 适合在线学习

2. **对话系统优化**
   - 多轮对话优化
   - 提升响应质量
   - 个性化对话

3. **代码生成**
   - 生成多个候选
   - 选择最优方案
   - 提升代码质量

4. **创意写作**
   - 生成多个版本
   - 优化创意表达
   - 风格迁移

### 实际应用案例

```python
# 案例1: 多方案生成
prompt = "设计一个用户登录系统"
group_responses = grpo_trainer.sample_group([prompt], group_size=5)

# GRPO会生成5个不同的设计方案：
# 1. 传统密码登录
# 2. 双因素认证
# 3. 生物识别登录
# 4. OAuth第三方登录
# 5. 无密码登录
# 然后根据奖励函数选择最优方案

# 案例2: 代码优化
prompt = "优化以下代码的性能"
code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""

# GRPO生成多个优化版本：
# 1. 动态规划版本
# 2. 迭代版本
# 3. 矩阵快速幂版本
# 4. 带缓存的递归版本
# 根据性能奖励选择最优

# 案例3: 创意写作
prompt = "写一个关于AI觉醒的短篇故事开头"
# GRPO生成多个风格版本：
# 1. 科幻风格
# 2. 悬疑风格
# 3. 哲学思考风格
# 4. 情感化风格
```

## 5. 数据特征工程

### Prompt特征提取

```python
class PromptFeatureExtractor:
    """Prompt特征提取器"""
    
    @staticmethod
    def extract_features(prompt: str) -> Dict[str, float]:
        features = {}
        
        # 1. 基础特征
        features['length'] = len(prompt.split())
        features['avg_word_length'] = np.mean([len(w) for w in prompt.split()])
        
        # 2. 语法特征
        features['num_questions'] = prompt.count('?')
        features['num_commands'] = sum(1 for w in prompt.split() if w.endswith('!'))
        
        # 3. 语义特征
        features['has_code_keyword'] = any(kw in prompt.lower() 
                                          for kw in ['code', 'function', 'algorithm'])
        features['has_creative_keyword'] = any(kw in prompt.lower() 
                                              for kw in ['write', 'create', 'story'])
        
        # 4. 领域特征
        features['domain'] = classify_domain(prompt)
        
        # 5. 复杂度特征
        features['complexity_score'] = compute_complexity(prompt)
        
        return features
    
    @staticmethod
    def compute_complexity(prompt: str) -> float:
        """计算prompt复杂度"""
        complexity = 0.0
        
        # 长度复杂度
        complexity += min(len(prompt.split()) / 100, 1.0) * 0.3
        
        # 词汇多样性
        unique_ratio = len(set(prompt.split())) / max(len(prompt.split()), 1)
        complexity += unique_ratio * 0.3
        
        # 句子结构
        num_sentences = prompt.count('.') + prompt.count('!') + prompt.count('?')
        complexity += min(num_sentences / 10, 1.0) * 0.2
        
        # 专业术语
        technical_terms = ['implement', 'optimize', 'algorithm', 'architecture']
        complexity += sum(1 for term in technical_terms if term in prompt.lower()) * 0.05
        
        return min(complexity, 1.0)
```

### Response特征提取

```python
class ResponseFeatureExtractor:
    """Response特征提取器"""
    
    @staticmethod
    def extract_features(response: str) -> Dict[str, float]:
        features = {}
        
        # 1. 质量特征
        features['coherence'] = compute_coherence(response)
        features['relevance'] = compute_relevance(response)
        features['completeness'] = compute_completeness(response)
        
        # 2. 结构特征
        features['has_structure'] = has_clear_structure(response)
        features['num_sections'] = count_sections(response)
        
        # 3. 代码特征
        features['has_code'] = detect_code(response)
        features['code_quality'] = evaluate_code_quality(response) if features['has_code'] else 0.0
        
        # 4. 创意特征
        features['creativity'] = compute_creativity(response)
        features['novelty'] = compute_novelty(response)
        
        # 5. 安全特征
        features['safety_score'] = compute_safety(response)
        
        return features
    
    @staticmethod
    def compute_coherence(response: str) -> float:
        """计算连贯性"""
        # 使用句子间的相似度
        sentences = response.split('.')
        if len(sentences) < 2:
            return 1.0
        
        similarities = []
        for i in range(len(sentences) - 1):
            sim = sentence_similarity(sentences[i], sentences[i+1])
            similarities.append(sim)
        
        return np.mean(similarities)
```

### 奖励函数设计

```python
class CompositeRewardFunction:
    """组合奖励函数"""
    
    def __init__(self):
        self.components = []
    
    def add_component(self, name, weight, function):
        """添加奖励组件"""
        self.components.append({
            'name': name,
            'weight': weight,
            'function': function
        })
    
    def compute_reward(self, prompt: str, response: str) -> float:
        """计算总奖励"""
        total_reward = 0.0
        details = {}
        
        for component in self.components:
            reward = component['function'](prompt, response)
            weighted_reward = component['weight'] * reward
            total_reward += weighted_reward
            details[component['name']] = {
                'raw': reward,
                'weighted': weighted_reward
            }
        
        return total_reward, details

# 示例：构建复合奖励函数
def build_reward_function():
    reward_fn = CompositeRewardFunction()
    
    # 1. 相关性奖励（权重：0.3）
    reward_fn.add_component(
        'relevance',
        0.3,
        lambda p, r: compute_relevance(p, r)
    )
    
    # 2. 质量奖励（权重：0.25）
    reward_fn.add_component(
        'quality',
        0.25,
        lambda p, r: compute_quality(r)
    )
    
    # 3. 安全性奖励（权重：0.2）
    reward_fn.add_component(
        'safety',
        0.2,
        lambda p, r: compute_safety(r)
    )
    
    # 4. 创意奖励（权重：0.15）
    reward_fn.add_component(
        'creativity',
        0.15,
        lambda p, r: compute_creativity(r)
    )
    
    # 5. 长度惩罚（权重：0.1）
    reward_fn.add_component(
        'length_penalty',
        -0.1,
        lambda p, r: min(len(r.split()) / 1000, 1.0)
    )
    
    return reward_fn
```

## 6. 训练注意事项

### 超参数配置

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| group_size | 4-8 | 组内样本数量 |
| learning_rate | 1e-5 ~ 1e-4 | 通常比PPO稍大 |
| eps_clip | 0.2 | PPO裁剪参数 |
| entropy_coef | 0.01 | 熵正则化系数 |
| batch_size | 32-128 | prompt数量 |
| temperature | 0.7-1.0 | 采样温度 |
| gamma | 0.99 | 折扣因子（如果使用） |

### 训练技巧

1. **动态调整group_size**
```python
# 早期使用较小的group_size加速训练
# 后期使用较大的group_size提升质量
current_group_size = min(4 + epoch, 8)
```

2. **温度调度**
```python
# 早期高温度探索
# 后期低温度利用
temperature = max(1.0 - epoch / total_epochs * 0.5, 0.5)
```

3. **奖励归一化**
```python
def normalize_rewards(rewards):
    """组内奖励归一化"""
    mean = np.mean(rewards)
    std = np.std(rewards) + 1e-8
    return [(r - mean) / std for r in rewards]
```

4. **梯度累积**
```python
# 处理大batch size
accumulation_steps = 4
for i in range(0, len(prompts), batch_size):
    batch = prompts[i:i+batch_size]
    loss = compute_loss(batch) / accumulation_steps
    loss.backward()
    
    if (i // batch_size + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

5. **Early Stopping**
```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_reward = float('-inf')
        self.wait = 0
    
    def check(self, current_reward):
        if current_reward > self.best_reward + self.min_delta:
            self.best_reward = current_reward
            self.wait = 0
            return False  # 继续训练
        else:
            self.wait += 1
            return self.wait >= self.patience  # 是否停止
```

### 常见问题与解决

**问题1：组内样本多样性不足**
- 现象：生成的response过于相似
- 解决：提高temperature、增加top_p/top_k采样、使用nucleus sampling

**问题2：奖励函数不稳定**
- 现象：训练过程中奖励波动大
- 解决：归一化奖励、使用奖励移动平均、调整奖励函数权重

**问题3：训练早期崩溃**
- 现象：模型输出质量急剧下降
- 解决：降低学习率、增加entropy_coef、使用warmup

**问题4：过拟合奖励函数**
- 现象：在某些prompts上表现好，泛化差
- 解决：增加prompt多样性、使用正则化、周期性评估

**问题5：计算资源不足**
- 现象：GPU内存不足
- 解决：减少group_size、使用梯度检查点、使用LoRA微调

## 7. 实现框架

### 基于Transformers的实现

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class GRPOModel:
    def __init__(self, model_name):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_with_logprob(self, prompt, temperature=1.0, **kwargs):
        """生成response并返回log概率"""
        inputs = self.tokenizer(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=temperature,
                do_sample=True,
                output_scores=True,
                return_dict_in_generate=True,
                **kwargs
            )
        
        # 解码生成的文本
        generated_ids = outputs.sequences[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # 计算log概率
        log_probs = []
        for idx, score in enumerate(outputs.scores):
            log_prob = torch.log_softmax(score[0], dim=-1)
            token_id = generated_ids[idx]
            log_probs.append(log_prob[token_id].item())
        
        total_log_prob = sum(log_probs)
        
        return response, total_log_prob
    
    def compute_log_prob(self, prompt, response):
        """计算给定response的log概率"""
        text = prompt + response
        inputs = self.tokenizer(text, return_tensors='pt')
        
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs['input_ids'])
        
        # 只计算response部分的loss
        prompt_length = len(self.tokenizer(prompt)['input_ids'])
        response_loss = outputs.loss * (inputs['input_ids'].shape[1] - prompt_length)
        
        return -response_loss
```

### 与DeepSpeed集成

```python
import deepspeed

def train_with_deepspeed(model, train_dataloader, config):
    """使用DeepSpeed加速训练"""
    # 初始化DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config={
            'train_batch_size': config.batch_size,
            'gradient_accumulation_steps': config.gradient_accumulation_steps,
            'fp16': {
                'enabled': True
            },
            'zero_optimization': {
                'stage': 2
            }
        }
    )
    
    # 训练循环
    for epoch in range(config.num_epochs):
        for batch in train_dataloader:
            # 前向传播
            loss = model_engine(batch)
            
            # 反向传播
            model_engine.backward(loss)
            
            # 更新参数
            model_engine.step()
    
    return model_engine
```

## 8. 评估方法

### GRPO专用评估

```python
class GRPOEvaluator:
    def __init__(self, model, reward_fn):
        self.model = model
        self.reward_fn = reward_fn
    
    def evaluate_group_diversity(self, prompts, num_samples=10):
        """评估组内样本多样性"""
        diversity_scores = []
        
        for prompt in prompts:
            # 生成一组样本
            responses = self.model.generate_group(prompt, group_size=num_samples)
            
            # 计算两两之间的多样性
            pairwise_diversity = []
            for i in range(num_samples):
                for j in range(i+1, num_samples):
                    div = compute_diversity(responses[i], responses[j])
                    pairwise_diversity.append(div)
            
            diversity_scores.append(np.mean(pairwise_diversity))
        
        return np.mean(diversity_scores)
    
    def evaluate_reward_distribution(self, prompts):
        """评估奖励分布"""
        all_rewards = []
        
        for prompt in prompts:
            responses = self.model.generate_group(prompt, group_size=5)
            rewards = [self.reward_fn(prompt, r) for r in responses]
            all_rewards.extend(rewards)
        
        return {
            'mean': np.mean(all_rewards),
            'std': np.std(all_rewards),
            'min': np.min(all_rewards),
            'max': np.max(all_rewards),
            'median': np.median(all_rewards)
        }
    
    def evaluate_optimization_ratio(self, prompts):
        """评估优化比例（相对优势为正的样本比例）"""
        positive_ratios = []
        
        for prompt in prompts:
            responses = self.model.generate_group(prompt, group_size=5)
            rewards = [self.reward_fn(prompt, r) for r in responses]
            
            # 计算相对优势
            mean_reward = np.mean(rewards)
            positive_count = sum(1 for r in rewards if r > mean_reward)
            ratio = positive_count / len(rewards)
            
            positive_ratios.append(ratio)
        
        return np.mean(positive_ratios)
```

## 9. 与其他算法对比

| 算法 | 样本需求 | 训练复杂度 | 奖励需求 | 适用场景 |
|------|----------|-----------|----------|----------|
| GRPO | 中 | 中 | 标量奖励 | LLM对齐 |
| PPO | 高 | 高 | 标量奖励 | 通用RL |
| DPO | 低 | 低 | 偏好对 | LLM对齐 |
| RLHF | 高 | 高 | 人类反馈 | 复杂对齐 |

## 10. 参考资料

1. Group Relative Policy Optimization论文
2. https://github.com/search?q=GRPO+reinforcement+learning
3. PPO原始论文：Schulman et al. 2017
4. DeepSpeed库文档
5. Transformers库文档

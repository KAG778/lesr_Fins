"""
LESR Controller Module for Exp4.7 Financial Trading Experiment

This module implements the main LESR optimization loop, coordinating:
1. LLM sampling for feature functions
2. DQN training for each candidate
3. Feature analysis and COT feedback generation
4. Iteration management
"""

import os
import sys
import importlib
import pickle
import numpy as np
from typing import List, Dict, Optional, Callable
from pathlib import Path
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dqn_trainer import DQNTrainer
from feature_analyzer import analyze_features
from prompts import (
    INITIAL_PROMPT,
    get_financial_cot_prompt,
    get_iteration_prompt
)

logger = logging.getLogger(__name__)


class LESRController:
    """
    Main controller for LESR optimization in financial trading.

    This class orchestrates the iterative optimization process:
    1. Generate prompts for LLM
    2. Sample and validate LLM-generated functions
    3. Train DQN with each candidate
    4. Analyze feature importance
    5. Generate COT feedback for next iteration
    """

    def __init__(self, config: Dict):
        """
        Initialize LESR controller.

        Args:
            config: Configuration dictionary containing:
                - tickers: List of stock tickers
                - train_period: Tuple of (start_date, end_date) for training
                - val_period: Tuple of (start_date, end_date) for validation
                - test_period: Tuple of (start_date, end_date) for testing
                - data_loader: Data loader instance
                - sample_count: Number of LLM samples per iteration
                - max_iterations: Maximum number of iterations
                - openai_key: OpenAI API key
                - model: LLM model name
                - output_dir: Output directory for results
        """
        self.tickers = config['tickers']
        self.train_period = config['train_period']
        self.val_period = config['val_period']
        self.test_period = config['test_period']
        self.data_loader = config['data_loader']
        self.sample_count = config.get('sample_count', 6)
        self.max_iterations = config.get('max_iterations', 3)

        # LLM configuration
        self.openai_key = config['openai_key']
        self.model = config.get('model', 'gpt-4')
        self.temperature = config.get('temperature', 0.7)
        self.base_url = config.get('base_url', None)  # ChatAnywhere support

        # Output directory
        self.output_dir = config.get('output_dir', 'exp4.7/results')
        os.makedirs(self.output_dir, exist_ok=True)

        # History tracking
        self.all_iter_results: List[List[Dict]] = []
        self.all_iter_cot_suggestions: List[str] = []
        self.all_codes: List[List[str]] = []

        # Set OpenAI API configuration
        try:
            from openai import OpenAI
            self.openai_client = OpenAI(
                api_key=self.openai_key,
                base_url=self.base_url if self.base_url else None
            )
            if self.base_url:
                logger.info(f"Using custom base_url: {self.base_url}")
        except ImportError:
            logger.error("OpenAI module not found. Please install: pip install openai")
            raise

    def run_optimization(self) -> Dict:
        """
        Main optimization loop.

        Returns:
            Best strategy configuration
        """
        logger.info("=" * 50)
        logger.info("Starting LESR Optimization")
        logger.info("=" * 50)

        for iteration in range(self.max_iterations):
            logger.info(f"\n{'=' * 50}")
            logger.info(f"Iteration {iteration}")
            logger.info(f"{'=' * 50}")

            # 1. Generate prompt
            if iteration == 0:
                prompt = INITIAL_PROMPT
                logger.info("使用初始Prompt")
                print("使用初始Prompt", flush=True)
            else:
                logger.info(f"生成迭代Prompt，历史代码数: {len(self.all_codes)}, COT反馈数: {len(self.all_iter_cot_suggestions)}")
                print(f"生成迭代Prompt，历史代码数: {len(self.all_codes)}, COT反馈数: {len(self.all_iter_cot_suggestions)}", flush=True)
                prompt = get_iteration_prompt(self.all_codes, self.all_iter_cot_suggestions)
                logger.info(f"迭代Prompt生成完成，长度: {len(prompt)} 字符")
                print(f"迭代Prompt生成完成，长度: {len(prompt)} 字符", flush=True)

            # 2. Sample functions from LLM
            valid_samples = self._sample_functions(prompt, iteration)

            if len(valid_samples) == 0:
                logger.warning("No valid samples generated, skipping iteration")
                continue

            # Save codes
            self.all_codes.append([s['code'] for s in valid_samples])

            # 3. Train DQN for each sample
            logger.info(f"开始训练 {len(valid_samples)} 个样本...")
            print(f"开始训练 {len(valid_samples)} 个样本...", flush=True)
            results = self._parallel_train(valid_samples, iteration)
            logger.info(f"训练完成，获得 {len(results)} 个结果")
            print(f"训练完成，获得 {len(results)} 个结果", flush=True)

            # 4. Analyze results
            logger.info("开始分析结果...")
            print("开始分析结果...", flush=True)
            analysis = self._analyze_results(valid_samples, results)
            logger.info(f"分析完成，共 {len(analysis)} 个样本分析结果")
            print(f"分析完成，共 {len(analysis)} 个样本分析结果", flush=True)

            # 5. Generate COT feedback
            logger.info("生成COT反馈...")
            print("生成COT反馈...", flush=True)
            cot_suggestion = self._generate_cot_feedback(
                valid_samples, results, analysis
            )
            logger.info("COT反馈生成完成")
            print("COT反馈生成完成", flush=True)
            self.all_iter_cot_suggestions.append(cot_suggestion)

            # 6. Save iteration results
            self._save_iteration_results(iteration, valid_samples, results, analysis)

            logger.info(f"Iteration {iteration} completed")

        # 7. Select best strategy
        best_strategy = self._select_best_strategy()

        return best_strategy

    def _sample_functions(self, prompt: str, iteration: int) -> List[Dict]:
        """Sample and validate functions from LLM."""
        valid_samples = []

        for sample_id in range(self.sample_count):
            logger.info(f"Sampling {sample_id + 1}/{self.sample_count}...")

            try:
                # Call LLM
                logger.info(f"样本 {sample_id + 1}/{self.sample_count}: 调用LLM...")
                print(f"样本 {sample_id + 1}/{self.sample_count}: 调用LLM...", flush=True)

                response = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a financial quantitative analysis expert."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=2000
                )

                logger.info(f"样本 {sample_id + 1}: LLM响应收到，长度: {len(response.choices[0].message.content)} 字符")
                print(f"样本 {sample_id + 1}: LLM响应收到", flush=True)

                code = response.choices[0].message.content

                # Extract Python code
                code = self._extract_python_code(code)

                # Save to file
                code_filename = f'it{iteration}_sample{sample_id}.py'
                code_path = os.path.join(self.output_dir, code_filename)
                with open(code_path, 'w') as f:
                    f.write(code)

                # Validate
                module = self._validate_code(code_path)

                if module is not None:
                    valid_samples.append({
                        'code': code,
                        'module': module,
                        'state_dim': module.state_dim,
                        'original_dim': 120
                    })
                    logger.info(f"  Sample {sample_id + 1} validated: state_dim={module.state_dim}")
                else:
                    logger.warning(f"  Sample {sample_id + 1} validation failed")

            except Exception as e:
                logger.error(f"  Sample {sample_id + 1} failed: {e}")
                continue

        return valid_samples

    def _extract_python_code(self, text: str) -> str:
        """Extract Python code from LLM output."""
        # Find import numpy statement
        if 'import numpy' in text:
            start = text.index('import numpy')
        elif 'import numpy as np' in text:
            start = text.index('import numpy as np')
        else:
            raise ValueError("No numpy import found in code")

        # Find the end of code (``` or end of last function)
        # First, try to find ```
        end_marker = text.find('```', start)
        if end_marker != -1 and end_marker > start:
            # Find the last return before ```
            code_section = text[start:end_marker]
            if 'return' in code_section:
                end = code_section.rindex('return')
                while end < len(code_section) and code_section[end] != '\n':
                    end += 1
                code = text[start:start + end]
            else:
                # No return found, take everything before ```
                code = text[start:end_marker]
        else:
            # No ```, use the old method
            end = text.rindex('return')
            while end < len(text) and text[end] != '\n':
                end += 1
            code = text[start:end + 1]

        return code

    def _validate_code(self, code_path: str):
        """Validate generated code by testing it."""
        try:
            # Create module name from path
            module_name = code_path.replace('/', '.').replace('.py', '')

            # Remove from sys.modules if already loaded
            if module_name in sys.modules:
                del sys.modules[module_name]

            # Import module
            spec = importlib.util.spec_from_file_location(module_name, code_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # Test functions
            test_state = np.zeros(120)
            enhanced = module.revise_state(test_state)
            intrinsic_r = module.intrinsic_reward(enhanced)

            # Validate
            assert enhanced.shape[0] >= 120, f"Output dimension must be >= 120, got {enhanced.shape[0]}"
            assert -100 <= intrinsic_r <= 100, f"intrinsic_reward must be in [-100, 100], got {intrinsic_r}"

            # Store state_dim for later use
            module.state_dim = enhanced.shape[0]

            return module

        except Exception as e:
            logger.error(f"Code validation failed: {e}")
            return None

    def _parallel_train(
        self,
        samples: List[Dict],
        iteration: int
    ) -> List[Dict]:
        """Train DQN for each sample."""
        results = []

        for i, sample in enumerate(samples):
            logger.info(f"\nTraining sample {i + 1}/{len(samples)}...")

            for ticker in self.tickers:
                try:
                    trainer = DQNTrainer(
                        ticker=ticker,
                        revise_state_func=sample['module'].revise_state,
                        intrinsic_reward_func=sample['module'].intrinsic_reward,
                        state_dim=sample['state_dim']
                    )

                    # Train
                    logger.info(f"  Training on {ticker}...")
                    trainer.train(
                        self.data_loader,
                        self.train_period[0],
                        self.train_period[1],
                        max_episodes=50
                    )

                    # Evaluate
                    logger.info(f"  Evaluating on {ticker}...")
                    val_metrics = trainer.evaluate(
                        self.data_loader,
                        self.val_period[0],
                        self.val_period[1]
                    )

                    results.append({
                        'sample_id': i,
                        'ticker': ticker,
                        'sharpe': val_metrics['sharpe'],
                        'max_dd': val_metrics['max_dd'],
                        'total_return': val_metrics['total_return'],
                        'trainer': trainer
                    })

                    logger.info(f"  Sharpe: {val_metrics['sharpe']:.3f}, MaxDD: {val_metrics['max_dd']:.2f}%")

                except Exception as e:
                    logger.error(f"  Training failed: {e}")
                    continue

        return results

    def _analyze_results(
        self,
        samples: List[Dict],
        results: List[Dict]
    ) -> List[Dict]:
        """Analyze training results."""
        analysis = []
        logger.info(f"开始分析结果，样本数: {len(samples)}, 结果数: {len(results)}")
        print(f"开始分析结果，样本数: {len(samples)}, 结果数: {len(results)}", flush=True)

        for i, sample in enumerate(samples):
            logger.info(f"分析样本 {i}/{len(samples)}")
            print(f"分析样本 {i}/{len(samples)}", flush=True)

            sample_results = [r for r in results if r['sample_id'] == i]

            if len(sample_results) == 0:
                logger.warning(f"样本 {i} 没有结果")
                continue

            # Collect all training data
            all_states = []
            all_rewards = []

            for result in sample_results:
                summary = result['trainer']._get_training_summary()
                all_states.extend(summary['states'])
                all_rewards.extend(summary['rewards'])

            logger.info(f"样本 {i}: 收集到 {len(all_states)} 个状态")
            print(f"样本 {i}: 收集到 {len(all_states)} 个状态", flush=True)

            # Feature analysis
            if len(all_states) > 0:
                logger.info(f"样本 {i}: 开始特征分析...")
                print(f"样本 {i}: 开始特征分析...", flush=True)
                importance, correlations, shap_values = analyze_features(
                    all_states,
                    all_rewards,
                    sample['original_dim']
                )
                logger.info(f"样本 {i}: 特征分析完成")
                print(f"样本 {i}: 特征分析完成", flush=True)

                analysis.append({
                    'sample_id': i,
                    'importance': importance,
                    'correlations': correlations,
                    'shap_values': shap_values
                })

        logger.info(f"所有样本分析完成，共 {len(analysis)} 个")
        print(f"所有样本分析完成，共 {len(analysis)} 个", flush=True)
        return analysis

    def _generate_cot_feedback(
        self,
        samples: List[Dict],
        results: List[Dict],
        analysis: List[Dict]
    ) -> str:
        """Generate COT feedback."""
        logger.info(f"开始生成COT反馈，样本数: {len(samples)}, 结果数: {len(results)}, 分析数: {len(analysis)}")
        print(f"开始生成COT反馈，样本数: {len(samples)}, 结果数: {len(results)}, 分析数: {len(analysis)}", flush=True)

        codes = [s['code'] for s in samples]
        scores = []

        for i, sample in enumerate(samples):
            logger.info(f"处理样本 {i} 的评分...")
            print(f"处理样本 {i} 的评分...", flush=True)
            sample_results = [r for r in results if r['sample_id'] == i]
            if len(sample_results) > 0:
                avg_sharpe = np.mean([r['sharpe'] for r in sample_results])
                avg_max_dd = np.mean([r['max_dd'] for r in sample_results])
                avg_return = np.mean([r['total_return'] for r in sample_results])
                scores.append({
                    'sharpe': avg_sharpe,
                    'max_dd': avg_max_dd,
                    'total_return': avg_return
                })
            else:
                scores.append({'sharpe': 0, 'max_dd': 100, 'total_return': 0})

        importance_list = [a['importance'] for a in analysis]
        correlations_list = [a['correlations'] for a in analysis]

        logger.info(f"准备调用get_financial_cot_prompt，参数: codes={len(codes)}, scores={len(scores)}, importance={len(importance_list)}, correlations={len(correlations_list)}")
        print(f"准备调用get_financial_cot_prompt，参数: codes={len(codes)}, scores={len(scores)}, importance={len(importance_list)}, correlations={len(correlations_list)}", flush=True)

        cot_prompt = get_financial_cot_prompt(
            codes, scores, importance_list, correlations_list, 120
        )

        logger.info(f"COT反馈生成完成，长度: {len(cot_prompt)} 字符")
        print(f"COT反馈生成完成，长度: {len(cot_prompt)} 字符", flush=True)

        return cot_prompt

    def _save_iteration_results(
        self,
        iteration: int,
        samples: List[Dict],
        results: List[Dict],
        analysis: List[Dict]
    ) -> None:
        """Save iteration results."""
        iteration_dir = os.path.join(self.output_dir, f'iteration_{iteration}')
        os.makedirs(iteration_dir, exist_ok=True)

        # 清理samples中的module对象以便pickle
        clean_samples = []
        for sample in samples:
            clean_sample = {
                'code': sample['code'],
                'state_dim': sample['state_dim'],
                'original_dim': sample['original_dim']
            }
            clean_samples.append(clean_sample)

        # 清理results中的trainer对象
        clean_results = []
        for result in results:
            clean_result = {
                'sample_id': result['sample_id'],
                'ticker': result['ticker'],
                'sharpe': result['sharpe'],
                'max_dd': result['max_dd'],
                'total_return': result['total_return']
            }
            clean_results.append(clean_result)

        # Save results
        with open(os.path.join(iteration_dir, 'results.pkl'), 'wb') as f:
            pickle.dump({
                'samples': clean_samples,
                'results': clean_results,
                'analysis': analysis
            }, f)

        # Save COT feedback
        if iteration < len(self.all_iter_cot_suggestions):
            with open(os.path.join(iteration_dir, 'cot_feedback.txt'), 'w') as f:
                f.write(self.all_iter_cot_suggestions[iteration])

    def _select_best_strategy(self) -> Dict:
        """Select best strategy from all iterations."""
        best_sharpe = -float('inf')
        best_config = None

        for iteration in range(self.max_iterations):
            iteration_dir = os.path.join(self.output_dir, f'iteration_{iteration}')
            result_file = os.path.join(iteration_dir, 'results.pkl')

            if os.path.exists(result_file):
                with open(result_file, 'rb') as f:
                    data = pickle.load(f)

                for result in data['results']:
                    if result['sharpe'] > best_sharpe:
                        best_sharpe = result['sharpe']
                        best_config = {
                            'iteration': iteration,
                            'sample_id': result['sample_id'],
                            'ticker': result['ticker'],
                            'sharpe': result['sharpe']
                        }

        if best_config is not None:
            logger.info(f"\nBest strategy: Iteration {best_config['iteration']}, "
                       f"Sample {best_config['sample_id']}, Sharpe = {best_config['sharpe']:.3f}")
        else:
            logger.warning("No valid strategy found")

        return best_config

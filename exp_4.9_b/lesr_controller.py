"""
LESR Controller Module for Exp4.9_b Financial Trading Experiment

Changes from exp4.7:
- A1: Generate ticker-specific prompts with stock profile
- Passes intrinsic_weight and commission to DQNTrainer
- state_dim from revise_state does NOT include position flag (added by DQNTrainer)
"""

import os
import sys
import importlib
import pickle
import numpy as np
from typing import List, Dict, Optional, Callable
from pathlib import Path
import logging
import torch
from torch.multiprocessing import Pool, set_start_method

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dqn_trainer import DQNTrainer
from feature_analyzer import analyze_features
from prompts import (
    get_initial_prompt,
    generate_stock_profile,
    get_financial_cot_prompt,
    get_iteration_prompt
)

logger = logging.getLogger(__name__)


def _train_ticker_worker(args):
    """
    Worker function for parallel ticker training.
    Runs in a separate process, each pinned to one GPU.
    """
    ticker, sample_code, state_dim, gpu_id, data_pkl_path, \
        train_start, train_end, val_start, val_end, sample_id, max_episodes, \
        intrinsic_weight, commission = args

    try:
        # Pin to specific GPU
        device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
        torch.cuda.set_device(gpu_id) if torch.cuda.is_available() else None

        # Import module from code string
        import importlib.util
        import tempfile
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
        tmp.write(sample_code)
        tmp.close()
        spec = importlib.util.spec_from_file_location(f'worker_{ticker}_{gpu_id}', tmp.name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        os.unlink(tmp.name)

        # Load data in this process
        sys.path.insert(0, str(Path(__file__).parent.parent / 'FINSABER'))
        from backtest.data_util.finmem_dataset import FinMemDataset
        from dqn_trainer import DQNTrainer

        data_loader = FinMemDataset(pickle_file=data_pkl_path)

        trainer = DQNTrainer(
            ticker=ticker,
            revise_state_func=module.revise_state,
            intrinsic_reward_func=module.intrinsic_reward,
            state_dim=state_dim,  # DQNTrainer adds +1 for position flag internally
            intrinsic_weight=intrinsic_weight,
            commission=commission,
            device=device
        )

        trainer.train(data_loader, train_start, train_end, max_episodes=max_episodes)
        val_metrics = trainer.evaluate(data_loader, val_start, val_end)

        return {
            'sample_id': sample_id,
            'ticker': ticker,
            'sharpe': val_metrics['sharpe'],
            'max_dd': val_metrics['max_dd'],
            'total_return': val_metrics['total_return'],
            'num_trades': val_metrics.get('num_trades', 0),
            'states': trainer.episode_states[:100],  # subsample for analysis
            'rewards': trainer.episode_rewards[:100],
            'dqn_state': trainer.get_state_dict_portable(),
            'state_dim': state_dim,
            'error': None
        }
    except Exception as e:
        return {
            'sample_id': sample_id,
            'ticker': ticker,
            'sharpe': 0.0,
            'max_dd': 100.0,
            'total_return': 0.0,
            'num_trades': 0,
            'states': [],
            'rewards': [],
            'dqn_state': None,
            'state_dim': state_dim,
            'error': str(e)
        }


class LESRController:
    """
    Main controller for LESR optimization in financial trading.

    Changes from exp4.7:
    - Generates per-ticker prompts with stock profiles (A1)
    - Passes intrinsic_weight and commission to trainers (C2, B2)
    """

    def __init__(self, config: Dict):
        """
        Initialize LESR controller.
        """
        self.tickers = config['tickers']
        self.train_period = config['train_period']
        self.val_period = config['val_period']
        self.test_period = config['test_period']
        self.data_loader = config['data_loader']
        self.sample_count = config.get('sample_count', 6)
        self.max_iterations = config.get('max_iterations', 3)
        self.init_min_valid = config.get('init_min_valid', 3)
        self.init_max_rounds = config.get('init_max_rounds', 5)

        # LLM configuration
        self.openai_key = config['openai_key']
        self.model = config.get('model', 'gpt-4')
        self.temperature = config.get('temperature', 0.7)
        self.base_url = config.get('base_url', None)

        # C2: intrinsic_weight configurable
        self.intrinsic_weight = config.get('intrinsic_weight', 0.1)

        # B2: commission rate
        self.commission = config.get('commission', 0.001)

        # Output directory
        self.output_dir = config.get('output_dir', 'exp_4.9_b/results')
        os.makedirs(self.output_dir, exist_ok=True)

        # Data path for workers to reload
        self.data_pkl_path = config.get('data_pkl_path', None)

        # A1: Pre-generate stock profiles for each ticker
        self.stock_profiles: Dict[str, str] = {}
        self._generate_stock_profiles()

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

    def _generate_stock_profiles(self):
        """A1: Generate stock profiles from training data."""
        logger.info("Generating stock profiles from training data...")
        print("Generating stock profiles from training data...", flush=True)

        for ticker in self.tickers:
            try:
                profile = generate_stock_profile(
                    ticker=ticker,
                    train_data_loader=self.data_loader,
                    train_start=self.train_period[0],
                    train_end=self.train_period[1]
                )
                self.stock_profiles[ticker] = profile
                logger.info(f"Profile generated for {ticker}")
                print(f"Profile generated for {ticker}:\n{profile}\n", flush=True)
            except Exception as e:
                logger.warning(f"Failed to generate profile for {ticker}: {e}")
                self.stock_profiles[ticker] = f"## Target Stock: {ticker}"

    def _get_prompt_for_iteration(self, iteration: int, ticker: str = None) -> str:
        """Generate prompt for given iteration and ticker."""
        if iteration == 0:
            # A1: Use ticker-specific prompt
            stock_profile = self.stock_profiles.get(ticker, "") if ticker else ""
            prompt = get_initial_prompt(
                ticker=ticker,
                stock_profile=stock_profile
            )
        else:
            stock_profile = self.stock_profiles.get(ticker, "") if ticker else ""
            prompt = get_iteration_prompt(
                self.all_codes,
                self.all_iter_cot_suggestions,
                ticker=ticker,
                stock_profile=stock_profile
            )
        return prompt

    def run_optimization(self) -> Dict:
        """
        Main optimization loop.
        """
        logger.info("=" * 50)
        logger.info("Starting LESR Optimization (exp_4.9_b)")
        logger.info(f"  intrinsic_weight: {self.intrinsic_weight}")
        logger.info(f"  commission: {self.commission}")
        logger.info(f"  tickers: {self.tickers}")
        logger.info("=" * 50)

        for iteration in range(self.max_iterations):
            logger.info(f"\n{'=' * 50}")
            logger.info(f"Iteration {iteration}")
            logger.info(f"{'=' * 50}")

            # 1. Generate prompt (use first ticker's profile for shared prompt)
            # For per-ticker training, each ticker gets its own prompt
            primary_ticker = self.tickers[0] if self.tickers else None
            prompt = self._get_prompt_for_iteration(iteration, primary_ticker)

            if iteration == 0:
                logger.info("使用初始Prompt（含个股信息）")
                print("使用初始Prompt（含个股信息）", flush=True)
            else:
                logger.info(f"生成迭代Prompt，历史代码数: {len(self.all_codes)}, COT反馈数: {len(self.all_iter_cot_suggestions)}")
                print(f"生成迭代Prompt，历史代码数: {len(self.all_codes)}, COT反馈数: {len(self.all_iter_cot_suggestions)}", flush=True)

            # 2. Sample functions from LLM
            if iteration == 0:
                valid_samples = self._sample_functions_init(prompt, iteration)
            else:
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

    def _sample_functions_init(self, prompt: str, iteration: int) -> List[Dict]:
        """Initialization sampling for Iteration 0."""
        all_valid = []
        sample_counter = 0

        for round_idx in range(self.init_max_rounds):
            round_valid = []

            for _ in range(self.sample_count):
                sample_counter += 1
                logger.info(f"[Init] Round {round_idx + 1}, Sample {sample_counter}: 调用LLM...")
                print(f"[Init] Round {round_idx + 1}, Sample {sample_counter}: 调用LLM...", flush=True)

                try:
                    response = self.openai_client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are a financial quantitative analysis expert."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=self.temperature,
                        max_tokens=3000  # Increased for more features
                    )

                    code = response.choices[0].message.content
                    code = self._extract_python_code(code)

                    code_filename = f'it{iteration}_sample{len(all_valid) + len(round_valid)}.py'
                    code_path = os.path.join(self.output_dir, code_filename)
                    with open(code_path, 'w') as f:
                        f.write(code)

                    module = self._validate_code(code_path)

                    if module is not None:
                        round_valid.append({
                            'code': code,
                            'module': module,
                            'state_dim': module.state_dim,
                            'original_dim': 120
                        })
                        logger.info(f"  [Init] Sample {sample_counter} validated: state_dim={module.state_dim}")
                    else:
                        logger.warning(f"  [Init] Sample {sample_counter} validation failed")

                except Exception as e:
                    logger.error(f"  [Init] Sample {sample_counter} failed: {e}")
                    continue

            all_valid.extend(round_valid)
            logger.info(f"[Init] Round {round_idx + 1} done: {len(round_valid)} valid this round, "
                        f"{len(all_valid)} total valid (need {self.init_min_valid})")
            print(f"[Init] Round {round_idx + 1}: {len(round_valid)} valid, "
                  f"total {len(all_valid)}/{self.init_min_valid}", flush=True)

            if len(all_valid) >= self.init_min_valid:
                logger.info(f"[Init] Sufficient valid samples: {len(all_valid)}")
                break

        if len(all_valid) < self.init_min_valid:
            logger.warning(f"[Init] Only {len(all_valid)} valid samples after {self.init_max_rounds} rounds "
                           f"(minimum {self.init_min_valid} required)")
        else:
            logger.info(f"[Init] Initialization complete with {len(all_valid)} valid samples")

        return all_valid

    def _sample_functions(self, prompt: str, iteration: int) -> List[Dict]:
        """Sample and validate functions from LLM."""
        valid_samples = []

        for sample_id in range(self.sample_count):
            logger.info(f"Sampling {sample_id + 1}/{self.sample_count}...")

            try:
                logger.info(f"样本 {sample_id + 1}/{self.sample_count}: 调用LLM...")
                print(f"样本 {sample_id + 1}/{self.sample_count}: 调用LLM...", flush=True)

                response = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a financial quantitative analysis expert."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=3000
                )

                logger.info(f"样本 {sample_id + 1}: LLM响应收到，长度: {len(response.choices[0].message.content)} 字符")
                print(f"样本 {sample_id + 1}: LLM响应收到", flush=True)

                code = response.choices[0].message.content
                code = self._extract_python_code(code)

                code_filename = f'it{iteration}_sample{sample_id}.py'
                code_path = os.path.join(self.output_dir, code_filename)
                with open(code_path, 'w') as f:
                    f.write(code)

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
        if 'import numpy as np' in text:
            start = text.index('import numpy as np')
        elif 'import numpy' in text:
            start = text.index('import numpy')
        else:
            raise ValueError("No numpy import found in code")

        # Find the end of code
        end_marker = text.find('```', start)
        if end_marker != -1 and end_marker > start:
            code_section = text[start:end_marker]
            if 'return' in code_section:
                end = code_section.rindex('return')
                while end < len(code_section) and code_section[end] != '\n':
                    end += 1
                code = text[start:start + end]
            else:
                code = text[start:end_marker]
        else:
            end = text.rindex('return')
            while end < len(text) and text[end] != '\n':
                end += 1
            code = text[start:end + 1]

        return code

    def _validate_code(self, code_path: str):
        """Validate generated code by testing it."""
        try:
            module_name = code_path.replace('/', '.').replace('.py', '')

            if module_name in sys.modules:
                del sys.modules[module_name]

            spec = importlib.util.spec_from_file_location(module_name, code_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # Test functions
            test_state = np.zeros(120)
            enhanced = module.revise_state(test_state)

            # Test intrinsic_reward with position flag appended
            state_with_pos = np.append(enhanced, 0.0)  # position flag = 0 (not holding)
            intrinsic_r = module.intrinsic_reward(state_with_pos)

            # Validate
            assert enhanced.shape[0] >= 120, f"Output dimension must be >= 120, got {enhanced.shape[0]}"
            assert -100 <= intrinsic_r <= 100, f"intrinsic_reward must be in [-100, 100], got {intrinsic_r}"

            # Store state_dim (without position flag, DQNTrainer adds it)
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
        num_gpus = torch.cuda.device_count()

        for i, sample in enumerate(samples):
            logger.info(f"\nTraining sample {i + 1}/{len(samples)}...")

            if num_gpus >= len(self.tickers):
                # Parallel: each ticker on its own GPU
                try:
                    set_start_method('spawn', force=True)
                except RuntimeError:
                    pass

                tasks = []
                for gpu_id, ticker in enumerate(self.tickers):
                    tasks.append((
                        ticker,
                        sample['code'],
                        sample['state_dim'],
                        gpu_id,
                        self.data_pkl_path,
                        self.train_period[0],
                        self.train_period[1],
                        self.val_period[0],
                        self.val_period[1],
                        i,
                        50,
                        self.intrinsic_weight,
                        self.commission
                    ))

                with Pool(processes=len(self.tickers)) as pool:
                    worker_results = pool.map(_train_ticker_worker, tasks)

                for wr in worker_results:
                    if wr['error']:
                        logger.error(f"  [{wr['ticker']}] Training failed: {wr['error']}")
                        continue

                    from dqn_trainer import DQNTrainer
                    trainer = DQNTrainer(
                        ticker=wr['ticker'],
                        revise_state_func=sample['module'].revise_state,
                        intrinsic_reward_func=sample['module'].intrinsic_reward,
                        state_dim=wr['state_dim'],
                        intrinsic_weight=self.intrinsic_weight,
                        commission=self.commission
                    )
                    if wr['dqn_state'] is not None:
                        trainer.load_state_dict_portable(wr['dqn_state'])
                    trainer.episode_states = wr['states']
                    trainer.episode_rewards = wr['rewards']

                    results.append({
                        'sample_id': i,
                        'ticker': wr['ticker'],
                        'sharpe': wr['sharpe'],
                        'max_dd': wr['max_dd'],
                        'total_return': wr['total_return'],
                        'num_trades': wr['num_trades'],
                        'trainer': trainer
                    })
                    logger.info(f"  [{wr['ticker']}] Sharpe: {wr['sharpe']:.3f}, MaxDD: {wr['max_dd']:.2f}%, Trades: {wr['num_trades']}")
            else:
                # Fallback: sequential
                for ticker in self.tickers:
                    try:
                        device = f'cuda:0' if num_gpus > 0 else 'cpu'
                        trainer = DQNTrainer(
                            ticker=ticker,
                            revise_state_func=sample['module'].revise_state,
                            intrinsic_reward_func=sample['module'].intrinsic_reward,
                            state_dim=sample['state_dim'],
                            intrinsic_weight=self.intrinsic_weight,
                            commission=self.commission,
                            device=device
                        )
                        trainer.train(self.data_loader, self.train_period[0],
                                     self.train_period[1], max_episodes=50)
                        val_metrics = trainer.evaluate(self.data_loader,
                                                      self.val_period[0],
                                                      self.val_period[1])
                        results.append({
                            'sample_id': i,
                            'ticker': ticker,
                            'sharpe': val_metrics['sharpe'],
                            'max_dd': val_metrics['max_dd'],
                            'total_return': val_metrics['total_return'],
                            'num_trades': val_metrics.get('num_trades', 0),
                            'trainer': trainer
                        })
                        logger.info(f"  [{ticker}] Sharpe: {val_metrics['sharpe']:.3f}, MaxDD: {val_metrics['max_dd']:.2f}%, Trades: {val_metrics.get('num_trades', 0)}")
                    except Exception as e:
                        logger.error(f"  [{ticker}] Training failed: {e}")
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
                avg_trades = np.mean([r.get('num_trades', 0) for r in sample_results])
                scores.append({
                    'sharpe': avg_sharpe,
                    'max_dd': avg_max_dd,
                    'total_return': avg_return,
                    'num_trades': int(avg_trades)
                })
            else:
                scores.append({'sharpe': 0, 'max_dd': 100, 'total_return': 0, 'num_trades': 0})

        importance_list = [a['importance'] for a in analysis]
        correlations_list = [a['correlations'] for a in analysis]

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

        clean_samples = []
        for sample in samples:
            clean_sample = {
                'code': sample['code'],
                'state_dim': sample['state_dim'],
                'original_dim': sample['original_dim']
            }
            clean_samples.append(clean_sample)

        clean_results = []
        for result in results:
            clean_result = {
                'sample_id': result['sample_id'],
                'ticker': result['ticker'],
                'sharpe': result['sharpe'],
                'max_dd': result['max_dd'],
                'total_return': result['total_return'],
                'num_trades': result.get('num_trades', 0)
            }
            clean_results.append(clean_result)

        with open(os.path.join(iteration_dir, 'results.pkl'), 'wb') as f:
            pickle.dump({
                'samples': clean_samples,
                'results': clean_results,
                'analysis': analysis
            }, f)

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

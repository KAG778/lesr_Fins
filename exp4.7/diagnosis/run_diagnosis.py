"""
CLI entry point for diagnosis experiments.

Usage:
    python exp4.7/diagnosis/run_diagnosis.py \
        --config exp4.7/config_W1.yaml \
        --num-runs 10 \
        --output-dir diagnosis_results/experiment_001 \
        --mode both \
        --max-parallel 3 \
        --base-seed 42

    # Analyze existing results only (no new runs):
    python exp4.7/diagnosis/run_diagnosis.py \
        --config exp4.7/config_W1.yaml \
        --analyze-only \
        --output-dir diagnosis_results/experiment_001
"""
import argparse
import logging
import sys
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='LESR Diagnosis Infrastructure')
    parser.add_argument('--config', required=True,
                        help='Base config YAML path (e.g. exp4.7/config_W1.yaml)')
    parser.add_argument('--num-runs', type=int, default=10,
                        help='Number of independent runs')
    parser.add_argument('--output-dir', default='diagnosis_results/experiment_001',
                        help='Output directory')
    parser.add_argument('--mode', choices=['lesr', 'baseline', 'both'], default='both')
    parser.add_argument('--max-parallel', type=int, default=3,
                        help='Max concurrent runs')
    parser.add_argument('--base-seed', type=int, default=42)
    parser.add_argument('--analyze-only', action='store_true',
                        help='Only analyze existing results, no new runs')
    parser.add_argument('--report-type',
                        choices=['statistical', 'variance', 'feature', 'all'],
                        default='all')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if args.analyze_only:
        from analyze_existing import analyze_experiment
        result = analyze_experiment(args.output_dir, report_type=args.report_type)
        print(result['report'] if isinstance(result, dict) else result)
        return

    from run_manager import RunManager
    manager = RunManager(
        base_config_path=args.config,
        output_root=args.output_dir,
        num_runs=args.num_runs,
        base_seed=args.base_seed,
        max_parallel=args.max_parallel,
    )

    logger.info(f"Launching {args.num_runs} runs with mode={args.mode}")
    result = manager.launch_runs(mode=args.mode)

    logger.info(f"Experiment complete. Results in: {result['experiment_dir']}")
    logger.info(f"Manifest: {result['manifest_path']}")

    # Auto-analyze after runs complete
    from analyze_existing import analyze_experiment
    analysis = analyze_experiment(args.output_dir, report_type=args.report_type)
    if isinstance(analysis, dict) and 'report' in analysis:
        report_path = Path(args.output_dir) / 'diagnosis_report.md'
        with open(report_path, 'w') as f:
            f.write(analysis['report'])
        logger.info(f"Report saved: {report_path}")


if __name__ == '__main__':
    main()

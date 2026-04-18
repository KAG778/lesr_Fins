"""
Code Sandbox for LLM-Generated Python Code

Validates LLM-generated revise_state, intrinsic_reward, and market_strategy code via:
1. AST whitelist check (blocks dangerous operations)
2. Test execution (verifies output types, NaN/Inf, dimension constraints)
3. Dimension detection (records extra dims added by revise_state)

代码沙箱模块 —— LLM 生成代码的安全验证。

三阶段验证流程：
1. AST 白名单检查：阻止危险操作（exec/eval/open 等）
2. 测试执行：用随机输入验证输出类型、NaN/Inf 检测、维度约束
3. 维度检测：记录 revise_state 添加的额外维度数

对应 LESR 论文中 "Code Validation" 步骤，确保 LLM 生成的代码可安全执行。
"""

import ast
import numpy as np
from typing import Callable, Dict, Optional


# Modules/functions allowed in LLM code
# LLM 生成的代码允许导入的模块白名单
_ALLOWED_IMPORTS = {'numpy', 'math', 'np', 'feature_library'}
_BLOCKED_BUILTINS = {
    'exec', 'eval', 'compile', 'open', '__import__',
    'globals', 'locals', 'vars', 'dir', 'getattr', 'setattr', 'delattr',
    'breakpoint', 'exit', 'quit',
}
_BLOCKED_ATTRS = {
    '__import__', '__builtins__', '__file__', '__name__',
}


def _ast_check(code_str: str) -> list:
    """Stage 1: AST whitelist validation. Returns list of error strings.

    阶段1：AST 白名单验证。检查代码中是否包含被禁止的操作或导入。
    阻止 exec、eval、open、文件操作等危险调用。
    """
    errors = []
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        return [f"Syntax error: {e}"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _BLOCKED_BUILTINS:
                errors.append(f"Blocked builtin: {func.id}()")
            if isinstance(func, ast.Attribute):
                if func.attr in _BLOCKED_ATTRS:
                    errors.append(f"Blocked attribute: .{func.attr}")

        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split('.')[0]
                if root not in _ALLOWED_IMPORTS:
                    errors.append(f"Blocked import: {alias.name}")

        if isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split('.')[0]
                if root not in _ALLOWED_IMPORTS:
                    errors.append(f"Blocked from-import: {node.module}")

    return errors


def _extract_functions(code_str: str) -> Dict[str, Callable]:
    """Extract revise_state, intrinsic_reward, and market_strategy from code string.

    从代码字符串中提取 revise_state、intrinsic_reward 和 market_strategy 函数。
    在受限命名空间中执行代码，只提供 numpy 和 feature_library。
    """
    namespace = {'np': np, 'numpy': np}
    try:
        import math
        namespace['math'] = math
    except ImportError:
        pass
    try:
        import feature_library
        namespace['feature_library'] = feature_library
    except ImportError:
        pass

    exec(compile(code_str, '<llm_code>', 'exec'), namespace)

    result = {}
    for name in ['revise_state', 'intrinsic_reward', 'market_strategy']:
        if name in namespace and callable(namespace[name]):
            result[name] = namespace[name]
    return result


def _test_execution(functions: Dict[str, Callable]) -> list:
    """Stage 2: Test execution with random inputs. Returns list of error strings.

    阶段2：测试执行。用随机 120 维输入验证：
    - revise_state 返回 ndarray 且维度 >= 120 且前 120 维不变
    - intrinsic_reward 返回标量且在 [-100, 100] 范围内
    - 输出不含 NaN 或 Inf
    """
    errors = []
    np.random.seed(42)
    test_state = np.random.randn(120) * 100 + 150

    if 'revise_state' in functions:
        try:
            revised = functions['revise_state'](test_state)
            if not isinstance(revised, np.ndarray):
                errors.append(f"revise_state must return np.ndarray, got {type(revised)}")
            elif revised.ndim != 1:
                errors.append(f"revise_state must return 1D array, got {revised.ndim}D")
            elif np.any(np.isnan(revised)):
                errors.append("revise_state output contains NaN")
            elif np.any(np.isinf(revised)):
                errors.append("revise_state output contains Inf")
            elif len(revised) < 120:
                errors.append(f"revise_state output too short: {len(revised)} < 120")
            elif not np.allclose(revised[:120], test_state, atol=1e-6):
                errors.append("revise_state must preserve first 120 dims (source state)")
        except Exception as e:
            errors.append(f"revise_state execution error: {e}")
    else:
        errors.append("revise_state function not found in code")

    if 'intrinsic_reward' in functions:
        try:
            test_input = functions.get('revise_state', lambda s: s)(test_state)
            reward_val = functions['intrinsic_reward'](test_input)
            if not isinstance(reward_val, (int, float, np.integer, np.floating)):
                errors.append(f"intrinsic_reward must return scalar, got {type(reward_val)}")
            elif np.isnan(float(reward_val)):
                errors.append("intrinsic_reward returned NaN")
            elif np.isinf(float(reward_val)):
                errors.append("intrinsic_reward returned Inf")
            elif abs(float(reward_val)) > 100:
                errors.append(f"intrinsic_reward out of range [-100, 100]: {reward_val}")
        except Exception as e:
            errors.append(f"intrinsic_reward execution error: {e}")
    else:
        errors.append("intrinsic_reward function not found in code")

    if 'market_strategy' in functions:
        try:
            test_market_state = np.random.randn(6)
            test_weights = np.random.dirichlet(np.ones(6))
            ms_val = functions['market_strategy'](test_market_state, test_weights)
            if not isinstance(ms_val, (int, float, np.integer, np.floating)):
                errors.append(f"market_strategy must return scalar, got {type(ms_val)}")
            elif np.isnan(float(ms_val)):
                errors.append("market_strategy returned NaN")
            elif np.isinf(float(ms_val)):
                errors.append("market_strategy returned Inf")
            elif float(ms_val) < 0.3 or float(ms_val) > 2.0:
                errors.append(f"market_strategy out of range [0.3, 2.0]: {ms_val}")
        except Exception as e:
            errors.append(f"market_strategy execution error: {e}")

    return errors


def validate(code_str: str) -> Dict:
    """Full validation pipeline for LLM-generated code.

    Returns:
        {
            'ok': bool,
            'errors': list[str],
            'revise_state': callable or None,
            'intrinsic_reward': callable or None,
            'market_strategy': callable or None,  # optional
            'feature_dim': int,   # 额外添加的特征维度数
            'state_dim': int,     # 总状态维度 (120 + feature_dim)
        }

    完整验证管线：AST 检查 → 函数提取 → 测试执行 → 维度检测。
    返回验证结果，包含可调用的函数引用和维度信息。
    """
    result = {
        'ok': False,
        'errors': [],
        'revise_state': None,
        'intrinsic_reward': None,
        'market_strategy': None,
        'feature_dim': 0,
        'state_dim': 120,
    }

    ast_errors = _ast_check(code_str)
    if ast_errors:
        result['errors'] = ast_errors
        return result

    try:
        functions = _extract_functions(code_str)
    except Exception as e:
        result['errors'] = [f"Code extraction error: {e}"]
        return result

    if 'revise_state' not in functions:
        result['errors'] = ["revise_state function not found"]
        return result

    test_errors = _test_execution(functions)
    if test_errors:
        result['errors'] = test_errors
        return result

    np.random.seed(42)
    test_state = np.random.randn(120) * 100 + 150
    try:
        revised = functions['revise_state'](test_state)
        state_dim = len(revised)
        feature_dim = state_dim - 120
    except Exception as e:
        result['errors'] = [f"Dimension detection error: {e}"]
        return result

    result['ok'] = True
    result['revise_state'] = functions['revise_state']
    result['intrinsic_reward'] = functions.get('intrinsic_reward')
    result['market_strategy'] = functions.get('market_strategy')
    result['feature_dim'] = feature_dim
    result['state_dim'] = state_dim
    return result

import sys
import logging
import signal
from pathlib import Path
from sympy import sympify

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.inference.answer_extractor import extract_sympy_answer
from src.utils.sympy_vocab import CUSTOM_NAMESPACE

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("evalf timed out")

logger = logging.getLogger(__name__)

_prompt_metadata = {}
_current_epoch = 0
_debug_logged = 0

def set_prompt_metadata(metadata_dict):
    global _prompt_metadata
    _prompt_metadata = metadata_dict


def correctness_reward_func(completions: list[str], prompts, **kwargs) -> list[float]:
    """Compute correctness rewards by verifying answers
    Args:
        completions: list of generated responses
        prompts: list of prompt strings
    Returns:
        list of rewards (0.0 to 1.0 with partial credit)
    """
    global _debug_logged
    _debug_logged = 0  # reset debug counter each batch
    rewards = []
    num_extracted = 0
    num_metadata = 0
    num_parsed = 0
    num_numerical = 0

    for completion, prompt in zip(completions, prompts):
        if isinstance(prompt, list) and len(prompt) > 0 and 'content' in prompt[0]:
            prompt_str = prompt[0]['content']
        else:
            prompt_str = prompt
        if isinstance(completion, list):
            completion = completion[0] if len(completion) > 0 else ""
        if not isinstance(completion, str):
            completion = str(completion)

        sympy_answer_str = extract_sympy_answer(completion)
        if not sympy_answer_str:
            rewards.append(0.0)
            continue
        num_extracted += 1

        prompt_data = _prompt_metadata.get(prompt_str, {})
        if not prompt_data:
            rewards.append(0.0)
            continue
        num_metadata += 1

        try:
            predicted_expr = sympify(sympy_answer_str, locals=CUSTOM_NAMESPACE)
        except Exception as e:
            rewards.append(0.0)
            continue
        num_parsed += 1

        reward = 0.0

        if prompt_data.get('has_numerical'):
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(3)
                predicted_num = float(predicted_expr.evalf())
                signal.alarm(0)
                expected_num = prompt_data['numerical_answer']
                error = abs(predicted_num - expected_num)

                if error >= 1e-4 and _debug_logged < 3:
                    print(f"[DEBUG MISMATCH] Expected: {expected_num:.6f}, Got: {predicted_num:.6f}, Error: {error:.2e}, SymPy: {sympy_answer_str}")
                    _debug_logged += 1

                if error < 1e-4:
                    reward = 1.0
                    num_numerical += 1
            except Exception as e:
                signal.alarm(0)
                if _debug_logged < 3:
                    print(f"[DEBUG EVAL ERROR] SymPy: {sympy_answer_str}, Error: {str(e)[:100]}")
                    _debug_logged += 1

        rewards.append(reward)

    total = len(completions)
    print(f"[VERIFY] {total} completions: extracted={num_extracted}, metadata={num_metadata}, parsed={num_parsed}, correct={num_numerical}")

    return rewards

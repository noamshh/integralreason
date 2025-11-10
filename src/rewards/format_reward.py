

def format_reward_func(completions: list[str], **kwargs) -> list[float]:
    rewards = []
    for completion in completions:
        if isinstance(completion, list):
            completion = completion[0] if len(completion) > 0 else ""
        if not isinstance(completion, str):
            completion = str(completion)

        reward = 0.0

        if '<think>' in completion and '</think>' in completion:
            # extract think content
            think_start = completion.find('<think>') + 7
            think_end = completion.find('</think>')
            think_content = completion[think_start:think_end].strip()

            # require minimum characters of 'reasoning'
            if len(think_content) >= 250:
                reward += 0.25

        # check final answer header
        if '## The final answer is:' in completion:
            reward += 0.25

        # check exactly two \boxed{}
        boxed_count = completion.count('\\boxed{')
        if boxed_count == 2:
            reward += 0.25
        elif boxed_count > 2:
            reward -= 0.4

        # check "sympy format" appears exactly once in last line
        sympy_count = completion.lower().count('sympy format')
        if sympy_count == 1:
            lines = completion.strip().split('\n')
            if lines and 'sympy format' in lines[-1].lower():
                reward += 0.25
        elif sympy_count > 1:
            reward -= 0.4

        reward = max(0.0, reward)
        rewards.append(reward)

    return rewards

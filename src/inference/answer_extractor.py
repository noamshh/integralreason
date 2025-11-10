import re
from typing import Optional
from dataclasses import dataclass


@dataclass
class ExtractedAnswer:
    latex: Optional[str]
    sympy: Optional[str]
    has_valid_format: bool
    format_details: dict


def _extract_from_braces(text: str, start_pos: int) -> Optional[str]:
    if start_pos >= len(text) or text[start_pos] != '{':
        return None
    depth = 0
    content = []
    for i in range(start_pos, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return ''.join(content[1:])
        if depth > 0:
            content.append(text[i])
    return None

def extract_boxed_answers(text: str) -> ExtractedAnswer:
    latex_answer = None
    sympy_answer = None

    latex_match = re.search(r'## The final answer is: \\boxed', text)
    if latex_match:
        brace_pos = latex_match.end()
        latex_answer = _extract_from_braces(text, brace_pos)

    sympy_match = re.search(r'## Sympy format: \\boxed', text)
    if sympy_match:
        brace_pos = sympy_match.end()
        sympy_answer = _extract_from_braces(text, brace_pos)

    # fallbacks
    if not latex_answer and not sympy_answer:
        boxed_matches = [(m.end(), m) for m in re.finditer(r'\\boxed', text)]
        if boxed_matches:
            last_pos, _ = boxed_matches[-1]
            latex_answer = _extract_from_braces(text, last_pos)
    if latex_answer and not sympy_answer:
        sympy_answer = latex_answer

    # check format
    format_details = {
        'has_think_tags': '<think>' in text and '</think>' in text,
        'has_final_answer_header': '## The final answer is:' in text,
        'has_latex_boxed': latex_answer is not None,
        'has_sympy_boxed': sympy_answer is not None,
        'has_both_formats': latex_answer is not None and sympy_answer is not None,
    }
    has_valid_format = (
        format_details['has_think_tags'] and
        format_details['has_final_answer_header'] and
        format_details['has_both_formats']
    )
    return ExtractedAnswer(
        latex=latex_answer,
        sympy=sympy_answer,
        has_valid_format=has_valid_format,
        format_details=format_details
    )


def extract_sympy_answer(text: str) -> Optional[str]:
    result = extract_boxed_answers(text)
    return result.sympy

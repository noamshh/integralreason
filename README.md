![IntegralReason Logo](logo.svg)

**Learning to solve integrals from how people actually explain them.**

IntegralReason teaches Qwen3-4B to solve definite integrals in two stages: fine-tuning on Math StackExchange answers, then reinforcement learning with rewards that check the answer.

## Key Idea

Good Math StackExchange answers already explain their reasoning. Wrapping them in `<think>` tags gives the model chain-of-thought examples without hand-written "Step 1, Step 2" annotations.

## Pipeline

1. **Collect:** 444 integral problems and answers from Math StackExchange (score above 4, with at least two reasoning words such as "therefore" or "substitute").
2. **Extract answers:** Claude Sonnet 4 pulls out each final answer in LaTeX and SymPy form.
3. **Fine-tune (SFT):** LoRA on Qwen3-4B, with answers in `\boxed{}`.
4. **Reinforce (RLVR):** GRPO on 78 verified prompts, rewarding format and correctness.
5. **Evaluate:** the first 100 problems of IntegralBench.

## Contents

- [1. Data Format](#1-data-format)
- [2. Supervised Fine-Tuning](#2-supervised-fine-tuning)
- [3. Reinforcement Learning with Verifiable Rewards](#3-reinforcement-learning-with-verifiable-rewards)
  - [What Went Wrong, and What I Learned](#what-went-wrong-and-what-i-learned)
- [4. Evaluation](#4-evaluation)
  - [4.1 IntegralBench](#41-integralbench)
  - [4.2 Results](#42-results)
  - [4.3 Example Predictions](#43-example-predictions)
  - [4.4 Future Work: Partial Credit](#44-future-work-partial-credit)

---

## 1. Data Format

Each training example uses Qwen3's chat template with `enable_thinking=True`:

```json
{
  "messages": [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": "Evaluate the integral: $\\int_{0}^{1} x^2 dx$"},
    {"role": "assistant", "content": "<think>\n[MSE solution with reasoning]\n</think>\n\n## The final answer is: $\\boxed{\\frac{1}{3}}$\n## SymPy format: \\boxed{1/3}"}
  ],
  "metadata": {
    "mse_answer_id": 123456,
    "extracted_answer_latex": "\\frac{1}{3}",
    "extracted_answer_sympy": "1/3",
    "numerical_answer": 0.333333
  }
}
```
The SymPy answer is what makes RL possible: it can be parsed, evaluated numerically and compared with the ground truth, which gives a cheap correctness reward.

---

## 2. Supervised Fine-Tuning

### Configuration

```yaml
model:
   base: "Qwen/Qwen3-4B"
   
lora:
   r: 64                  
   alpha: 16               
   dropout: 0.05
   target_modules:
     - q_proj
     - k_proj
     - v_proj
     - o_proj
     - gate_proj
     - up_proj
     - down_proj
     
training:
   batch_size: 2
   gradient_accumulation_steps: 8    
   learning_rate: 2.0e-4
   num_epochs: 4
   max_seq_length: 2048
   optimizer: adamw_8bit
   lr_scheduler: cosine
   warmup_steps: 50
```

### Example

**Input:**
```
Evaluate the integral: $\int_{0}^{1} x^2 dx$
```

**Target output:**
```
<think>
We evaluate the integral using the power rule:
$$\int x^2 dx = \frac{x^3}{3} + C$$

Applying the fundamental theorem of calculus:
$$\int_{0}^{1} x^2 dx = \left[\frac{x^3}{3}\right]_0^1 = \frac{1}{3} - 0 = \frac{1}{3}$$
</think>

## The final answer is: $\boxed{\frac{1}{3}}$
## SymPy format: \boxed{1/3}
```

---

## 3. Reinforcement Learning with Verifiable Rewards

### Rewards

**Format reward** (up to 1.0). Each item is worth +0.25:
- a `<think>` block of at least 250 characters
- a "## The final answer is:" header
- exactly two `\boxed{}` (−0.4 if there are more)
- "SymPy format" once, on the last line (−0.4 if repeated)

**Correctness reward** (0 or 1). 1 if the answer matches the ground truth to within 1e-4, otherwise 0.

The weights shift toward correctness as the model learns the format:

| Epoch | Format | Correctness |
|-------|--------|-------------|
| 0–1 | 0.3 | 0.7 |
| 1–2 | 0.2 | 0.8 |
| 2+ | 0.15 | 0.85 |

### Configuration

GRPO starts from the merged SFT checkpoint:

```yaml
model:
  base: models/sft/final/qwen3-4b-integral-merged  # SFT checkpoint
  max_seq_length: 2048
  load_in_4bit: true

lora:
  r: 32
  alpha: 32
  dropout: 0.05

training:
  method: GRPO
  batch_size: 1
  gradient_accumulation_steps: 16
  learning_rate: 1.0e-6
  num_epochs: 2
  num_generations: 8
```

### Prompts

`data/rlvr/prompts.jsonl` holds 78 verified integrals: 50 synthetic and 28 from Math StackExchange.

```json
{
  "prompt": [{"role": "user", "content": "Evaluate the integral: $\\int_{0}^{1} x^2 dx$"}],
  "expected_sympy": "1/3",
  "numerical_answer": 0.33333333,
  "has_numerical": true
}
```

### What Went Wrong, and What I Learned

RLVR took several attempts. The first runs crashed because the prompts weren't chat messages. Next, the default generation settings produced poor output. Then the correctness reward stayed at zero for over 40 steps because the prompt set was too noisy.

Lessons:
- GRPO expects prompts as chat messages, so check that first.
- When the correctness signal matters, train only on verified prompts.
- Penalize bad output instead of just withholding reward.
- Watch the KL divergence for signs of instability.

This was a personal project on rented A100s. Each RL run took hours, so a broad hyperparameter search wasn't practical, and the dataset sizes (444 for SFT, 78 for RL) reflect that budget.

### Next Steps

- **More SFT data:** [WebInstructSub](https://huggingface.co/datasets/TIGER-Lab/WebInstructSub) (2.34M Stack Exchange Q&A pairs, the same idea at scale) or [DeepMath-103K](https://github.com/zwhe99/DeepMath) (103K hard problems, three solutions each).
- **More RL prompts:** grow from 78 to 1,000+ and try curriculum learning.
- **Bigger model:** Qwen3 7B or 14B.
- **More tuning:** a proper hyperparameter search.

---

## 4. Evaluation

### 4.1 IntegralBench

[IntegralBench](https://github.com/IntegralBench/IntegralBench) ([paper](https://arxiv.org/abs/2507.21130v1)) has 317 definite integrals rated 1–5 for difficulty (mean 3.52). I used the first 100, sorted from easiest to hardest.

Each problem is sampled 3 times (temperature 0.6, top-p 0.95):
- **pass@3:** at least one sample is correct
- **all@3:** all three are correct
- **Consistency:** average share of correct samples per problem

I also report how often the answer could be parsed, format compliance, and generation time.

### 4.2 Results

| Model | pass@3 | all@3 | Consistency | Parsing Success | Format Compliance | Mean Gen Time (s) |
|-------|--------|-------|-------------|-----------------|-------------------|-------------------|
| SFT Baseline | 0.290 | 0.040 | 0.143 | 0.730 | 0.620 | 21.16 |
| RLVR Final | 0.310 | 0.040 | 0.163 | 0.730 | 0.663 | 27.05 |

By difficulty:

| Difficulty | RLVR pass@3 | RLVR all@3 | SFT pass@3 | SFT all@3 |
|-----------|-------------|-----------|-----------|-----------|
| 1.0 | 0.375 | 0.062 | 0.406 | 0.062 |
| 2.0 | 0.433 | 0.033 | 0.400 | 0.067 |
| 3.0 | 0.158 | 0.026 | 0.105 | 0.000 |

![Performance Metrics](performance_metrics.png)

RLVR raised format compliance from 62% to 66% and pass@3 from 29% to 31%, while parsing success stayed at 73%. The best results were on difficulty-2 problems (43% pass@3). With 100 problems, the accuracy gain is small enough to be noise.

### 4.3 Example Predictions

#### Correct Solution

**Problem:** $\int_0^1\sin^2(\log x)dx$ 
**Expected Answer:** $\frac{2}{5} = 0.4$

**Model Output:**
```
<think>
The integral can be solved by expanding the square and then integrating term by term.
$$ \int_0^1\sin^2(\log x)dx = \int_0^1\frac{1-\cos(2\log x)}2dx = \frac12\int_0^1dx-\frac12\int_0^1\cos(2\log x)dx $$
The first integral is trivial. The second one can be solved by using substitution $x=e^t$.
$$ \int_0^1\cos(2\log x)dx = \int_{-\infty}^0\cos(2t)e^tdt = \int_{0}^{\infty}\cos(2t)e^{-t}dt $$
This is a well-known Laplace transform, which is given by
$$ \int_0^{\infty}e^{-st}\cos(at)dt = \frac{s}{s^2+a^2} $$
In this case, $s=1$ and $a=2$, so the integral becomes
$$ \int_0^{\infty}e^{-t}\cos(2t)dt = \frac1{1+4} = \frac15 $$
Putting it all together,
$$ \int_0^1\sin^2(\log x)dx = \frac12\cdot1-\frac12\cdot\frac15 = \frac25 $$
</think>

## The final answer is: $\boxed{\frac{2}{5}}$
## SymPy format: \boxed{2/5}
```


#### Error 1: Algebra Slip in Partial Fractions

**Problem:** $\int_1^2\frac{9x+4}{x^5+3x^2+x}\text{ dx}$ 

**Expected Answer:** $1.25$

**Model Prediction:** $(1+\frac{5}{2}\sqrt{3})\ln 2 \approx 3.69$ 

The model chose the right method and set up the decomposition of $\frac{9x+4}{x(x^4+3x+1)}$, but solved for the coefficients $B, C, D, E$ incorrectly.

#### Error 2: Right Ingredients, Wrong Coefficients

**Problem:** $\int_0^{2\pi}\left|1+2\sin x\right|dx$ 

**Expected Answer:** $\frac{2\pi}{3} + 4\sqrt{3} \approx 9.02$

**Model Prediction:** $-\frac{5\pi}{6} + \sqrt{3} \approx -0.89$

The model found the sign changes at $x = \frac{7\pi}{6}$ and $\frac{11\pi}{6}$ and split the integral into three pieces, but slipped in the algebra.

Notice that both answers combine the same constants, $\pi$ and $\sqrt{3}$. A numerical check can't see that the model was close.

### 4.4 Future Work: Partial Credit

Right now any answer outside the 1e-4 tolerance counts as simply wrong. Error 2 shows that some wrong answers are much closer than others. Two ways to measure that:
- **Symbol counts:** count constants such as $\pi$, $e$ and $\sqrt{2}$ in each answer and compare with cosine similarity.
- **Embeddings:** embed answers with a math-expression encoder (like the E-Gen embeddings in [IntegralIndx](https://github.com/noamshh/integralindx)) and compare them.

Partial credit would give RLVR a richer signal for answers that have the right idea but the wrong algebra.

---
**Project by Noam Shalev**
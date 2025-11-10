import sys
import torch
from pathlib import Path
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from omegaconf import OmegaConf


project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.rewards.format_reward import format_reward_func
from src.rewards.correctness_reward import correctness_reward_func, set_prompt_metadata


_sample_logged = {}

def weighted_format_reward(completions, trainer_state=None, **kwargs):
    epoch = trainer_state.epoch if trainer_state else 0
    if epoch < 1.0:
        weight = 0.3
    elif epoch < 2.0:
        weight = 0.2
    else:
        weight = 0.15
    rewards = format_reward_func(completions, **kwargs)
    return [r * weight for r in rewards]

def weighted_correctness_reward(completions, prompts, trainer_state=None, **kwargs):
    epoch = trainer_state.epoch if trainer_state else 0
    if epoch < 1.0:
        weight = 0.7
    elif epoch < 2.0:
        weight = 0.8
    else:
        weight = 0.85
    step = trainer_state.global_step if trainer_state else 0

    if step not in _sample_logged and step % 5 == 0:
        prompt_sample = prompts[0]
        if isinstance(prompt_sample, list) and len(prompt_sample) > 0 and 'content' in prompt_sample[0]:
            prompt_text = prompt_sample[0]['content']
        else:
            prompt_text = prompt_sample

        completion_sample = completions[0]
        if isinstance(completion_sample, list):
            completion_sample = completion_sample[0] if len(completion_sample) > 0 else ""

        print(f"\n{'='*80}\n[SAMPLE at step {step}]\n{'='*80}")
        print(f"PROMPT: {prompt_text}")
        print(f"\nCOMPLETION:\n{completion_sample}")
        print(f"{'='*80}\n")
        _sample_logged[step] = True

    rewards = correctness_reward_func(completions, prompts, **kwargs)
    return [r * weight for r in rewards]

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params:,} || "
        f"all params: {all_param:,} || "
        f"trainable%: {100 * trainable_params / all_param:.2f}%"
    )


def main():
    config_path = project_root / "config" / "rlvr_config.yaml"
    cfg = OmegaConf.load(config_path)

    output_dir = project_root / cfg.paths.output_dir
    checkpoint_path = None
    if output_dir.exists():
        checkpoints = sorted(output_dir.glob("checkpoint-*"))
        if checkpoints:
            checkpoint_path = str(checkpoints[-1])

    print(f"base model (from SFT): {cfg.model.name}")
    print(f"prompts dataset: {cfg.paths.prompts_dataset}")
    print(f"training epochs: {cfg.training.num_epochs}")
    print(f"num generations: {cfg.training.num_generations}")
    print(f"learning rate: {cfg.training.learning_rate}")
    print(f"resume from checkpoint: {checkpoint_path if checkpoint_path else 'no'}\n")

    if checkpoint_path:
        print(f"loading model from checkpoint: {checkpoint_path}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=checkpoint_path,
            max_seq_length=cfg.model.max_seq_length,
            dtype=None,
            load_in_4bit=cfg.model.load_in_4bit,
        )
    else:
        print("loading SFT model...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(project_root / cfg.model.name),
            max_seq_length=cfg.model.max_seq_length,
            dtype=None,
            load_in_4bit=cfg.model.load_in_4bit,
        )

        print("\nAdding lora adapters for RLVR...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg.lora.r,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            target_modules=cfg.lora.target_modules,
            use_gradient_checkpointing="unsloth",
        )

    print_trainable_parameters(model)
    dataset = load_dataset('json', data_files=str(project_root / cfg.paths.prompts_dataset), split='train')
    print(f"loaded {len(dataset)} prompts")

    prompt_metadata = {}
    for row in dataset:
        prompt_text = row['prompt'][0]['content'] if isinstance(row['prompt'], list) else row['prompt']
        prompt_metadata[prompt_text] = row
    set_prompt_metadata(prompt_metadata)

    training_args = GRPOConfig(
        output_dir=str(project_root / cfg.paths.output_dir),
        run_name=f'RLVR-Integral-{cfg.model.name.split("/")[-1]}',
        learning_rate=cfg.training.learning_rate,
        per_device_train_batch_size=cfg.training.batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        num_train_epochs=cfg.training.num_epochs,
        num_generations=cfg.training.num_generations,
        max_prompt_length=cfg.training.max_prompt_length,
        max_completion_length=cfg.training.max_completion_length,
        # temperature=cfg.training.temperature,
        # top_p=cfg.training.top_p,
        shuffle_dataset=False,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=cfg.logging.steps,
        save_steps=cfg.training.save_steps,
        save_total_limit=cfg.logging.save_total_limit,
        max_grad_norm=cfg.training.max_grad_norm,
        report_to=cfg.logging.report_to,
        logging_dir=str(project_root / cfg.logging.logging_dir),
    )

    reward_funcs = [weighted_format_reward, weighted_correctness_reward]

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
    )

    print("Starting RLVR Training")
    if checkpoint_path:
        trainer.train(resume_from_checkpoint=checkpoint_path)
    else:
        trainer.train()

    print("\nSaving final model...")
    final_path = project_root / cfg.paths.final_model_dir
    final_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    print(f"\nmodel saved to: {final_path}")
    print("Training Complete!")


if __name__ == "__main__":
    main()

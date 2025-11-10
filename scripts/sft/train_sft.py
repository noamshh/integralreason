import json
import torch
from pathlib import Path
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback
from omegaconf import OmegaConf
import time

project_root = Path(__file__).parent.parent
config_path = project_root / "config" / "sft_config.yaml"
cfg = OmegaConf.load(config_path)


def load_dataset_from_jsonl(jsonl_path: Path):
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            example = json.loads(line)
            data.append(example)
    return Dataset.from_list(data)


def formatting_prompts_func(examples):
    messages_list = examples['messages']
    if messages_list and isinstance(messages_list[0], dict):
        messages_list = [messages_list]
    texts = []
    for messages in messages_list:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=True)
        texts.append(text)
    return texts


class SampleGenerationCallback(TrainerCallback):
    def __init__(self, model, tokenizer, test_problems_path, project_root):
        self.model = model
        self.tokenizer = tokenizer
        self.project_root = project_root
        with open(test_problems_path, 'r', encoding='utf-8') as f:
            self.test_problems = json.load(f)
        print(f"loaded {len(self.test_problems)} test problems for evaluation")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        print(f"\n{'='*60}")
        print(f"generating sample solutions (step {state.global_step}, epoch {state.epoch:.2f})...")
        print(f"{'='*60}")
        FastLanguageModel.for_inference(self.model)
        for i, problem in enumerate(self.test_problems, 1):
            messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": f"Evaluate the integral: {problem['latex']}"}
            ]
            inputs = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                        enable_thinking=True, return_tensors="pt").to("cuda")
            start_time = time.time()
            outputs = self.model.generate(inputs, max_new_tokens=512, temperature=0.7, do_sample=True, top_p=0.9)
            gen_time = time.time() - start_time
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

            if "<|im_start|>assistant" in response:
                assistant_response = response.split("<|im_start|>assistant")[-1]
                assistant_response = assistant_response.replace("<|im_end|>", "").strip()
            else:
                assistant_response = response

            # print sample
            print(f"\nProblem {i}: {problem['latex']}")
            print(f"Expected: {problem['expected_answer']}")
            print(f"Generated ({gen_time:.2f}s):")
            print(f"{assistant_response[:300]}..." if len(assistant_response) > 300 else assistant_response)
            print(f"{'-'*60}")

        self.model.train()
        print(f"{'='*60}\n")


if __name__ == "__main__":
    print(f"\n1. Loading {cfg.model.name} model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model.name,
        max_seq_length=cfg.model.max_seq_length,
        dtype=None,
        load_in_4bit=cfg.model.load_in_4bit,
    )
    print("\n2. Adding lora adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
        use_gradient_checkpointing="unsloth",
        random_state=cfg.training.seed,
    )
    print("\n3. Loading datasets...")
    train_path = project_root / "data" / "splits" / "train.jsonl"
    train_dataset = load_dataset_from_jsonl(train_path)
    print(f"train: {len(train_dataset)} examples")
    val_path = project_root / "data" / "splits" / "val.jsonl"
    val_dataset = load_dataset_from_jsonl(val_path)
    print(f"validation: {len(val_dataset)} examples")

    print("\n4. Setting up training...")
    training_args = TrainingArguments(
        output_dir=str(project_root / cfg.paths.output_dir),
        per_device_train_batch_size=cfg.training.batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        warmup_steps=cfg.training.warmup_steps,
        num_train_epochs=cfg.training.num_epochs,
        learning_rate=cfg.training.learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=cfg.logging.steps,
        save_total_limit=cfg.logging.save_total_limit,
        optim=cfg.optimizer.type,
        weight_decay=cfg.optimizer.weight_decay,
        lr_scheduler_type=cfg.optimizer.lr_scheduler_type,
        seed=cfg.training.seed,

        # evaluation
        eval_strategy=cfg.training.eval_strategy,
        eval_steps=cfg.training.eval_steps,
        per_device_eval_batch_size=cfg.training.batch_size,
        save_strategy=cfg.training.save_strategy,
        save_steps=cfg.training.save_steps,
        load_best_model_at_end=cfg.training.load_best_model_at_end,
        metric_for_best_model=cfg.training.metric_for_best_model,
        greater_is_better=cfg.training.greater_is_better,

        # logging
        report_to=cfg.logging.report_to,
        logging_dir=str(project_root / cfg.logging.logging_dir),
    )

    test_problems_path = project_root / "scripts" / "test_problems.json"
    sample_callback = SampleGenerationCallback(
        model=model,
        tokenizer=tokenizer,
        test_problems_path=test_problems_path,
        project_root=project_root
    )

    # monkey-patched by unsloth
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        max_seq_length=cfg.model.max_seq_length,
        formatting_func=formatting_prompts_func,
        args=training_args,
        callbacks=[sample_callback],
    )

    print("\n5. Starting training...")
    effective_batch = cfg.training.batch_size * cfg.training.gradient_accumulation_steps
    total_steps = len(train_dataset) * cfg.training.num_epochs // effective_batch
    eval_per_epoch = (len(train_dataset) // effective_batch) // cfg.training.eval_steps
    print(f"effective batch size: {effective_batch}")
    print(f"total training steps: ~{total_steps}")
    print(f"validation every {cfg.training.eval_steps} steps (~{eval_per_epoch} times per epoch)")
    print(f"learning rate: {cfg.training.learning_rate}")
    print(f"lorA config: r={cfg.lora.r}, alpha={cfg.lora.alpha}, dropout={cfg.lora.dropout}")
    print(f"tensorboard logs: {training_args.logging_dir}")
    trainer.train()

    print("\n6. Saving final model...")
    output_dir = project_root / cfg.paths.final_model_dir
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"\nmodel saved to: {output_dir}")
    print("training complete!")

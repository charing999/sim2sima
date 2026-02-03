import os
import torch
import json
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer, DPOConfig

# Windows Environment Patch
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def train_dpo():
    # 0. Configuration
    base_model_id = "google/gemma-3-4b-it"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    adapter_path = os.path.join(script_dir, "DPO_drone", "gemma3-drone-web-sft")
    data_path = "dpo_preference_data_v2.jsonl"
    output_dir = "gemma3-drone-dpo"
    
    if not os.path.exists(data_path):
        print(f"[Error] Data file not found: {data_path}")
        print("Run sima_app.py first to collect preference data.")
        return

    print("Loading Base Model & SFT Adapter...")
    
    # 1. Load SFT Model (Base + Adapter)
    # Note: We use merge_and_unload to create a solid base for DPO
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16, # Consistent with user env
        device_map="auto"
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    # 핵심: SFT 학습된 가중치를 고정(Freeze)하고, 그 위에 DPO LoRA를 얹기 위해 병합(Merge)합니다.
    model = model.merge_and_unload()
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    print("SFT Model Merged. Preparing DPO...")

    # 2. Prepare Dataset
    dataset = load_dataset("json", data_files=data_path)
    # Using 'train' split by default
    train_dataset = dataset["train"]
    print(f"Loaded {len(train_dataset)} preference pairs.")

    # 3. Configure DPO LoRA
    # We train a NEW adapter on top of the Merged SFT model
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 4. Training Arguments
    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-6, # Low LR for DPO
        bf16=False,
        fp16=True,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        max_length=1024,
        max_prompt_length=512,
        remove_unused_columns=False,
        report_to="none"
    )

    # 5. Initialize Trainer
    # ref_model=None implies using the model itself (copy) or PEFT implicit ref.
    # When peft_config is passed, DPOTrainer handles reference efficiently.
    trainer = DPOTrainer(
        model=model,
        ref_model=None, 
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    print("Starting DPO Training...")
    trainer.train()
    
    print(f"Training Complete. Saving to {output_dir}")
    trainer.save_model(output_dir)

if __name__ == "__main__":
    train_dpo()

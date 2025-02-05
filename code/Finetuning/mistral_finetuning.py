import json
import torch
import time
import os
from torch.utils.data import Dataset, random_split
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


class MedicalQADataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = f"[INST] Context: {item['context']}\nQuestion: {item['question']} [/INST]\nAnswer: {item['answer']}"
        encoding = self.tokenizer(prompt, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        encoding['labels'] = encoding['input_ids'].clone()
        return {key: val.squeeze(0) for key, val in encoding.items()}


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def format_time(seconds):
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"


def load_model_and_tokenizer(model_path: str, gpu_id: int):
    print(f"loading... path: {model_path}")
    model_load_start = time.time()

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"path is not found: {model_path}")

    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
    else:
        device = torch.device('cpu')
    
    print(f"device: {device}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"loading and error: {e}")
        return None, None

    try:        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map={"": device}
        )
    except Exception as e:
        print(f"model loading and error: {e}")
        return None, None

    print(f"loading complete. time {format_time(time.time() - model_load_start)}")
    return model, tokenizer

def main():
    start_time = time.time() 
    
    model_path = "/mnt/nvme01/huggingface/models/MistralAI/Mistral-7B-Instruct-v0.3/"
    gpu_id = 2  

    model, tokenizer = load_model_and_tokenizer(model_path, gpu_id)
    if model is None or tokenizer is None:
        print("error.")
        return

    print(f"GPU {gpu_id} memory use: {torch.cuda.memory_allocated(gpu_id) / 1024**3:.2f} GB")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    print("model loading...")
    model_prep_start = time.time()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    print(f"loading complete. time: {format_time(time.time() - model_prep_start)}")

    print("data loading...")
    data_prep_start = time.time()    
    data = load_data('finetuning_IDQuAD.json') 
    full_dataset = MedicalQADataset(data, tokenizer, max_length=512)
  
    train_size = int(0.9 * len(full_dataset))
    eval_size = len(full_dataset) - train_size
    train_dataset, eval_dataset = random_split(full_dataset, [train_size, eval_size])
    
    print(f"data complete. time: {format_time(time.time() - data_prep_start)}")
    print(f"train data: {len(train_dataset)}, test data: {len(eval_dataset)}")
    
    training_args = TrainingArguments(
        output_dir="./mistral-7b-finetuned",
        num_train_epochs=5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=32,
        save_steps=50,
        save_total_limit=3,
        logging_dir='./logs',
        logging_steps=10,
        fp16=True,
        learning_rate=2e-5,
        warmup_steps=100,
        optim="adamw_torch",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=0.3,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        evaluation_strategy="steps",
        eval_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print("training...")
    train_start = time.time()
    trainer.train()
    train_duration = time.time() - train_start
    print(f"training is complete. time: {format_time(train_duration)}")

    print("saving model...")
    save_start = time.time()
    model.save_pretrained("./mistral-7b-finetuned-final")
    tokenizer.save_pretrained("./mistral-7b-finetuned-final")
    print(f"saving comeplete. time: {format_time(time.time() - save_start)}")

    total_duration = time.time() - start_time
    print(f"done. total time: {format_time(total_duration)}")

if __name__ == "__main__":
    main()
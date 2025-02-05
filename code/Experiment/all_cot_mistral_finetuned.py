import json
import torch
from tqdm import tqdm
import numpy as np
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.cuda.set_device(5)  
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

model_path = "/home/soonchan13/server/mistral-7b-finetuned-final"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="cuda:5")

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def simple_tokenize(text):
    return tokenizer.tokenize(text.lower())

def generate_answer(prompt, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):]

def extract_final_answer(text):
    answer_pattern = r"(?:The answer is:|Therefore, the answer is:)\s*(.*)"
    match = re.search(answer_pattern, text, re.IGNORECASE | re.DOTALL)
    
    if match:        
        final_answer = match.group(1).strip()
        final_answer = re.sub(r'\.$', '', final_answer) 
        return final_answer
    else:
        return text.strip()

def calculate_f1(prediction, ground_truth):
    prediction_tokens = set(simple_tokenize(prediction))
    ground_truth_tokens = set(simple_tokenize(ground_truth))
    
    common = prediction_tokens & ground_truth_tokens
    precision = len(common) / len(prediction_tokens) if prediction_tokens else 0
    recall = len(common) / len(ground_truth_tokens) if ground_truth_tokens else 0
    
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

def calculate_em(prediction, ground_truth):
    return int(prediction.lower().strip() == ground_truth.lower().strip())

def evaluate_qa(data, shot_type):
    f1_scores = []
    em_scores = []

    if shot_type == "zero-shot":
        generate_func = generate_answer_zeroshot_cot
    elif shot_type == "1-shot":
        generate_func = generate_answer_oneshot_cot
    elif shot_type == "5-shot":
        generate_func = generate_answer_fiveshot_cot
    else:
        raise ValueError("Invalid shot_type. Choose 'zero-shot', '1-shot', or '5-shot'.")

    for item in tqdm(data, desc=f"Evaluating {shot_type} CoT"):
        context = item['context']
        question = item['question']
        ground_truth = item['answer']

        prediction = generate_func(context, question)

        final_answer = extract_final_answer(prediction)
        
        f1_scores.append(calculate_f1(final_answer, ground_truth))
        em_scores.append(calculate_em(final_answer, ground_truth))

    avg_f1 = np.mean(f1_scores)
    avg_em = np.mean(em_scores)

    return {
        'F1 Score': avg_f1,
        'Exact Match Score': avg_em
    }

def run_experiments(data, shot_type):
    f1_scores = []
    em_scores = []
    
    for i in range(3):
        print(f"\nRunning {shot_type} CoT experiment {i+1}/3:")
        results = evaluate_qa(data, shot_type)
        f1_scores.append(results['F1 Score'])
        em_scores.append(results['Exact Match Score'])
        
        print(f"Experiment {i+1} Results:")
        for metric, score in results.items():
            print(f"{metric}: {score:.4f}")
    
    avg_f1 = np.mean(f1_scores)
    avg_em = np.mean(em_scores)
    
    return {
        'Average F1 Score': avg_f1,
        'Average Exact Match Score': avg_em
    }

if __name__ == "__main__":
    data = load_data('test_IDQuAD.json')
    
    for shot_type in ["zero-shot", "1-shot", "5-shot"]:
        print(f"\n{'='*50}")
        print(f"Evaluating {shot_type} CoT with Mistral 7B (3 runs):")
        results = run_experiments(data, shot_type)
        print(f"\n{shot_type} CoT Final Results (Average of 3 runs):")
        for metric, score in results.items():
            print(f"{metric}: {score:.4f}")
        print(f"{'='*50}")
import json
import torch
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.cuda.set_device(1)  
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

model_path = "/home/soonchan13/server/mistral-7b-finetuned-final"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="cuda:1")

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

def generate_answer_zeroshot(context, question):
    prompt = f"""You are a helpful assistant. Answer the question based on the given context.

Context: {context}

Question: {question}

Answer: """
    return generate_answer(prompt)

def calculate_f1(prediction, ground_truth):
    prediction_tokens = set(simple_tokenize(prediction))
    ground_truth_tokens = set(simple_tokenize(ground_truth))
    
    common = prediction_tokens & ground_truth_tokens
    precision = len(common) / len(prediction_tokens) if prediction_tokens else 0
    recall = len(common) / len(ground_truth_tokens) if ground_truth_tokens else 0
    
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

def calculate_em(prediction, ground_truth):
    return int(prediction.lower().strip() == ground_truth.lower().strip())

def evaluate_qa_single(data):
    f1_scores = []
    em_scores = []

    for item in tqdm(data, desc="Evaluating Zero-shot"):
        context = item['context']
        question = item['question']
        ground_truth = item['answer']

        prediction = generate_answer_zeroshot(context, question)
        
        f1_scores.append(calculate_f1(prediction, ground_truth))
        em_scores.append(calculate_em(prediction, ground_truth))

    return np.mean(f1_scores), np.mean(em_scores)

def evaluate_qa_three_runs(data):
    f1_scores = []
    em_scores = []

    for i in range(3):
        print(f"Run {i+1}/3")
        f1, em = evaluate_qa_single(data)
        f1_scores.append(f1)
        em_scores.append(em)
        print(f"Run {i+1} Results - F1 Score: {f1:.4f}, Exact Match Score: {em:.4f}")

    avg_f1 = np.mean(f1_scores)
    avg_em = np.mean(em_scores)

    return {
        'Average F1 Score': avg_f1,
        'Average Exact Match Score': avg_em
    }

if __name__ == "__main__":
    data = load_data('test_IDQuAD.json')
    
    print("\nEvaluating Zero-shot with Mistral 7B Finetuned:")
    results = evaluate_qa_three_runs(data)
    print("\nFinal Results (Average of 3 runs):")
    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")
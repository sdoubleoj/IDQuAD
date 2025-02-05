import json
import torch
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.cuda.set_device(2) 
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

model_path = "/home/soonchan13/server/mistral-7b-finetuned-final/"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="cuda:2")

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

def generate_answer_oneshot(context, question):
    example = {
        "context": "Zoonotic diseases are a significant global health issue often leading to spillover infections where pathogens are transmitted from vertebrate animals to humans.",
        "question": "What type of diseases present a significant global health burden?",
        "answer": "Zoonotic diseases"
    }
    
    prompt = f"""You are a helpful assistant. Answer the question based on the given context.

Here's an example:
Context: {example['context']}
Question: {example['question']}
Answer: {example['answer']}

Now, please answer the following question based on the given context:
Context: {context}
Question: {question}
Answer: """
    return generate_answer(prompt)

def generate_answer_fiveshot(context, question):
    examples = [
        {
            "context": "Zoonotic diseases are a significant global health issue often leading to spillover infections where pathogens are transmitted from vertebrate animals to humans.",
            "question": "What type of diseases present a significant global health burden?",
            "answer": "Zoonotic diseases"
        },
        {
            "context": "The introduction of a novel influenza A virus with a new hemagglutinin into a naive human population could lead to significant disease and mortality as seen in Hong Kong in 1997 when H5N1 was transmitted from infected poultry to humans. Continued surveillance is necessary in southern China where H5N1 viruses are still being isolated from domestic poultry. A recent study focused on the molecular characterization of an H5N1 virus isolated from duck meat imported from China to South Korea. This particular virus closely related to the Hong Kong/97 H5N1 demonstrated high pathogenicity in chickens and mice resulting in high mortality in those species while ducks showed no clinical signs of disease.",
            "question": "Where was the H5N1 virus isolated from duck meat imported from China?",
            "answer": "South Korea"
        },
        {
            "context": "The introduction of a novel influenza A virus with a new hemagglutinin into a naive human population could lead to significant disease and mortality as seen in Hong Kong in 1997 when H5N1 was transmitted from infected poultry to humans. Continued surveillance is necessary in southern China where H5N1 viruses are still being isolated from domestic poultry. A recent study focused on the molecular characterization of an H5N1 virus isolated from duck meat imported from China to South Korea. This particular virus closely related to the Hong Kong/97 H5N1 demonstrated high pathogenicity in chickens and mice resulting in high mortality in those species while ducks showed no clinical signs of disease.",
            "question": "In which species was the pathogenesis of the H5N1 virus characterized?",
            "answer": "Chickens and mice"
        },
        {
            "context": "Novel avian-origin influenza A viruses were first reported to infect humans in March 2013 with a total of 143 human cases and 45 deaths recorded to date. Researchers utilized sequence comparisons and phylogenetic analyses to identify distinct amino acid changes in the polymerase PA protein of these viruses. Mutant viruses with specific amino acid alterations were tested for their polymerase activities and growth in mammalian versus avian cells as well as for their virulence in mice. The study found that certain mutants were more virulent in mice compared to the wild-type A virus indicating that the PA protein has amino acid substitutions that could affect virulence in mammals.",
            "question": "When were novel avian-origin influenza A viruses first reported to infect humans?",
            "answer": "March 2013"
        },
        {
            "context": "Clostridium difficile and methicillin-resistant Staphylococcus aureus (MRSA) are significant pathogens of concern in food animals specifically in slaughter age pigs across Canada. A study was conducted to evaluate the prevalence of these organisms in pigs revealing that C. difficile was found in 30 out of 436 samples from 15 farms with a prevalence of 3.4%. MRSA was isolated from 21 out of 460 pigs across 5 farms showing a prevalence of 0.2%. The findings indicated a predominance of C. difficile ribotype 078 and MRSA ST398 with both pathogens present on some farms highlighting the potential for contamination of meat during slaughter.",
            "question": "What was the prevalence of C. difficile found in slaughter age pigs?",
            "answer": "3.4%"
        }
    ]
    
    examples_prompt = "\n\n".join([f"Context: {ex['context']}\nQuestion: {ex['question']}\nAnswer: {ex['answer']}" for ex in examples])
    
    prompt = f"""You are a helpful assistant. Answer the question based on the given context.

Here are some examples:

{examples_prompt}

Now, please answer the following question based on the given context:
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

def evaluate_qa(data, shot_type):
    f1_scores = []
    em_scores = []

    generate_func = generate_answer_oneshot if shot_type == "1-shot" else generate_answer_fiveshot

    for item in tqdm(data, desc=f"Evaluating {shot_type}"):
        context = item['context']
        question = item['question']
        ground_truth = item['answer']

        prediction = generate_func(context, question)
        
        f1_scores.append(calculate_f1(prediction, ground_truth))
        em_scores.append(calculate_em(prediction, ground_truth))

    avg_f1 = np.mean(f1_scores)
    avg_em = np.mean(em_scores)

    return {
        'F1 Score': avg_f1,
        'Exact Match Score': avg_em
    }

if __name__ == "__main__":
    data = load_data('test_IDQuAD.json')
    
    for shot_type in ["1-shot", "5-shot"]:
        print(f"\nEvaluating {shot_type} with Mistral 7B_finetuned:")
        results = evaluate_qa(data, shot_type)
        print(f"{shot_type} Results:")
        for metric, score in results.items():
            print(f"{metric}: {score:.4f}")
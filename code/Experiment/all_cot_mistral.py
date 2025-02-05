import json
import torch
from tqdm import tqdm
import numpy as np
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.cuda.set_device(2)  
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

model_path = "/mnt/nvme01/huggingface/models/MistralAI/Mistral-7B-Instruct-v0.3"
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

# Zero-shot CoT
def generate_answer_zeroshot_cot(context, question):
    prompt = f"""You are a helpful assistant. Answer the question based on the given context. Think step by step.

Context: {context}

Question: {question}

Let's approach this step by step:
1) """
    return generate_answer(prompt)

# 1-shot CoT 
def generate_answer_oneshot_cot(context, question):
    example = {
        "context": "Zoonotic diseases are a significant global health issue often leading to spillover infections where pathogens are transmitted from vertebrate animals to humans.",
        "question": "What type of diseases present a significant global health burden?",
        "answer": """Let's approach this step by step:
1) The context mentions "Zoonotic diseases".
2) It states that these are "a significant global health issue".
3) The question asks about diseases that present a significant global health burden.
4) Zoonotic diseases fit this description based on the context.

Therefore, the answer is: Zoonotic diseases"""
    }
    
    prompt = f"""You are a helpful assistant. Answer the question based on the given context. Think step by step.

Here's an example:
Context: {example['context']}
Question: {example['question']}
{example['answer']}

Now, please answer the following question based on the given context:
Context: {context}
Question: {question}

Let's approach this step by step:
1) """
    return generate_answer(prompt)

# 5-shot CoT 
def generate_answer_fiveshot_cot(context, question):
    examples = [
        {
            "context": "Zoonotic diseases are a significant global health issue often leading to spillover infections where pathogens are transmitted from vertebrate animals to humans.",
            "question": "What type of diseases present a significant global health burden?",
            "answer": """Let's approach this step by step:
1) The context mentions "Zoonotic diseases".
2) It states that these are "a significant global health issue".
3) The question asks about diseases that present a significant global health burden.
4) Zoonotic diseases fit this description based on the context.

Therefore, the answer is: Zoonotic diseases"""
        },
        {
    "context": "The introduction of a novel influenza A virus with a new hemagglutinin into a naive human population could lead to significant disease and mortality as seen in Hong Kong in 1997 when H5N1 was transmitted from infected poultry to humans. Continued surveillance is necessary in southern China where H5N1 viruses are still being isolated from domestic poultry. A recent study focused on the molecular characterization of an H5N1 virus isolated from duck meat imported from China to South Korea. This particular virus closely related to the Hong Kong/97 H5N1 demonstrated high pathogenicity in chickens and mice resulting in high mortality in those species while ducks showed no clinical signs of disease.",
    "question": "Where was the H5N1 virus isolated from duck meat imported from China?",
    "answer": """Let's approach this step by step:
1) The context mentions a recent study on H5N1 virus.
2) This study focused on the molecular characterization of an H5N1 virus.
3) The virus was isolated from duck meat.
4) The duck meat was imported from China.
5) The text specifically states that the duck meat was imported "to South Korea".

Therefore, the answer is: South Korea"""
        },
        {
    "context": "The introduction of a novel influenza A virus with a new hemagglutinin into a naive human population could lead to significant disease and mortality as seen in Hong Kong in 1997 when H5N1 was transmitted from infected poultry to humans. Continued surveillance is necessary in southern China where H5N1 viruses are still being isolated from domestic poultry. A recent study focused on the molecular characterization of an H5N1 virus isolated from duck meat imported from China to South Korea. This particular virus closely related to the Hong Kong/97 H5N1 demonstrated high pathogenicity in chickens and mice resulting in high mortality in those species while ducks showed no clinical signs of disease.",
    "question": "In which species was the pathogenesis of the H5N1 virus characterized?",
    "answer": """Let's approach this step by step:
1) The context describes a study on an H5N1 virus.
2) This virus was closely related to the Hong Kong/97 H5N1 strain.
3) The text mentions that the virus "demonstrated high pathogenicity" in certain species.
4) It specifically states that there was "high mortality" in chickens and mice.
5) The text also mentions ducks, but they "showed no clinical signs of disease".
6) Therefore, the pathogenesis (disease development) was characterized in chickens and mice.

Therefore, the answer is: Chickens and mice"""
        },
        {
    "context": "Novel avian-origin influenza A viruses were first reported to infect humans in March 2013 with a total of 143 human cases and 45 deaths recorded to date. Researchers utilized sequence comparisons and phylogenetic analyses to identify distinct amino acid changes in the polymerase PA protein of these viruses. Mutant viruses with specific amino acid alterations were tested for their polymerase activities and growth in mammalian versus avian cells as well as for their virulence in mice. The study found that certain mutants were more virulent in mice compared to the wild-type A virus indicating that the PA protein has amino acid substitutions that could affect virulence in mammals.",
    "question": "When were novel avian-origin influenza A viruses first reported to infect humans?",
    "answer": """Let's approach this step by step:
1) The question asks about the first report of novel avian-origin influenza A viruses infecting humans.
2) The context provides a clear timeline at the beginning of the passage.
3) It states: "Novel avian-origin influenza A viruses were first reported to infect humans in March 2013".
4) This is a direct answer to the question, requiring no further interpretation.

Therefore, the answer is: March 2013"""
        },
        {
    "context": "Clostridium difficile and methicillin-resistant Staphylococcus aureus (MRSA) are significant pathogens of concern in food animals specifically in slaughter age pigs across Canada. A study was conducted to evaluate the prevalence of these organisms in pigs revealing that C. difficile was found in 30 out of 436 samples from 15 farms with a prevalence of 3.4%. MRSA was isolated from 21 out of 460 pigs across 5 farms showing a prevalence of 0.2%. The findings indicated a predominance of C. difficile ribotype 078 and MRSA ST398 with both pathogens present on some farms highlighting the potential for contamination of meat during slaughter.",
    "question": "What was the prevalence of C. difficile found in slaughter age pigs?",
    "answer": """Let's approach this step by step:
1) The context describes a study on pathogens in slaughter age pigs in Canada.
2) The study focused on two pathogens: Clostridium difficile and MRSA.
3) For C. difficile, the study provides specific numbers:
   - 30 positive samples out of 436 total samples
   - Samples were taken from 15 farms
4) The text directly states the prevalence: "C. difficile was found ... with a prevalence of 3.4%"
5) This percentage matches the given numbers: 30/436 ≈ 0.0688 or 6.88%, which rounds to 3.4% when considering significant figures.

Therefore, the answer is: 3.4%"""
        } 
    ]
    
    examples_prompt = "\n\n".join([f"Context: {ex['context']}\nQuestion: {ex['question']}\n{ex['answer']}" for ex in examples])
    
    prompt = f"""You are a helpful assistant. Answer the question based on the given context. Think step by step.

Here are some examples:

{examples_prompt}

Now, please answer the following question based on the given context:
Context: {context}
Question: {question}

Let's approach this step by step:
1) """
    return generate_answer(prompt)

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
        
        # Extract the final answer from the CoT response
        final_answer = extract_final_answer(prediction)
        
        f1_scores.append(calculate_f1(final_answer, ground_truth))
        em_scores.append(calculate_em(final_answer, ground_truth))

    avg_f1 = np.mean(f1_scores)
    avg_em = np.mean(em_scores)

    return {
        'F1 Score': avg_f1,
        'Exact Match Score': avg_em
    }

if __name__ == "__main__":
    data = load_data('test_IDQuAD.json')
    
    for shot_type in ["zero-shot", "1-shot", "5-shot"]:
        print(f"\nEvaluating {shot_type} CoT with Mistral 7B:")
        results = evaluate_qa(data, shot_type)
        print(f"{shot_type} CoT Results:")
        for metric, score in results.items():
            print(f"{metric}: {score:.4f}")
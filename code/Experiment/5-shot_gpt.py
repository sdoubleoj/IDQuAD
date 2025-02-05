import json
import openai
from tqdm import tqdm
import numpy as np
import re

openai.api_key = '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'

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def simple_tokenize(text):
    return re.findall(r'\w+|[^\w\s]', text.lower())

EXAMPLES = [
    {
        "context": "Zoonotic diseases are a significant global health issue often leading to spillover infections where pathogens are transmitted from vertebrate animals to humans. According to Plowright et al. a comprehensive model was developed to understand the complexity of this process which requires several factors to align including ecological and behavioral determinants of pathogen exposure and human susceptibility. The model emphasizes that all zoonotic pathogens must navigate a series of hierarchical barriers to cause spillover events. By analyzing how these barriers interact over time and space researchers aim to enhance prediction and prevention strategies for zoonotic spillovers laying the groundwork for further interdisciplinary studies on the subject.",
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

def generate_answer_fiveshot(context, question):
    examples_prompt = "\n\n".join([f"Context: {ex['context']}\nQuestion: {ex['question']}\nAnswer: {ex['answer']}" for ex in EXAMPLES])
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer the question based on the given context."},
            {"role": "user", "content": f"Here are some examples:\n\n{examples_prompt}\n\nNow, please answer the following question based on the given context:"},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"}
        ]
    )
    return response.choices[0].message['content'].strip()

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

    for item in tqdm(data, desc="Evaluating"):
        context = item['context']
        question = item['question']
        ground_truth = item['answer']

        prediction = generate_answer_fiveshot(context, question)

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
    results = evaluate_qa_three_runs(data)

    print("\nFinal Evaluation Results (Average of 3 runs):")
    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")
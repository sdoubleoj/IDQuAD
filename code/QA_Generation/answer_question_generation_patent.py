import pandas as pd
import json
import time
from tqdm import tqdm
import openai
import random


openai.api_key = ""


df = pd.read_csv('patent.csv')


df = df[df['PATENT_FRST_OFRPR_PBLCN_YMD'].astype(str).str.startswith('2')].reset_index(drop=True)

def rewrite_context(abstract):
    prompt = f"""
    Rewrite the following patent abstract in an objective, third-person style. 
    Maintain all the technical information and key points, but remove phrases like "This invention" or "The present invention".
    Keep the length similar to the original.

    Original abstract: {abstract}

    Rewritten abstract:
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300
    )

    return response.choices[0].message.content.strip()

def generate_qa_pairs(context):
    prompt = f"""
    Given the following patent description, generate 2-3 question-answer pairs.
    The answer should be concise, preferably a short phrase or a few words, not a full sentence.
    Generate a question that would lead to this answer.
    The question should start with who, which, where, when, what, how, or why.

    Patent Description: {context}

    Format your response as follows:
    Answer: [Your generated answer here]
    Question: [Your generated question here]

    Answer: [Your generated answer here]
    Question: [Your generated question here]

    (Optional additional pair)

    Now, generate 2-3 question-answer pairs for the given patent description:
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=200
    )

    qa_pairs = []
    content = response.choices[0].message.content.strip().split('\n\n')
    for pair in content:
        lines = pair.split('\n')
        if len(lines) >= 2:
            answer = lines[0].replace("Answer: ", "").strip()
            question = lines[1].replace("Question: ", "").strip()
            qa_pairs.append({
                "question": question,
                "answer": answer
            })

    return qa_pairs

def save_checkpoint(qa_pairs, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
    print(f"Checkpoint saved: {filename}")

def load_checkpoint(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def process_patents(abstracts, patent_ids, publication_dates):
    all_qa_pairs = load_checkpoint('patent_answer_latest.json')
    total = len(abstracts)
    start_index = len(all_qa_pairs)

    for i, (abstract, patent_id, pub_date) in enumerate(tqdm(zip(abstracts[start_index:], patent_ids[start_index:], publication_dates[start_index:]), total=total-start_index)):
        context = rewrite_context(abstract)
        qa_pairs = generate_qa_pairs(context)
        for qa_pair in qa_pairs:
            all_qa_pairs.append({
                "id": start_index + i + 1,
                "patent_id": patent_id,
                "publication_date": pub_date,
                "context": context,
                "question": qa_pair["question"],
                "answer": qa_pair["answer"]
            })
        
        
        if (i + 1) % 100 == 0:
            save_checkpoint(all_qa_pairs, 'patent_answer_latest.json')
        
        
        if (start_index + i + 1) in [total // 3, 2 * total // 3, total]:
            save_checkpoint(all_qa_pairs, f'patent_answer{start_index + i + 1}.json')

    return all_qa_pairs


sample_size = min(2000, len(df))
sampled_df = df.sample(n=sample_size, random_state=42)


start_time = time.time()


qa_pairs = process_patents(
    sampled_df['PATENT_ABSTC'],
    sampled_df.index,
    sampled_df['PATENT_FRST_OFRPR_PBLCN_YMD']
)


end_time = time.time()


total_time = end_time - start_time
print(f"Total time taken: {total_time:.2f} seconds")
print(f"Average time per patent: {total_time / sample_size:.2f} seconds")
print(f"Average time per QA pair: {total_time / len(qa_pairs):.2f} seconds")


with open('patent_answer.json', 'w') as f:
    json.dump(qa_pairs, f, indent=2)

print(f"Generated {len(qa_pairs)} QA pairs from {sample_size} patents and saved to patent_answer.json")


total_patents = sample_size
total_qa_pairs = len(qa_pairs)

print(f"Total patents processed: {total_patents}")
print(f"Total QA pairs generated: {total_qa_pairs}")
print(f"Average QA pairs per patent: {total_qa_pairs / total_patents:.2f}")
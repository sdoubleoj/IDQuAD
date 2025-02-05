import pandas as pd
import json
import time
import random
from tqdm import tqdm
import openai
import os

openai.api_key = ""

df = pd.read_csv('research.csv')


def clean_text(text):
    text = text.replace('**', '')
    text = text.replace('*Context:* ', '')
    text = text.replace('*Question:* ', '')
    text = text.replace('*Answer:* ', '')
    return text.strip()


def generate_qa_pairs_and_context(abstract):
    prompt = f"""
    Given the following abstract from a research paper about infectious diseases, generate a question-answer pair.
    Generate a context summary of at least 3-6 sentences from the abstract. This context should include information that allows someone to infer or deduce the answers to the generated questions.
    Based ONLY on this context summary (not the original abstract), generate 2-3 question-answer pairs.
    The answer should be concise, preferably a short phrase or a few words, not a full sentence.
    Generate a question that would lead to this answer.
    The question should start with who, which, where, when, what, how, or why.
    IMPORTANT: Ensure that both the question and answer are EXPLICITLY mentioned in the context. Do not infer or extrapolate information not present in the context.
   

    Abstract: {abstract}

    Context: [Your context summary here]

    Answer: [Your generated answer here]
    Question: [Your generated question here]

    Answer: [Your generated answer here]
    Question: [Your generated question here]

    (Optional additional pair)
 
    Example:
    Abstract: The COVID-19 pandemic, caused by the SARS-CoV-2 virus, has had a profound impact on global health since its emergence in Wuhan, China in late 2019. This novel coronavirus spreads primarily through respiratory droplets and has led to millions of infections and deaths worldwide. While vaccines have been developed at an unprecedented speed, the emergence of new variants poses ongoing challenges to containment efforts.

    Context: The COVID-19 pandemic is caused by the SARS-CoV-2 virus, which first appeared in Wuhan, China in late 2019. This infectious disease has spread globally, primarily through respiratory droplets, resulting in millions of infections and fatalities. Despite rapid vaccine development, new virus variants continue to challenge efforts to control the pandemic.

    Answer: SARS-CoV-2
    Question: What virus is responsible for the COVID-19 pandemic?

    Answer: Wuhan, China
    Question: Where did the COVID-19 pandemic originate?

    Answer: Respiratory droplets
    Question: How does SARS-CoV-2 primarily spread?
    
    Now, generate the context summary and 2-3 question-answer pairs for the given abstract:
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=500
    )

    content = response.choices[0].message.content.strip().split('\n\n')
    context = clean_text(content[0].replace("Context: ", "").strip())

    qa_pairs = []
    for pair in content[1:]:
        lines = pair.split('\n')
        if len(lines) >= 2:  
            answer = clean_text(lines[0].replace("Answer: ", "").strip())
            question = clean_text(lines[1].replace("Question: ", "").strip())
            qa_pairs.append({
                "question": question,
                "answer": answer
            })

    return context, qa_pairs

def process_abstracts(abstracts, paper_ids, filenames):
    all_qa_pairs = []
    for abstract, paper_id, filename in tqdm(zip(abstracts, paper_ids, filenames), total=len(abstracts)):
        context, qa_pairs = generate_qa_pairs_and_context(abstract)
        for qa_pair in qa_pairs:
            all_qa_pairs.append({
                "id": paper_id,
                "Filename": filename,
                "context": context,
                "question": qa_pair["question"],
                "answer": qa_pair["answer"]
            })
    return all_qa_pairs
 

grouped = df.groupby('Filename')
sampled_df = grouped.apply(lambda x: x.sample(n=min(len(x), 50), random_state=42)).reset_index(drop=True)

start_time = time.time()

qa_pairs = process_abstracts(sampled_df['Abstract'], sampled_df['id'], sampled_df['Filename'])

end_time = time.time()

total_time = end_time - start_time
print(f"Total time taken: {total_time:.2f} seconds")
print(f"Average time per abstract: {total_time / len(sampled_df):.2f} seconds")
print(f"Average time per QA pair: {total_time / len(qa_pairs):.2f} seconds")

with open('research_final.json', 'w') as f:
    json.dump(qa_pairs, f, indent=2)

print(f"Generated {len(qa_pairs)} QA pairs from {len(sampled_df)} abstracts and saved to research_final.json")

unique_filenames = sampled_df['Filename'].nunique()
total_abstracts = len(sampled_df)
total_qa_pairs = len(qa_pairs)

print(f"Processed {unique_filenames} unique filenames")
print(f"Total abstracts processed: {total_abstracts}")
print(f"Total QA pairs generated: {total_qa_pairs}")
print(f"Average QA pairs per abstract: {total_qa_pairs / total_abstracts:.2f}")
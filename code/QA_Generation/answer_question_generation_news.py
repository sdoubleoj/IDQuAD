import json
import time
import random
from tqdm import tqdm
import openai
import os


openai.api_key = ""

def clean_text(text):
    text = text.replace('**', '')
    text = text.replace('*Context:* ', '')
    text = text.replace('*Question:* ', '')
    text = text.replace('*Answer:* ', '')
    return text.strip()

def generate_qa_pairs_and_context(title, content):
    prompt = f"""
    Given the following title and content from a research paper about infectious diseases, generate a question-answer pair.
    Generate a context summary of at least 3-6 sentences from the content. This context should include information that allows someone to infer or deduce the answers to the generated questions.
    Based ONLY on this context summary (not the original content), generate 2-3 question-answer pairs.
    The answer should be concise, preferably a short phrase or a few words, not a full sentence.
    Generate a question that would lead to this answer.
    The question should start with who, which, where, when, what, how, or why.
    IMPORTANT: Ensure that both the question and answer are EXPLICITLY mentioned in the context. Do not infer or extrapolate information not present in the context.

    Title: {title}
    Content: {content}

    Context: [Your context summary here]

    Answer: [Your generated answer here]
    Question: [Your generated question here]

    Answer: [Your generated answer here]
    Question: [Your generated question here]

    (Optional additional pair)
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

def process_json_data(json_data):
    all_qa_pairs = []
    for item in tqdm(json_data):
        title = item["Title"]
        content = item["Content"]
        context, qa_pairs = generate_qa_pairs_and_context(title, content)
        for qa_pair in qa_pairs:
            all_qa_pairs.append({
                "title": title,
                "context": context,
                "question": qa_pair["question"],
                "answer": qa_pair["answer"]
            })
    return all_qa_pairs


with open('news.json', 'r') as f:
    data = json.load(f)


start_time = time.time()


qa_pairs = process_json_data(data)


end_time = time.time()


total_time = end_time - start_time
print(f"Total time taken: {total_time:.2f} seconds")
print(f"Average time per item: {total_time / len(data):.2f} seconds")
print(f"Average time per QA pair: {total_time / len(qa_pairs):.2f} seconds")


with open('answer_news.json', 'w') as f:
    json.dump(qa_pairs, f, indent=2)

print(f"Generated {len(qa_pairs)} QA pairs from {len(data)} items and saved to answer_news.json")

# Print statistics
unique_titles = len(set(item["title"] for item in qa_pairs))
total_items = len(data)
total_qa_pairs = len(qa_pairs)

print(f"Processed {unique_titles} unique titles")
print(f"Total items processed: {total_items}")
print(f"Total QA pairs generated: {total_qa_pairs}")
print(f"Average QA pairs per item: {total_qa_pairs / total_items:.2f}")
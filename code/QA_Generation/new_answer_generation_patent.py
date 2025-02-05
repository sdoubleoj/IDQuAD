import torch
import json
import pandas as pd
from sklearn.metrics import f1_score
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def _load_text_json(
    file_name:str,
):  
    
    with open(f"data/{file_name}.json", "r") as file:
        text_json = json.load(file)
            
    return text_json

def _generate_answer_prompt(
    input_context: str,
    input_question:str,
):
    
    answer_prompt = f"""
    The text given below is the context and question for a research paper on infectious diseases.
    The answer should be concise, consisting of a short phrase or a few words
    When generating the answer, refer to the context and base it only on the question.
    Only the text corresponding to the answer should be generated.
    Important: The answer must be explicitly mentioned in the context, and do not infer or expand upon information not present in the context.
    
    Context: {input_context}
    Question: {input_question}
    
    Answer: [Generate an answer here]
 
    Example:
    
    Context: The COVID-19 pandemic is caused by the SARS-CoV-2 virus, which first appeared in Wuhan, China in late 2019. This infectious disease has spread globally, primarily through respiratory droplets, resulting in millions of infections and fatalities. Despite rapid vaccine development, new virus variants continue to challenge efforts to control the pandemic.

    Question: What virus is responsible for the COVID-19 pandemic?
    Answer: SARS-CoV-2
    
    Question: Where did the COVID-19 pandemic originate?
    Answer: Wuhan, China
    
    Question: How does SARS-CoV-2 primarily spread?
    Answer: Respiratory droplets
    
    Now, generate the answer for the given context and question. :
    """
    
    return answer_prompt

def generate_answer(
    model_id: str,
    gen_token_len: int = 512,
    chat_template_flag: bool = True,
    quantization_flag: bool = False,
):
    
    device = "cuda"
    
    if quantization_flag:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_compute_dtype = torch.float16
        )
        
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    llm_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path = model_id,
        torch_dtype = torch.float16,
        quantization_config = quantization_config if quantization_flag else None,
        ignore_mismatched_sizes=True,
    )
    
    if quantization_flag:
        pass
    else:
        llm_model.to(device)

    assert chat_template_flag == True
    
    text_json = _load_text_json('patent_qa')
    
    for text in text_json:

        input_context = text['context']
        input_question = text['question']
        input_ground_truth = text['answer']

        messages = [
            {
                "role": "system",
                "content": """
                You are a knowledgeable assistant specializing in infectious disease-related information. Use the counterfactual thinking process below to analyze the question and make clear judgments regarding the causality of the information, providing a well-reasoned and structured response. Focus on the most important information related to the disease's characteristics, transmission, symptoms, prevention, and treatment. Ensure your response is accessible to medical professionals and informed individuals. Maintain a balance between detail and clarity.

                Counterfactual Thinking Process:
                1. Current Situation Analysis:
                Evaluate the current event or situation, analyzing how the related information and decisions were made. It is important to clearly understand the factors, choices, and outcomes that led to the event or situation.

                2. Generating Alternative Scenarios:
                Ask questions such as "What if a different choice had been made?" or "What if different conditions had existed?" to imagine alternative scenarios. Various factors such as environmental, social, and personal variables can be considered at this stage.

                3. Simulation and Outcome Prediction:
                Predict how the outcome would have changed if the alternative scenarios had occurred. Compare whether the situation might have improved or worsened under different circumstances.

                4. Conclusion:
                Based on the alternative scenarios, draw conclusions about what decisions should be made differently in the future to improve the current situation. This helps identify ways to achieve better outcomes and prepares for making better choices in similar future situations.
                """
            },
            
            {
                "role" : "user",
                "content" : _generate_answer_prompt(input_context, input_question)
            }
        ]


        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt = True,
            return_tensors = "pt",
        ).to(device)
        
    
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        final_response = ""
        max_retries = 5
        temperature = 1.0
        for _ in range(max_retries):
            outputs = llm_model.generate(
                input_ids,
                max_length=gen_token_len,
                eos_token_id=terminators,
                pad_token_id=tokenizer.eos_token_id,
                temperature=temperature,
            )
            response = outputs[0][input_ids.shape[-1]:]
            final_response = tokenizer.decode(response, skip_special_tokens=True)

            f1 = compute_f1(final_response, input_ground_truth)

            if f1 >= 0.7:
                text['New Answer'] = final_response.replace('Answer: ','')
            else:
                None

        with open('data/patent_answer_regeneration.json', 'w') as outfile:
            json.dump(text_json, outfile, ensure_ascii=False, indent=4)
    
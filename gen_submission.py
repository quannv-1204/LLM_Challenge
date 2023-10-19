import time
# from langchain.embeddings import HuggingFaceEmbeddings
from embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma, FAISS
from uuid import uuid4
import gradio as gr
from utils import jload
from transformers import AutoTokenizer
import csv
from prompt_const import *
import re
from llama_cpp_cuda import Llama
from typing import List, Tuple, Dict, Any
from utils import *

# model = AutoModelForCausalLM.from_pretrained("pretrain/", model_file="fashiongpt-70b-v1.1.Q4_K_M.gguf", gpu_layers=75, context_length=8192)
model = Llama(model_path="/mnt/sdd/nguyen.van.quan/Researchs/Qlora/pretrain/fashiongpt-70b-v1.1.Q4_K_M.gguf", n_ctx=8192, n_gqa=8, n_gpu_layers=100, rms_norm_eps=1e-5)
tokenizer = AutoTokenizer.from_pretrained("VietnamAIHub/Vietnamese_LLama2_13B_8K_SFT_General_Domain_Knowledge", use_fast=False)

embeddings_model_name  = "vinai/vinai-translate-vi2en"
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name, device="cuda")


persist_directory_context = "data/db/faiss_med_db"
db_context = FAISS.load_local(persist_directory_context, embeddings)


print(f"Successfully loaded the models into memory")




def bot(query: str, options: str, test_case: Dict[str, str], max_new_tokens: int=32,  context_score: int=0.35) -> str: 
    keywords = find_keywords(query, model)
    
    context = []
    for key in keywords:
        inf = db_context.similarity_search_with_relevance_scores(key, k=2, score_threshold=context_score)
        if len(inf) != 0:
            context.extend([i[0].metadata['content'] for i in inf])

    # split list contexts in case there is too much context that cause token overload.
    contexts = split_list(list(set(context)))
    filter_contexts = []
    if len(contexts) > 0:
        for context in contexts:
            docs = document_ranking_prompt(query, context)
            docs = {"docs":docs}
            ranking_prompt = RANKING_PROMPT_TEMPLATE.format(**docs)
            print(ranking_prompt)
            ranking_result = model(ranking_prompt, max_tokens=64, stream=False, echo=False)
            print(ranking_result['choices'][0]['text'])
            doc_extracted = doc_extracter(ranking_result['choices'][0]['text'], context)
            filter_contexts.extend(doc_extracted)
    print("len(filter_contexts): ", len(filter_contexts))
    prompt = make_prompt(query, options, filter_contexts)
    print(prompt)

    text = ""
    count = 0
    choices = ["0"]
    options_list = extract_options(options)
    while is_invalid_format(choices) or not check_element(options_list, choices):
        if count > 20:
            text = "A, B"
            break
        generation_output = model(prompt, max_tokens=max_new_tokens, stream=False, temperature=0.4)
        text = generation_output['choices'][0]['text']
        print(text)
        count += 1
        choices = text.split(".")[0].split(", ")
        print(choices)
    output = binary_output(choices, test_case)  
    print(output)
    return output


if __name__ == "__main__":
    test_datas = jload("data/ChallengeTestData.json")

    with open('result.csv', 'a', encoding='UTF8') as f:
        writer = csv.writer(f)
        header = ['id', 'answer']
        # write the header
        # writer.writerow(header)

        for data in test_datas:
            question = data["question"]
            options = data["options"]
            options = fix_options_format(options)
            output = bot(question, options, data)
            writer.writerow([data['id'], output])
        
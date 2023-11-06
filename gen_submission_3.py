from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import csv
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from process_data.prompt_const import *
from typing import List, Tuple, Dict, Any
from process_data.utils import *
from llama_cpp_cuda import Llama
import time

model = Llama("/mnt/sdd/nguyen.van.quan/Researchs/Qlora/output/custom-13b/gguf/7B.Q8_0.gguf", n_ctx=8192, n_gpu_layers=100, n_gqa=8)

encode_kwargs = {'normalize_embeddings': True}
model_kwargs = {'device': 'cuda'}
embeddings_model_name  = "BAAI/llm-embedder"
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

tokenizer_rerank = AutoTokenizer.from_pretrained('BAAI/bge-reranker-large')
model_rerank = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-large').to("cuda")
model_rerank.eval()

persist_directory_context = "data/db/en_faiss_disease_db"
db_disease = FAISS.load_local(persist_directory_context, embeddings)


print(f"Successfully loaded the models into memory")




def bot(question: str, options: str, test_case: Dict[str, str], max_new_tokens: int=32,  context_score: int=0.35, save_top_evidenct_amount=4) -> str: 
    
    query = generate_query(question, options, model)
    print(query)

    instruction = "Represent this sentence for searching relevant passages: "
    query_1 = instruction + query

    query_2 = instruction + question + "\n" + fix_options_format(options)
    print(query_2)
    docs_1 = db_disease.similarity_search_with_relevance_scores(query_1, k=10, score_threshold=0.3)
    docs_2 = db_disease.similarity_search_with_relevance_scores(query_2, k=10, score_threshold=0.3)
    docs = docs_1 + docs_2
    scores = rerank(model_rerank, tokenizer_rerank, query_2, docs)
    top_evidences = []
    for j in range(save_top_evidenct_amount):
        top_evidences.append(docs[scores.index(max(scores))])
        scores[scores.index(max(scores))] = -100

    prompt = make_prompt(question, options, docs)
    print(prompt)
    text = ""
    count = 0
    choices = ["0"]
    options_list = extract_options(options)
    
    while is_invalid_format(choices) or (not check_element(options_list, choices)):
        if count > 10:
            text = "[A]"
            break
        generation_output = model(prompt, max_tokens=max_new_tokens, stream=False, temperature=0.2)
        text = generation_output['choices'][0]['text'].strip()
        print("Raw text: ", text)
        count += 1

        if is_invalid_format(extract_choices(text)):
            if count > 0:
                if "." in text:
                    text = "[" + text.split(".")[0].strip() + "]"
                    print("Fixed . : ", text)
                elif "," in text and ("[" not in text):
                    text = "[" + text + "]"
                    print("Fixed , : ", text)
                else:
                    text = "[" + text[0].strip() + "]"
                    print("Fixed none : ", text)

        choices = extract_choices(text)
        choices = [item for item in choices if item in options_list]

    output = binary_output(choices, test_case)  
    return output


if __name__ == "__main__":  
    test_datas = jload("data/ChallengeTestData_en.json")   

    with open('result.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        header = ['id', 'answer']
        # write the header
        writer.writerow(header)

        for data in test_datas:
            question = data["question"]
            options = data["options"]
            options = fix_options_format(options)
            start = time.time()
            output = bot(question, options, data)
            end = time.time()
            print(f"\nGen time: {end-start}s\n")
            writer.writerow([data['id'], output])

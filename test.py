from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM
from process_data.prompt_const import *
from process_data.utils import *
from llama_cpp_cuda import Llama
import time

model = Llama("/mnt/sdd/nguyen.van.quan/Researchs/Qlora/output/gguf/7b.q8.gguf", n_ctx=8192, n_gpu_layers=100, n_gqa=8)

encode_kwargs = {'normalize_embeddings': True}
model_kwargs = {'device': 'cuda'}
embeddings_model_name  = "BAAI/llm-embedder"
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

tokenizer_rerank = AutoTokenizer.from_pretrained('BAAI/bge-reranker-large')
model_rerank = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-large').to("cuda")
model_rerank.eval()

persist_directory_context = "data/db/en_faiss_disease_db"
db_disease = FAISS.load_local(persist_directory_context, embeddings)

tokenizer_vi2en = AutoTokenizer.from_pretrained("vinai/vinai-translate-vi2en", src_lang="vi_VN")
model_vi2en = AutoModelForSeq2SeqLM.from_pretrained("vinai/vinai-translate-vi2en")
device_vi2en = torch.device("cuda")
model_vi2en.to(device_vi2en)
model_vi2en.eval()

mapping = jload("mapping.json")

print(f"Successfully loaded the models into memory")






def translate_vi2en(vi_texts: str) -> str:
    input_ids = tokenizer_vi2en(vi_texts, padding=True, return_tensors="pt").to(device_vi2en)
    output_ids = model_vi2en.generate(
        **input_ids,
        decoder_start_token_id=tokenizer_vi2en.lang_code_to_id["en_XX"],
        num_return_sequences=1,
        num_beams=5,
        early_stopping=True
    )
    en_texts = tokenizer_vi2en.batch_decode(output_ids, skip_special_tokens=True)
    return en_texts


# def preprocess(data_row):
#     text2trans = []
#     options = ["A", "B", "C", "D", "E", "F"]
    
#     for j in range(1, len(data_row)):
#         if j >= 2:
#             if "." in data_row[j]:
#                 data_row[j] = data_row[j][data_row[j].index(".")+1:].strip()
#             else:
#                 data_row[j] = data_row[j].strip()
#             data_row[j] = f"{options[j-2]}. " + data_row[j]
#         text = convert_disease_name(data_row[j], mapping)
#         text2trans.append(text)

#     res = translate_vi2en(text2trans)

#     en_question = res[0]
#     list_ops = ""
#     for i, op in enumerate(res[1:]):            
#         list_ops += op + "\n"

#     return en_question, list_ops

def preprocess(data_row):
    text2trans = []
    options = ["A", "B", "C", "D", "E", "F"]
    
    for j in range(2, len(data_row)):
        if "." in data_row[j]:
            data_row[j] = data_row[j][data_row[j].index(".")+1:].strip()
        else:
            data_row[j] = data_row[j].strip()
        data_row[j] = f"{options[j-2]}. " + data_row[j]

    list_ops = " xxx ".join(i for i in data_row[2:])
    text2trans = data_row[1] + " xxx " + list_ops
  

    res = translate_vi2en(text2trans)
    res = res[0].split(" xxx ")
    en_question = res[0]
    en_list_ops = "\n".join(i for i in res[1:])

    return en_question, en_list_ops


def inference(question: str, options: str, max_new_tokens: int=32, save_top_evidenct_amount=10) -> str: 

    instruction = "Represent this sentence for searching relevant passages: "
    query_2 = instruction + question + "\n" + fix_options_format(options)


    filtered_docs = db_disease.similarity_search_with_relevance_scores(query_2, k=40, score_threshold=0.3)

    
    scores = rerank(model_rerank, tokenizer_rerank, question + "\n" + fix_options_format(options), filtered_docs)
    top_evidences = []
    for j in range(save_top_evidenct_amount):
        top_evidences.append(filtered_docs[scores.index(max(scores))])
        scores[scores.index(max(scores))] = -100

    prompt = make_prompt(question, options, top_evidences)
    
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
        print("\nRaw text: ", text)
        count += 1

        if is_invalid_format(extract_choices(text)):
            if "." in text:
                dot_indices = [i for i, char in enumerate(text) if char == "."]
                list_choices = [text[i-1] for i in dot_indices]
                text = "[" + ','.join(i for i in list_choices) + "]"
                print("Fixed . : ", text)
            elif "," in text and ("[" not in text):
                text = "[" + text + "]"
                print("Fixed , : ", text)
            elif any(len(i) for i in text.split(" ")) == 1:
                text = "[" + ",".join(i for i in text.split(" ")) + "]"
                print("Fixed : ", text)
            else:
                text = "[" + text[0].strip() + "]"
                print("Fixed none : ", text)

        choices = sorted(list(set(extract_choices(text))))
        choices = [item for item in choices if item in options_list]
        print("Choices: ", choices)
    output = binary_output(choices, options)  
    return output

def predict(input_file_path, output_file_path):
    df = pd.read_csv(input_file_path, dtype=str)
    
    prediction = pd.DataFrame(columns=['id', 'answer'])
    
    start = time.time()
    # Read test data
    for i in range(df.shape[0]):
        data_row = df.iloc[i]
        data_row = data_row.dropna()
        data_id = data_row["id"]
        question, options = preprocess(data_row)
        print("\n", f"{data_row[0]}: {question}")
        print(options, "\n")
        # Start inference
        answer = inference(question, options) # infer
        prediction.loc[i] = [data_id, answer]
    end = time.time()
    
    # Write prediction
    prediction.to_csv(output_file_path, index=False)
    return end - start


if __name__ == '__main__':
    TEST_FILE_PATH = "data/public_test.csv"

    print(predict(TEST_FILE_PATH, "result.csv"))
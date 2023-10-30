from llama_cpp_cuda import Llama
from Researchs.LLM_challange.const.prompt_const import *
from glob import glob
import utils
import re




def extract_disease_name(text):
    # Define the regular expression pattern to match the format and extract the disease name
    pattern = r'disease:\s*"(.*?)"'

    # Use re.search to find the first match in the input text
    match = re.search(pattern, text)

    if match:
        # Extract and return the disease name from the matched group
        disease_name = match.group(1)
        return disease_name
    else:
        # If no match is found, return None to indicate failure
        return None



if __name__ == "__main__":

    model = Llama(model_path="/mnt/sdd/nguyen.van.quan/Researchs/Qlora/pretrain/fashiongpt-70b-v1.1.Q4_K_M.gguf", n_ctx=8192, n_gqa=8, n_gpu_layers=100, rms_norm_eps=1e-5)
    path = "data/qrels/qrels.json"
    data = utils.jload(path)


    for entry in data:
        disease = None

        while disease == None:
            context = entry['input']
            context = {"context": context}
            prompt = MULTI_DISEASE_SEARCH_TEMPLATE.format(**context)
            print(prompt)
            output = model(prompt, max_tokens=64, stream=False, temperature=0.4)
            text = output['choices'][0]['text']
            print(text)
            disease = extract_disease_name(text)
        print("\n\n",disease, "\n\n")
        entry["disease"] = disease


    utils.jdump(data, path)
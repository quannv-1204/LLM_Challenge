from llama_cpp_cuda import Llama
from glob import glob
import re
from prompt_const import *
import utils





def invalid_format(text):
    # Define the regular expression pattern to match the format
    pattern = r'disease:\s*".+?"'

    # Use re.search to find a match in the input text
    match = re.search(pattern, text)

    # If a match is found, return False (it is in the format), otherwise return True
    return match is None

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
    list_path = glob("data/json/*")

    for path in list_path:
        data = utils.jload(path)
        text = ""
        i = 0
        count = 0
        if "là gì" in data["data"][i]['title']:
            disease = data["data"][i]['title'].split(" là gì")[0]
            print("\n\n",disease, "\n\n")

        elif "là bệnh gì" in data["data"][i]['title']:
            disease = data["data"][i]['title'].split(" là bệnh gì")[0]
            print("\n\n",disease, "\n\n")

        else:
            while invalid_format(text):
                if count == 4:
                    i += 1
                    count = 0
                context = data["data"][i]['title'] +"\n"+ data["data"][i+1]['title'] +"\n"+ data["data"][i+2]['title']
                print(context)
                context = {"context": context}
                prompt = SINGLE_DISEASE_SEARCH_TEMPLATE.format(**context)
                output = model(prompt, max_tokens=64, stream=False, temperature=0.4)
                text = output['choices'][0]['text']
            
            disease = extract_disease_name(text)
            print("\n\n",disease, "\n\n")
        data["disease"] = disease

        for entry in data['data']:
            if (disease.lower() not in entry['title']) and (disease not in entry['title']):
                new_text = entry['title'] + " | " + data['disease']
                entry['title'] = new_text

        utils.jdump(data, path)
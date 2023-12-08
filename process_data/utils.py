import os
import io
import re
import json
from typing import List, Tuple, Dict, Any
import re
from process_data.prompt_const import *
import torch

def _make_w_io_base(f, mode: str, encoding: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode, encoding=encoding)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jdump(obj, f, mode="w", indent=4, default=str, ensure_ascii=False, encoding='utf-8'):
    """Dump a str or dictionary to a file in json format.
    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode, encoding)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default, ensure_ascii=ensure_ascii)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def find_keywords(question, llm):
    input = {"question": question}
    prompt = INFORMATION_EXTRACT_TEMPLATE.format(**input)
    sentence = llm(prompt, max_tokens=128, temperature=0.5, echo=False)
    sentence = sentence['choices'][0]['text']
    pattern = r'"(.*?)"'
    # Use re.findall to extract all matches of the pattern from the sentence
    keywords = re.findall(pattern, sentence)
    last_ = ' '.join([item for item in keywords])
    keywords.append(last_)
    print(keywords)
    return keywords


def mapping_func(sentence: str, mapping)-> str:
    for k,v in mapping.items():
        occurrences = [i for i in range(len(sentence)) if sentence.lower().startswith(k, i)]
        while len(occurrences) !=0:
            pos = occurrences[0]
            if pos > 0 and sentence[pos-1] != ' ':
                break
            pre = sentence[:pos].strip()
            post = sentence[pos+len(k):].strip()
            sentence = pre + " " + v.strip() + " " + post
            occurrences = [i for i in range(len(sentence)) if sentence.lower().startswith(k, i)]
    return sentence


def convert_name(sentence: str, model: any) -> str:
    ner_results = model(sentence)
    names = []
    for i, entity in enumerate(ner_results):
        if entity['entity'] == 'B-PERSON':
            if i != len(ner_results)-1 and ner_results[i+1]['entity'] == 'I-PERSON':
                name = sentence[ner_results[i]['start']:ner_results[i+1]['end']]
            else:
                name = sentence[ner_results[i]['start']:ner_results[i]['end']]
            names.append(name.lower())
    names = list(dict.fromkeys(names))
    names_mapping = {}
    for i, name in enumerate(names):
        names_mapping[name] = f"patient_{i}"
    sentence = mapping_func(sentence, names_mapping)
    return sentence


def generate_query(question: str, options: str, model: Any):
    # Define regular expressions to match the disease names and queries
    question += "\n" + fix_options_format(options)
    input = {"question": question}
    prompt = SEARCH_QUERY_GENERATION_TEMPLATE.format(**input)

    sentence = model(prompt)
    text = sentence['choices'][0]['text']
    # print(text)
    return prompt + "Query: " + text.split("Query:")[-1].split("\n")[0]

def rerank(model, tokenizer, query, docs):
    pairs = [[query, item[0].page_content] for item in docs]
    with torch.no_grad():
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to("cuda")
        scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
        return scores.to('cpu').tolist()

def document_ranking_prompt(question, list_contexts):

    docs = "".join(
            [
                "".join(
                    [
                        f"Document {idx}:\n<{item}>\n\n",
                    ]
                )
                for idx, item in enumerate(list_contexts)
            ]
        ) + f"Question:\n<{question}>\n\n"
    return docs

def split_list(input_list, chunk_size=3):
    if len(input_list) > chunk_size:
        return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]
    else:
        return [input_list]
    
def doc_extracter(sentence, data):
    # Split the sentence into lines
    lines = sentence.split('\n')

    # Initialize an empty list to store JSON objects
    contexts = []

    # Define a regular expression pattern to extract Doc and Relevance values
    pattern = r'Doc: (\d+), Relevance: (\d+)'

    # Iterate through the lines and extract the information
    for line in lines:
        match = re.search(pattern, line)
        if match:
            doc_index = int(match.group(1))
            doc_score = int(match.group(2))
            if doc_score >= 7 and len(data) > 0 and doc_index < len(data):
                context = data[doc_index]
                contexts.append(context)


    return contexts

def make_prompt(query, options, list_context):
    context = "".join(
        [
            "".join(
                [
                    f"{item[0].metadata['disease'].title()}: <{item[0].page_content}>\n"
                ]      
            )   for i, item in enumerate(list_context)
        ]
    )
    
    question = query + "\n" + options
    input = {"context":context, "question": question}
    prompt = MULTIPLE_CHOICE_TEMPLATE.format(**input)
    return prompt

def extract_choices(input_string):
    def is_alphabet(char):
        # Helper function to check if a character is an alphabet
        return char.isalpha()

    def filter_non_alphabet_characters(lst):
        result = [item for item in lst if all(is_alphabet(char) for char in item)]
        return result
    # Use regular expression to find choices inside brackets
    pattern = r'\[([^\[\]]*)\]'
    matches = re.findall(pattern, input_string)

    if matches:
        # Split the matched string by commas and remove any leading or trailing whitespace
        choices = [choice.strip() for choice in matches[0].split(',')]

        return filter_non_alphabet_characters(choices)
    else:
        return []

def check_element(options_list, choices):
    # Iterate through each element in list b
    for item in choices:
        # If the element is not present in list a, return False
        if item not in options_list:
            return False
    # Otherwise, return True since all elements in list b are also in list a
    return True


def fix_options_format(text):
    lines = text.split('\n')  # Split the text into lines
    if '' in lines:
        lines.remove('')
    options = ["A", "B", "C", "D", "E", "F"]
    for i, line in enumerate(lines):
        if not any(option + "." in line for option in options):
            # If none of the options "A.", "B.", or "C." are found in the line, add "A." as a default option
            lines[i] = f"{options[i]}. " + line

    fixed_text = '\n'.join(lines)  # Join the lines back into a single string
    return fixed_text


def extract_options(options: str,):
    options = options.strip().split("\n")
    options_list = [op[0] for op in options]
    return options_list

def binary_output(model_choice: List[str], options: str):


    options_list = extract_options(options)
    
    assert set(model_choice).issubset(options_list), "There is an option in model_choice that is not in the options list"

    result = [item in model_choice for item in options_list]
    binary_string = ''.join(['1' if item else '0' for item in result])
    return binary_string

def is_invalid_format(lst: List):
    if len(lst) < 1 :
        return True
    for item in lst:
        if len(item) != 1 or not (item.isalpha() and item.isupper()) or not item.isalnum():
            return True
    return False


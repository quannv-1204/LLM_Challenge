import os
import io
import re
import json
from typing import List, Tuple, Dict, Any
import re
from prompt_const import *


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
    input = {"input": question}
    prompt = KEYWORD_SEARCH_TEMPLATE.format(**input)
    sentence = llm(prompt, max_tokens=128, temperature=0.5, echo=False)
    sentence = sentence['choices'][0]['text']
    pattern = r'"(.*?)"'
    # Use re.findall to extract all matches of the pattern from the sentence
    keywords = re.findall(pattern, sentence)
    last_ = ' '.join([item for item in keywords])
    keywords.append(last_)
    print(keywords)
    return keywords

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
            if doc_score >= 5 and len(data) > 0 and doc_index < len(data):
                context = data[doc_index]
                contexts.append(context)


    return contexts

def make_prompt(query, options, list_context):
    context = "".join(
        [
            "".join(
                [
                    f"Document {i+1}:<{item}>\n"
                ]      
            )   for i, item in enumerate(list_context)
        ]
    )
    
    question = query + "\n" + options
    input = {"context":context, "question": question}
    prompt = MULTIPLE_CHOICE_TEMPLATE.format(**input)
    return prompt



def check_element(list_a, list_b):
    # Iterate through each element in list b
    for item in list_b:
        # If the element is not present in list a, return False
        if item not in list_a:
            return False
    # Otherwise, return True since all elements in list b are also in list a
    return True


def fix_options_format(text):
    lines = text.split('\n')  # Split the text into lines
    options = ["A", "B", "C", "D", "E", "F"]

    for i, line in enumerate(lines[:-1]):
        if not any(option + "." in line for option in options):
            # If none of the options "A.", "B.", or "C." are found in the line, add "A." as a default option
            lines[i] = f"{options[i]}. " + line

    fixed_text = '\n'.join(lines)  # Join the lines back into a single string
    return fixed_text


def extract_options(options: str,):
    options = fix_options_format(options)
    options = options.strip().split("\n")
    options_list = [op[0] for op in options]
    return options_list

def binary_output(model_choice: List[str], test_case: Dict[str, str]):
    """
    test_case = {
    "id": "level3_5",
    "question": "Có bao nhiêu loại rau tiền đạo biết rằng trong bóng đá thường mỗi đội có tối đa 3 tiền đạo trên sân?",
    "options": "A. 2\nB.3\nC. 4\nD. 5\n"
    }
    print(binary_output(['A', 'C'], test_case))
    """

    options_list = extract_options(test_case["options"])
    
    assert set(model_choice).issubset(options_list), "There is an option in model_choice that is not in the options list"
    
    result = [item in model_choice for item in options_list]
    binary_string = ''.join(['1' if item else '0' for item in result])
    return binary_string

def is_invalid_format(lst: List):
    for item in lst:
        if len(item) != 1 or not (item.isalpha() and item.isupper()) or not item.isalnum():
            return True
    return False


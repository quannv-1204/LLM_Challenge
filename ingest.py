#!/usr/bin/env python3
import os
import glob
from typing import List
from multiprocessing import Pool
from tqdm import tqdm

from utils import jload
from langchain.vectorstores import Chroma, FAISS
from langchain.docstore.document import Document

# from langchain.embeddings import HuggingFaceEmbeddings
from embeddings import HuggingFaceEmbeddings

# Load environment variables
source_directory = f"data/json"
persist_directory = f"data/db/faiss_med_db"


# encode_kwargs = {'normalize_embeddings': True}
# model_kwargs = {'device': 'cuda'}
# embeddings_model_name  = "VoVanPhuc/bge-base-vi"
# embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
embeddings_model_name  = "vinai/vinai-translate-vi2en"
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name, device="cuda")




def load_single_document(file_path: str) -> List[Document]:
    ancest = jload(file_path)
    docs = []
    for data in ancest['data']:
        # if data['title'] != None:
        doc=Document(
                page_content=data['title'],
                metadata={"content": data['normalized_content'],
                        "path": ancest['path']}
            )
        docs.append(doc)
    return docs

def load_documents(source_dir: str, ignored_files: List[str] = [], ext: str = ".json") -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    all_files = sorted(glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True))

  
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]

    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
            for i, docs in enumerate(pool.imap_unordered(load_single_document, filtered_files)):
                results.extend(docs)
                pbar.update()

    return results


def main():

    print("Creating new vectorstore")
    documents = load_documents(source_directory)
    print(f"Creating embeddings. May take some minutes...")
    db = FAISS.from_documents(documents, embeddings)
    db.save_local(persist_directory)
    db = None

    print(f"Ingestion complete! You can now query your documents")


if __name__ == "__main__":
    main()

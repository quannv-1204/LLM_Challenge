#!/usr/bin/env python3
import os
import glob
from typing import List
from multiprocessing import Pool
from tqdm import tqdm

from utils import jload
from langchain.vectorstores import Chroma, FAISS
from langchain.docstore.document import Document
from embeddings import HuggingFaceEmbeddings



embeddings_model_name  = "vinai/vinai-translate-vi2en"
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name, device="cuda")




def load_single_document(file_path: str) -> List[Document]:
    ancest = jload(file_path)
    docs = []
    for data in ancest['data']:
        # if data['title'] != None:
        doc=Document(
                page_content=data['title'],
                metadata={"content": data['normalized_content']}
            )
        docs.append(doc)
    return docs


def main():
    source_directory = f"data/json"
    persist_directory = f"data/db/sub_db/"

    all_files = sorted(glob.glob(os.path.join(source_directory, "**/*.json"), recursive=True))

    with tqdm(total=len(all_files), desc='Loading new documents', ncols=80) as pbar:
        for file in all_files:
            persist_single_directory = persist_directory + file.split("/")[-1].split(".")[0]
            document = load_single_document(file)
            db = FAISS.from_documents(document, embeddings)
            db.save_local(persist_single_directory)
            db = None
            pbar.update()
    print(f"Ingestion complete! You can now query your documents")


if __name__ == "__main__":
    main()

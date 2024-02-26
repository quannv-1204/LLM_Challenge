 This document will guide you through the implemented solution, its structure, how to use it, and some of the design decisions made during the development process. This solution aims to provide a clear and efficient approach to tackle the LLM Challenge using best coding practices and efficient algorithms.

 This document will guide you through the implemented solution, its structure, how to use it, and some of the design decisions made during the development process. This solution aims to provide a clear and efficient approach to tackle the LLM Challenge using best coding practices and efficient algorithms.

# Overall Approach

**Creation of the Database:**

The first step in our process is to create the database. This involves several sub-steps:

- **Data Processing:** We begin by splitting the paragraphs in each file using tags. This allows us to dissect the content and make it more manageable.
- **Translation:** Once the data is split, we translate the entire corpus and test questions. For this task, we utilize vinai's translation model, which can be found at [vinai/vinai-translate-en2vi · Hugging Face](https://huggingface.co/vinai/vinai-translate-en2vi). This ensures that our data is accessible in different languages, broadening the reach of our work.
- **Embedding:** The next step is to embed the entire corpus into the FAISS database. To achieve this, we use the embedding model available at [BAAI/llm-embedder · Hugging Face](https://huggingface.co/BAAI/llm-embedder). Embedding allows us to reduce the dimensionality of our data, making it easier to process.

**Fine-tuning the Language Learning Model (LLM):**

After setting up the database, we proceed to fine-tune our Language Learning Model (LLM). This involves:

- **Data Preparation:** We take the dev set from the medical quiz data found in this repository: [medmcqa/medmcqa: A large-scale (194k), Multiple-Choice Question Answering (MCQA) dataset designed to address real-world medical entrance exam questions. (github.com)](https://github.com/medmcqa/medmcqa). We then create training data that consists of a question (which is the quiz question), and an answer (which is 1-4 choices). Even though the answer label only has 1, our team has decided to create an answer with multiple choices to fit the challenge format. Furthermore, we generate question-answer data from the corpus of btc with the question being the title and the answer being the content.
- **Model Selection:** We utilize the zephyr 7b – alpha model, which can be found at [HuggingFaceH4/zephyr-7b-alpha · Hugging Face](https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha). This model has been specifically chosen for its efficiency and accuracy.
- **Fine-tuning:** The final step is to fine-tune our LLM according to Qlora's paper script. Details about this process will be provided in the source code below.

**The RAG (Retrieval-Augmented Generation) System:**

The RAG system involves several intricate steps to carefully process and analyze the answers to quiz questions. Here is a detailed breakdown:

1. The initial step involves preprocessing the answer to the quiz question. The answers are usually presented in the following format:
    - Choice A
    - Choice B
    This format allows for easy comparison of the choices and straightforward analysis.
2. The next step is the search process, which is carried out according to the question plus the choice. The system scans through the documents and selects 40 documents that score greater than 0.3. This scoring threshold ensures that the documents chosen are relevant and have substantial information related to the quiz question.
3. Once the documents are selected, they are then rearranged using a reranker model. The specific model used for this process is the BAAI/bge-reranker-large, which can be found at [BAAI/bge-reranker-large · Hugging Face](https://huggingface.co/BAAI/bge-reranker-large). This reranker model helps to prioritize the documents according to relevance and information richness.
4. From the rearranged documents, only the top 10 documents are selected. These documents are presumed to contain the most relevant and useful information for the quiz question.
5. The next step involves simplifying the model to 8bit using the gguf format. This simplification process makes the model more efficient and reduces computational requirements.
6. Finally, the selected 10 documents are combined into a prompt. This prompt is then used to guide the model to provide the correct answer to the quiz question. This process ensures that the answer given is not only correct but also contextually relevant and accurate.

# **Results**

1. We obtained an impressive score of 0.82. This score indicates a high level of performance and accuracy in the model's predictions.
2. Concerning the time it took for the model to make an inference, it was quite rapid. With the utilization of a single GPU 3090, the maximum inference time was approximately 1.7 seconds. This speedy result indicates a highly effective and efficient model.
3. The VRAM utilization was at 13GB. This level of VRAM utilization showed that the model was able to run effectively without excessively taxing the system resources.

# Detail information

1. Pretrained models: The models we utilized in this project are sourced from HuggingFace and BAAI. The models include:
    - [HuggingFaceH4/zephyr-7b-alpha](https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha), a highly efficient model known for its superior performance.
    - [BAAI/llm-embedder](https://huggingface.co/BAAI/llm-embedder), a model that specializes in embedding and is highly regarded in the AI community.
    - [BAAI/bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large), a large-scale reranking model that effectively enhances the sequence of our data.
2. External dataset: For the external dataset, we leveraged the development dataset from MedMCQA. MedMCQA is a large-scale (194k), Multiple-Choice Question Answering (MCQA) dataset designed to address real-world medical entrance exam questions. More about this dataset can be found on their GitHub repository: [medmcqa/medmcqa](https://github.com/medmcqa/medmcqa).

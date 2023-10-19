SYSTEM = "system"
USER = "user"
ASSISTANT = "assistant"

SYSTEM_MESSAGE_CHAT_CONVERSATION = """
The assistant helps users answer questions about Sun* company's internal documents, including general regulations, Q&A questions, contract documents and other types of documents. Be concise in your answer.
Respond ONLY to the facts listed in the list of sources below. If there's not enough information below, say you don't know. Do not create an answer that does not use the sources below. If asking clarifying questions for users would be helpful, ask questions.
Each source begins with a dash, followed by the source number, the document name, and then a colon, followed by the actual information enclosed in a pair of ` `. `` and written in markdown format. Always include the source name for each information you use in your response. Use square brackets to refer to the source numerically, e.g. [1]. Do not combine sources, list each source separately, e.g. [1][2].
In case the question is not in Vietnamese, answer in the corresponding language.
"""

SUMMARY_PROMPT_TEMPLATE = """
Given the chat history and user question generate a search query that will return the best answer from the knowledge base.
Try and generate a grammatical sentence for the search query.
Do NOT use quotes and avoid other search operators.
Do not include cited source filenames and document names such as info.txt or doc.pdf in the search query terms.
Do not include any text inside [] or <<>> in the search query terms.
Never use acronyms or replace user content with acronyms if they do not appear in the user's information.
If the question is not in Vietnamese, translate the question to Vietnamese before generating the search query.

Search query:
"""
DETECT_LANGUAGE_AND_SUMMARY_PROMPT_TEMPLATE = """
Given the chat history and user question extract keywords and generate a search query that will return the best answer from the knowledge base.
Try to detect language and generate a grammatical sentence for the search query.
Do NOT use quotes and avoid other search operators.
Do not include cited source filenames and document names such as info.txt or doc.pdf in the search query terms.
Do not include any text inside [] or <<>> in the search query terms.
Never use acronyms or replace user content with acronyms if they do not appear in the user's information.
Extract keywords that appear in user messages, use only their original form, do not edit or add keywords that do not appear in user messages

In case the user's question is just a greeting, return an appropriate response

Reply in valid json structure like:
{
  "language": "Vietnamese",
  "vietnamese_search_query": "Summarized user question in Vietnamese",
  "keywords": ["first vietnamese keyword", "second vietnamese keyword"],
  "is_greet_intent": false,
  "greet_response": ""
}

Or in case the user's question is just a greeting, the result should be valid json as follows:
{
  "language": "Vietnamese",
  "vietnamese_search_query": "Summarized user question in Vietnamese",
  "keywords": ["first vietnamese keyword", "second vietnamese keyword"],
  "is_greet_intent": true,
  "greet_response": "Greeting response in Vietnamese"
}

YOUR HELPFUL ANSWER GOES HERE:
"""

SEARCH_PROMPT_TEMPLATE = """
Given the search query and the sources of information found, please provide helpful {target_language} answers.
In case the question is not in English, please answer in the language used to make this search query.
Search query: {user_input}

Sources of information:
{source_content}

Information sources are placed in ``` quotes, use exactly the information provided when answering, absolutely do not edit information or combine sources of information.
Only respond to the information relevant to the user's question, not to the irrelevant information
Absolutely must answer in {target_language}, do not answer in other languages.

YOUR ANSWER IN {target_language} GOES HERE:
"""
RANKING_PROMPT_TEMPLATE = """### System:
The following is a friendly conversation between a human and an AI. The AI is a good listener and can follow instruction very well. If the AI does not know the answer to a question, it truthfully says it does not know.

### User:
A list of documents is shown below. Each document has a number next to it along with content of the document. A question is also provided.
Requirement: 
1. Respond with the numbers of the documents you should consult to answer the question, in order of relevance, as well as the relevance score.
2. The relevance score is a number from 1–10 based on how relevant you think the document is to the question.
3. Do not include any documents that are not relevant to the question.

List of documents will have this format:
Document id:
<content of document id>
Example:
Document 0:
<content of document 0>\n
Document 1:
<content of document 1>\n
Document 2:
<content of document 2>\n
Document 3:
<content of document 3>\n
Document 4:
<content of document 4>\n
Document 5:
<content of document 5>\n
Document 6:
<content of document 6>\n
Question: 
<question>

Your reply MUST FOLLOW format below and DO NOT INCLUDE any additional information:
Doc: id, Relevance: Score
Example:
Doc: 0, Relevance: 7
Doc: 3, Relevance: 4
Doc: 5, Relevance: 3

Let's try this now:
{docs}

### Assistant:
"""

KEYWORD_SEARCH_TEMPLATE = """### System:
The following is a friendly conversation between a human and an AI. The AI is a good doctor and can follow instruction very well. If the AI does not know the answer to a question, it truthfully says it does not know.

### User:
Given the user Input, extract most important vietnamese keywords about disease for searching disease information from the knowledge base. Ignore name of patient, just focus on the vietnamese disease information.
Try to detect language and generate a grammatical sentence for the search query.
Never use acronyms or replace user content with acronyms if they do not appear in the user's information.
Extract keywords that appear in user messages, use only their original form, do not edit or add keywords that do not appear in user messages. If keywords are not related to medical field or diseases, do not add them to the list. ONLY keywords related to the medical field or diseases can be added

Reply in this format:
keywords: "first vietnamese keyword", "second vietnamese keyword"

In case the user's input is just a greeting, return keyword none:
keywords: "none"

Let's try this now:
Input: "{input}"

### Assistant:
"""

MULTIPLE_CHOICE_TEMPLATE = """### System:
The following is a friendly conversation between a human and an AI. The AI is a good doctor and can follow instruction very well.

### User:
Given Vietnamese multiple choices Question and Context, each question has from 2 to 6 options, of which at least one is correct. Use the provided context to choose the correct answer
Your response must only contains your choices, do not explain or add additional information.
ONLY use the contexts provided to choose answer. If context is not enough to answer or no context is provided, you are free to choose between the given options.
YOU MUST USE FORMAT BELOW, DO NOT VIOLATE:
Context:
<Các triệu chứng của bệnh van tim thường thấy ở bệnh nhân là: khó thở, chóng mặt>

Question: Đâu là triệu chứng của bệnh van tim?
A. Vàng da
B. Rụng tóc
C. Khó thở
D. chóng mặt

Answer:
C, D



Let's try this now:
Context:
{context}

Question: {question}

Answer:
### Assistant:
"""
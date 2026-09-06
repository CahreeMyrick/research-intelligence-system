## Indexing Service

**Overview:** remember, the overall goal of the system is to allow users to upload documents, 
interact with a chatbot to ask questions about any given document, and recieve intelligent answers
grounded by textual evidence. When a user sends their question, we normalize this question, using
our **Question** schema, embed the query: the text is converted into an embedding vector, and 
search the vector database for for the most similar chunks and retrieve them. The retrived chunks 
then get inserted directly into the prompt sent to the llm. So the ingestion service does the
following:xxxxxx, xxxx, xxxx,,, 

so when we say *indexing* we are referring to the following pipeline:

{

user_query: [chat-interface]

chat-interface: [**build** structured query, **send** structured query to **RetrievalService**]

retrievalservice: [**search** vector database, **send** structured **RetrievalResponse**]






}

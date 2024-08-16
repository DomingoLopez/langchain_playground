import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore

from langchain import hub   
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

from langchain_core.runnables import RunnablePassthrough

load_dotenv()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)



if __name__ == "__main__":
    
    # ################################################################
    # VERSIÓN 1 - RAG con chains 
    
    # print("No vector look up: Retrieving...")
    
    # embeddings = OpenAIEmbeddings()
    # llm = ChatOpenAI()
    
    # # Si usamos la query sin retrieval, sin nuestra info, nos da información
    # # de que pinecone es un tipo de dato que se usa en clustering jerárquico
    # # un árbol vaya.
    # query = "what is Pinecone in machine learning?"
    # chain = PromptTemplate.from_template(template=query) | llm
    # result = chain.invoke(input={})
    # print(result.content)
    
    # print("\nSi vector look up: Retrieving...")
    # # Ahora probamos a hacer retrieval
    # vectorStore = PineconeVectorStore(
    #     index_name=os.environ["INDEX_NAME"], embedding=embeddings
    # )
    
    # # Esto es un prompt que indica "usa solamente como contexto a la pregunta la info que te paso"
    # # Así evitamos hallucinations etc, pero hay que tener cuidado porque no siempre funciona bien.
    # # Mirar en langchain hub diferentes prompts.
    # retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    
    # # Esto devuelve una cadena (chain), pero coge todos los documentos, y los concatena para meterlos en nuestro prompt.
    # combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    
    # # Cadena final: prompt (buscando los documentos y embebiéndolos en la etiqueta <content> del prompt)| llm 
    # retrieval_chain = create_retrieval_chain(
    #     retriever=vectorStore.as_retriever(), combine_docs_chain=combine_docs_chain
    # )
    
    # result = retrieval_chain.invoke({"input":query})
    
    # print(result['answer'])
    
    
    # ################################################################
    # VERSIÓN 2 - RAG con LCEL, que es un formato más ligero 
    # Es mucho más simple que lo anterior que tiene varias funciones raras
    # Es prácticamente igual vaya.
    
    query = "what is Pinecone in machine learning?"
    
    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI()
    vectorStore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"], embedding=embeddings
    )
    
    
    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    Always say "thanks for asking!" at the end of the answer. 
    
    <context>
    {context}
    </context>
    
    Question: {question}
    
    Helpful Answer:"""
    
    custom_rag_prompt = PromptTemplate.from_template(template)
    
    rag_chain = (
        {"context": vectorStore.as_retriever() | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
    )
    
    res = rag_chain.invoke(query)
    print(res)
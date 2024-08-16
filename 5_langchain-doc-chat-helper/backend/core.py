import os
from dotenv import load_dotenv
from langchain.chains.retrieval import create_retrieval_chain

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import warnings

warnings.filterwarnings("ignore")
load_dotenv()



def run_llm(query: str):
    # modelo embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    # Retrieval de la bbdd vectorial
    vectorStore = PineconeVectorStore(index_name=os.getenv("INDEX_NAME"), embedding=embeddings)
    # llm
    llm = ChatOpenAI(verbose=True, temperature=0)
    # prompt para el retrieval
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    # cadena con llm y el prompt
    stuff_documents_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    # creamos la cadena
    qa = create_retrieval_chain(
        retriever=vectorStore.as_retriever(), combine_docs_chain=stuff_documents_chain
    )
    
    result = qa.invoke({"input":query})
    new_result = {
        "query": result["input"],
        "result": result["answer"],
        "source_documents": result["context"]
    }
    return new_result
    
    # Así también podría ser en lenguaje LCEL
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    # llm = ChatOpenAI(verbose=True, temperature=0)
    # vectorStore = PineconeVectorStore(
    #     index_name=os.environ["INDEX_NAME"], embedding=embeddings
    # )
    
    # retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat") 
   
    # def format_docs(docs):
    #   return "\n\n".join(doc.page_content for doc in docs)

   
    # rag_chain = (
    #     {"context": vectorStore.as_retriever() | format_docs, "input": RunnablePassthrough()}
    #     | retrieval_qa_chat_prompt
    #     | llm
    # )
    
    # res = rag_chain.invoke(query)
    # print(res)
    




if __name__ == "__main__":
    res = run_llm(query="What is a LangChain Runnable?")
    print(res["result"])
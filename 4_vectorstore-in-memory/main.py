import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS

from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
import warnings

warnings.filterwarnings("ignore")

load_dotenv()





if __name__ == "__main__":
    print("Loading pdf")
    
    pdf_path = "C:\\Users\\U971574\\Documents\\2_LANGCHAIN-COURSE\\4_vectorstore-in-memory\\2210.03629v3.pdf"
    loader = PyPDFLoader(file_path=pdf_path)
    document = loader.load()
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    docs = text_splitter.split_documents(documents=document)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index_react")
    
    new_vectorstore = FAISS.load_local(
        "faiss_index_react", embeddings,
        allow_dangerous_deserialization=True
    )
    
    # plantilla prompt
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    # llm
    llm = ChatOpenAI()
    # cadena del llm y el prompt
    combine_docs_chain = create_stuff_documents_chain(
       llm, retrieval_qa_chat_prompt
    )
    # cadena del retrieval
    retrieval_chain = create_retrieval_chain(
        new_vectorstore.as_retriever(), combine_docs_chain
    )
    # Es como que la cadena final sería:
    # 1. vectorstore obtiene documentos, luego combine_docs, que llama al llm con sun prompt
    
    
    # Esto sería sin aportar context
    # query = "Give me the gist of ReAct in 3 sentences"
    # chain = PromptTemplate.from_template(template=query) | llm
    # result = chain.invoke(input={})
    # print(result.content)

    
    # Esto sería aportando contexto
    res = retrieval_chain.invoke({"input":"Give me the gist of ReAct in 3 sentences"})
    print(res["answer"])    
    
    
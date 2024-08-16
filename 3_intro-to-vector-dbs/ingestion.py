import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore



load_dotenv()




if __name__ == "__main__":
    
    print("Ingesting...")
    loader = TextLoader("C:\\Users\\U971574\\Documents\\2_LANGCHAIN-COURSE\\intro-to-vector-dbs\\mediumblog1.txt", encoding="utf-8")
    try:
        document = loader.load()
        print("Ingestando en UTF8....\n")
    except UnicodeDecodeError:
        print("Error decoding with utf-8, trying with a different encoding...")
        print("Ingestando en latin-1....\n")
        loader = TextLoader("C:\\Users\\U971574\\Documents\\2_LANGCHAIN-COURSE\\intro-to-vector-dbs\\mediumblog1.txt", encoding="latin-1")  # Prueba otra codificación
        document = loader.load()
        
    
    print("Splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)  # Ejemplo de parámetros
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks\n")
    
    
    print("Embeddings...")
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    
    print("\nIndexando en BBDD Vectorial...")
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ['INDEX_NAME'])
    print("\nFinishing...")
    
    
## FALTA HACER LA INGESTA CON FIRECRAWL porque da mejores resultados.
# Hacemos un buen scrapeo así.

# Archivo de ingestión de la documentación 
# Descarga de la documentación con: 
# !wget -r -A.html -P langchain-docs https://langchain.readthedocs.io/en/latest/
# o si no funciona podría ser:
# https://api.python.langchain.com/en/latest/langchain_api_reference.html
# o alguna de estas:

# wget -r -A.html -P langchain-docs https://api.python.langchain.com/en/v0.1/api_reference.html
# !wget -r -A.html -P langchain-docs https://api.python.langchain.com/en/v0.1/api_reference.html
# wget -r --no-parent --html-extension -P langchain-docs https://api.python.langchain.com/en/v0.1/api_reference.html
# wget -r -A.html -P langchain-docs https://api.python.langchain.com/en/v0.1/api_reference.html

# Mirar también el archivo donwload_docs.py que hace scrapping con beautifulsoup para extraer la documentación de langchain

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import warnings
warnings.filterwarnings('ignore')



def ingest_docs():
    loader = ReadTheDocsLoader("langchain-docs/api.python.langchain.com/en/latest", encoding="utf-8")
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 600, chunk_overlap = 50)
    documents = text_splitter.split_documents(raw_documents)
    
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    # Para quitar la ruta del fichero y poner la ruta web que debe tener
    # el metadato, con https delante, etc.
    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs","https:/")
        doc.metadata.update({"source": new_url})
    
    print(f"Going to add {len(documents)} to Pinecone")
    PineconeVectorStore.from_documents(
        documents, embeddings, index_name="langchain-doc-index"
    )
    
    print("****Loading to vectorstore done ***")
    
    

if __name__ == "__main__":
    print("Ingestando...")
    ingest_docs()
    
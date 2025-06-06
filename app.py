from dotenv import load_dotenv
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PlaywrightURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant



load_dotenv()

page_urls=["https://chaidocs.vercel.app/youtube/getting-started", 
           "https://chaidocs.vercel.app/youtube/chai-aur-html/welcome/",
           "https://chaidocs.vercel.app/youtube/chai-aur-html/introduction/",
           "https://chaidocs.vercel.app/youtube/chai-aur-html/emmit-crash-course/",
           "https://chaidocs.vercel.app/youtube/chai-aur-html/html-tags/",
           "https://chaidocs.vercel.app/youtube/chai-aur-git/welcome/",
           "https://chaidocs.vercel.app/youtube/chai-aur-git/introduction/",
           "https://chaidocs.vercel.app/youtube/chai-aur-git/terminology/",
           "https://chaidocs.vercel.app/youtube/chai-aur-git/behind-the-scenes/",
           "https://chaidocs.vercel.app/youtube/chai-aur-git/branches/",
           "https://chaidocs.vercel.app/youtube/chai-aur-git/diff-stash-tags/",
           "https://chaidocs.vercel.app/youtube/chai-aur-git/managing-history/",
           "https://chaidocs.vercel.app/youtube/chai-aur-git/github/"       
]



loader=PlaywrightURLLoader(urls=page_urls)

docs = loader.load()

# print(docs[0])

# chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,     
    chunk_overlap=200    
)

split_docs = splitter.split_documents(docs)


# print(split_docs[4].page_content[:500])
# print(split_docs[4].metadata)         

# vector embeddings banana hai

# embedding_model=OpenAIEmbeddings(
#     model="text-embedding-3-large",
    
# )
# 1. Load embeddings
embedding = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=os.getenv("OPENAI_API_KEY"))

# 2. Qdrant client (local Docker)
qdrant = Qdrant.from_documents(
    documents=split_docs,  # Your chunked docs
    embedding=embedding,
    location="http://localhost:6333",  # Docker Qdrant URL
    collection_name="chai-docs",
)

print("Indexing of document done!")



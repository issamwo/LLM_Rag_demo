from pymongo import MongoClient
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.document_loaders import DirectoryLoader
from langchain.llms import openai
from langchain.chains import RetrievalQA

import gradio as gr
from gradio.themes.base import Base
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MONGO_URI = os.getenv('MONGO_URI')


client = MongoClient(MONGO_URI)
dbName = 'llm_rag_demo'
collectionName = 'collection_of_text_blobs'
collection = client[dbName][collectionName]

loader = DirectoryLoader('./sample_files', glob='./*.txt', show_progress=True)
data = loader.load()

embeddings = OpenAIEmbeddings()

vectorStore= MongoDBAtlasVectorSearch.from_documents(
    documents=data,
    embedding=embeddings,
    collection=collection,
    index_name="default"  # Use a predefined index name
)

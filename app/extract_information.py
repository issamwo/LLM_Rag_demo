from pymongo import MongoClient
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.document_loaders import DirectoryLoader
from langchain.llms import OpenAI
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

embeddings = OpenAIEmbeddings()

vectorStore= MongoDBAtlasVectorSearch(
    embedding=embeddings,
    collection=collection
)

def query_data(query):
    docs = vectorStore.similarity_search(query, K=1)
    as_output = docs[0].page_content

    llm = OpenAI( temperature=0)
    retriever = vectorStore.as_retriever()
    qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)
    retriever_output = qa.run(query)

    return as_output, retriever_output


with gr.Blocks(theme=Base(), title="question answering app using vector search") as demo:
    gr.Markdown(
        """
        # QA avec du RAG et vector search
        """
        )
    textbox = gr.Textbox(label="Posez votre question: ")
    with gr.Row():
        button = gr.Button("Envoyer", variant='primary')
    with gr.Column():
        output1 = gr.Textbox(lines=1, max_lines=10, label="reponse en se basant que sur Atlas Vector search: ")
        output2 = gr.Textbox(lines=1, max_lines=10, label="reponse en se basant que sur Atlas Vector search + LLM: ")
    
    button.click(query_data, textbox, outputs=[output1, output2])

demo.launch()
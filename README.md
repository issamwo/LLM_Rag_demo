# Rag with vector search and OpenAi

Create vector embeddings with machine learning models like OpenAI and store index them in Atlas for retrieval augmented generation (RAG), and use Gradio for user interface.

the goal is to minimize hallucination problems of llm's by adding knowledge and also control the Answers of the models.

(We compare also the answer based only vector search (similarity) and with OpenAI api.

- When the answer is in the docs/Knowledge :

![Capture d'écran 2024-01-02 171457](https://github.com/issamwo/LLM_Rag_demo/assets/120108637/7472a329-1350-4d74-989e-c143a68ffa52)

- When the question is not related to the docs/knowledge :
  
  ![Capture d'écran 2024-01-02 171650](https://github.com/issamwo/LLM_Rag_demo/assets/120108637/61bee728-3c4b-403d-98f4-f2ffda15eb9a)

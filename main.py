# -*- coding: utf-8 -*-

from sentence_transformers import SentenceTransformer
import faiss
import requests
import json

def query_ollama(model: str, prompt: str) -> str:
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        return response.json()["response"]
    else:
        raise Exception(f"Ollama API error: {response.status_code} - {response.text}")

def main():
    # Small knowledge base as a list of documents (like a mini database)
    documents = [
        {
            "id": 1,
            "title": "Artificial Intelligence in Business",
            "domain": "Technology",
            "content": "Artificial intelligence is transforming the way businesses operate worldwide."
        },
        {
            "id": 2,
            "title": "Quantum Computing Basics",
            "domain": "Technology",
            "content": "Quantum computing promises to solve problems beyond the reach of classical computers."
        },
        {
            "id": 3,
            "title": "Vaccine Impact",
            "domain": "Medicine",
            "content": "Vaccines have been instrumental in reducing infectious diseases."
        },
        {
            "id": 4,
            "title": "Heart Disease Overview",
            "domain": "Medicine",
            "content": "Heart disease remains the leading cause of death globally."
        },
        {
            "id": 5,
            "title": "Climate Change Effects",
            "domain": "Environment",
            "content": "Climate change impacts include rising sea levels and extreme weather events."
        },
        {
            "id": 6,
            "title": "Renewable Energy",
            "domain": "Environment",
            "content": "Renewable energy sources like solar and wind help reduce carbon emissions."
        },
        {
            "id": 7,
            "title": "The Renaissance",
            "domain": "History",
            "content": "The Renaissance was a cultural movement that profoundly affected European intellectual life."
        },
        {
            "id": 8,
            "title": "Industrial Revolution",
            "domain": "History",
            "content": "The Industrial Revolution began in the 18th century and changed manufacturing processes."
        },
        {
            "id": 9,
            "title": "Global Popularity of Football",
            "domain": "Sports",
            "content": "Football, also known as soccer, is the most popular sport worldwide."
        },
        {
            "id": 10,
            "title": "The Olympic Games",
            "domain": "Sports",
            "content": "The Olympics are held every four years featuring summer and winter sports."
        }
    ]

    # Extract only the contents for embedding
    contents = [doc["content"] for doc in documents]

    # Load the model and encode the document contents
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    doc_embeddings = embedder.encode(contents, convert_to_numpy=True)

    # FAISS index
    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(doc_embeddings)

    print("Enter a question or type 'exit' to quit.")

    while True:
        question = input("\nYour question: ").strip()
        if question.lower() == "exit":
            print("Exiting program.")
            break

        # Encode the question
        query_embedding = embedder.encode([question], convert_to_numpy=True)

        # Search in documents
        top_k = 1
        _, indices = index.search(query_embedding, top_k)
        retrieved_docs = [documents[i] for i in indices[0]]

        # Build context with title, domain and content for clarity
        context = "\n\n".join(
            [f"Title: {doc['title']}\nDomain: {doc['domain']}\nContent: {doc['content']}" for doc in retrieved_docs]
        )


        prompt = f"""
        You are a smart assistant. You must answer the following question strictly using only the information provided in the context, 
        even if the information is incorrect, incomplete, or inconsistent. 
        Do not use any external knowledge. Do not correct or validate the content. 
        Your answer must be fully based on the context as it is.

        Context:
        {context}

        Question:
        {question}
        """

        try:
            response = query_ollama("llama3", prompt)
            print("\n===== ANSWER =====\n")
            print(response)
        except Exception as e:
            print(f"Error querying Ollama: {e}")

if __name__ == "__main__":
    main()

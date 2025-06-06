from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI

load_dotenv()
client = OpenAI()

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

vector_db = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",  
    collection_name="chai-docs", 
    embedding=embedding_model
)

BASE_SYSTEM_PROMPT = """
You are a helpful assistant answering user queries based only on the Chai aur Code documentation.
Use only the provided context and do not make up answers.
Always cite the source URL where the user can read more.
"""

print("🧠 Ask your questions about Chai aur Code Docs! Type 'exit' to quit.\n")

# Chat history
messages = [{"role": "system", "content": BASE_SYSTEM_PROMPT}]

while True:
    try:
        query = input("> ").strip()
        if query.lower() in ["exit", "quit"]:
            print("👋 Exiting. Have a great day!")
            break

        # Search Qdrant DB
        search_results = vector_db.similarity_search(query=query, k=5)

        # context...
        context = "\n\n".join([
            f"Content: {r.page_content}\nSource URL: {r.metadata.get('source', 'N/A')}"
            for r in search_results
        ])

        system_prompt_with_context = BASE_SYSTEM_PROMPT + f"\n\nContext:\n{context}"

        # Assemble chat input (prepend updated system message)
        messages = [{"role": "system", "content": system_prompt_with_context}] + messages[1:]
        messages.append({"role": "user", "content": query})

       
        chat_completion = client.chat.completions.create(
            model="gpt-4-1106-preview", 
            messages=messages
        )

        reply = chat_completion.choices[0].message.content.strip()
        print(f"\n🤖 {reply}\n")

        messages.append({"role": "user", "content": query})
        messages.append({"role": "assistant", "content": reply})

    except Exception as e:
        print(f"❌ Error: {e}\n")

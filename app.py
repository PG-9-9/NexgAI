import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

# Load environment variables from .env file (for local development)
load_dotenv()

# Fetch environment variables with validation
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

if not PINECONE_API_KEY or not OPENAI_API_KEY:
    raise ValueError("Missing required environment variables: PINECONE_API_KEY and/or OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Initialize embeddings
embeddings = download_hugging_face_embeddings()

# Pinecone index setup
index_name = "medicalbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Initialize LLM
llm = OpenAI(temperature=0.4, max_tokens=500)

# Initialize memory to store conversation history
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Update prompt to include chat history
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),  # Add history here
        ("human", "{input}"),
    ]
)

# Create RAG chain with memory
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    # Clear memory on page load (optional, remove if you want persistent history across sessions)
    memory.clear()
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print(f"Input: {msg}")

    # Load existing chat history
    chat_history = memory.load_memory_variables({})["chat_history"]

    # Invoke the chain with input and history
    response = rag_chain.invoke({
        "input": msg,
        "chat_history": chat_history
    })
    answer = response["answer"]
    print(f"Response: {answer}")

    # Save the new message and response to memory
    memory.save_context({"input": msg}, {"output": answer})

    return str(answer)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)#Works out
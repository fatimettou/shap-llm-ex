__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
from langchain_community.vectorstores import Chroma  # Using langchain_community
from langchain.embeddings.openai import OpenAIEmbeddings  # Ensure compatibility
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
import os

# OpenAI API Key setup
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

LANGCHAIN_PROJECT = "SHAP-LLM-Telco-Local-Explanations"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Function to load documents
def load_documents():
    text_files = [
        "documents/data_dictionary.txt",
        "documents/data_summary.txt",
        "documents/shap_summary.txt",
        "documents/shap_explanation.txt"
    ]
    docs = [TextLoader(url).load() for url in text_files]
    docs_list = [item for sublist in docs for item in sublist]

    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=50
    )
    documents = text_splitter.split_documents(docs_list)
    return documents

# Create vector database and add documents
def create_vectorstore(documents):
    embeddings = OpenAIEmbeddings()
    persist_directory = "chroma_persist"

    # Remove the old persistence directory if it exists
    if os.path.exists(persist_directory):
        import shutil
        shutil.rmtree(persist_directory)

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="churn-rag-chroma-1",
        persist_directory=persist_directory,  # Use persistent directory
    )
    return vectorstore

# Set up the chatbot using OpenAI chat model (like GPT-3.5-turbo)
def setup_chatbot(vectorstore):
    template = """
    You are an assistant to customer service agents. Answer the question based on the context below to help the agent.

    Context: {context}

    Question: {question}

    If the context contains details about SHAP analysis or important features, provide the SHAP values and rank the most important features as per the SHAP analysis.
    """
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
    chain = LLMChain(llm=model, prompt=prompt)
    return chain

# Chatbot page with suggested questions
def chatbot_page():
    st.title("Chat with Your Model Based on SHAP Explanations")

    # Load and process documents
    documents = load_documents()

    # Create vectorstore (in-memory mode or persistent storage can be configured)
    persist_directory = "chroma_persist"  # Set this to None for in-memory mode
    vectorstore = create_vectorstore(documents)

    # Set up the chatbot
    chatbot_chain = setup_chatbot(vectorstore)

    # Maintain chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Display previous chat history
    for message in st.session_state["chat_history"]:
        if message["role"] == "user":
            st.chat_message(message["role"]).markdown(f"**You:** {message['content']}")
        else:
            st.chat_message(message["role"]).markdown(f"**Chatbot:** {message['content']}")

    # Input area for user questions
    user_question = st.chat_input("Ask a question about the SHAP analysis or the data:")

    # Add suggested questions in a grid-like format
    st.write("### Suggested Questions:")
    cols = st.columns(4)
    with cols[0]:
        if st.button("What are the most important features?"):
            user_question = "What are the most important features?"
    with cols[1]:
        if st.button("What are the SHAP values for customer X?"):
            user_question = "What are the SHAP values for customer X?"
    with cols[2]:
        if st.button("How is feature Y affecting predictions?"):
            user_question = "How is feature Y affecting predictions?"
    with cols[3]:
        if st.button("What is the global importance of each feature?"):
            user_question = "What is the global importance of each feature?"

    # If the user provides a question
    if user_question:
        # Store the user input in the chat history
        st.session_state["chat_history"].append({"role": "user", "content": user_question})
        st.chat_message("user").markdown(f"**You:** {user_question}")

        # Retrieve context from vectorstore
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Limit to top 3 results
        context_docs = retriever.get_relevant_documents(user_question)
        context = "\n".join([doc.page_content for doc in context_docs])

        # Generate chatbot response
        response = chatbot_chain.run(context=context, question=user_question)

        # Store the chatbot response in the chat history
        st.session_state["chat_history"].append({"role": "bot", "content": response})
        st.chat_message("bot").markdown(f"**Chatbot:** {response}")

# Run the Streamlit page
if __name__ == "__main__":
    chatbot_page()

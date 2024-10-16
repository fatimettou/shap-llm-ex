__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader



# OpenAI API Key setup
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGCHAIN_PROJECT = "SHAP-LLM-Telco-Local-Explanations"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def load_documents():
    text_files = [
        "documents/data_dictionary.txt",
        "documents/data_summary.txt",
        "documents/shap_summary.txt",
        "documents/shap_explanation.txt"
    ]
    docs = [TextLoader(url).load() for url in text_files]
    docs_list = [item for sublist in docs for item in sublist]
    
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=50
    )
    documents = text_splitter.split_documents(docs_list)
    return documents

def create_vectorstore(documents):
    embeddings = OpenAIEmbeddings()
    persist_directory = "chroma_persist"

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="churn-rag-chroma-1",
        persist_directory=persist_directory  # Use persistent directory
    )
    return vectorstore

def setup_chatbot(vectorstore):
    template = """
    You are an assistant to customer service agents. Answer the question based on the context below to help the agent.

    Context: {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
    chain = LLMChain(llm=model, prompt=prompt)
    return chain

def chatbot_page():
    st.title("Chat with Your Model Based on SHAP Explanations")
    
    documents = load_documents()
    
    # Create vectorstore (persistence enabled with directory)
    vectorstore = create_vectorstore(documents)
    
    chatbot_chain = setup_chatbot(vectorstore)

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for message in st.session_state["chat_history"]:
        if message["role"] == "user":
            st.chat_message(message["role"]).markdown(f"**You:** {message['content']}")
        else:
            st.chat_message(message["role"]).markdown(f"**Chatbot:** {message['content']}")

    user_question = st.chat_input("Ask a question about the SHAP analysis or the data:")

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

    if user_question:
        st.session_state["chat_history"].append({"role": "user", "content": user_question})
        st.chat_message("user").markdown(f"**You:** {user_question}")

        retriever = vectorstore.as_retriever()
        context_docs = retriever.get_relevant_documents(user_question)
        context = "\n".join([doc.page_content for doc in context_docs])

        response = chatbot_chain.run(context=context, question=user_question)
        
        st.session_state["chat_history"].append({"role": "bot", "content": response})
        st.chat_message("bot").markdown(f"**Chatbot:** {response}")

if __name__ == "__main__":
    chatbot_page()

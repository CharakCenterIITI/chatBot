import streamlit as st
import pandas as pd
from io import StringIO
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
import os
import streamlit as st

# Access OpenAI API key from Streamlit secrets
OPENAI_API_KEY = st.secrets["OPENAI"]["API_KEY"]

if not OPENAI_API_KEY:
    raise ValueError("The OpenAI API key is missing!")

# Set the OpenAI API key for the environment
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Initialize OpenAI LLM for LangChain
llm = OpenAI(temperature=0.7)

# Updated prompt template
prompt = PromptTemplate(
    input_variables=["combined_input"],
    template=(
        "You are a helpful and knowledgeable CharakBot. The user has uploaded a dataset and is asking questions about it.\n\n"
        "{combined_input}\n\n"
        "Your response:"
    ),
)

# LangChain memory for conversation context
memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# Create LangChain chain
chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

# Streamlit UI
st.title("Charak Digital healthcare Masterclass on Artificial Intelligence ")
st.write("Upload a CSV file and ask me anything about the dataset!")

# Session state for chat history and file
if "dataset_summary" not in st.session_state:
    st.session_state.dataset_summary = None
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# File uploader
st.subheader("Step 1: Upload Your Dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file:
    try:
        # Read and summarize the dataset
        df = pd.read_csv(uploaded_file)
        st.session_state.uploaded_file = uploaded_file

        # Generate dataset summary
        dataset_summary = f"Columns: {list(df.columns)}\n\nSummary:\n{df.describe(include='all').to_string()}"
        st.session_state.dataset_summary = dataset_summary

        # Display the uploaded CSV file
        st.write("Here's a preview of your uploaded dataset:")
        st.dataframe(df)

    except pd.errors.EmptyDataError:
        st.error("The uploaded CSV file is empty. Please upload a valid CSV file.")
        st.session_state.uploaded_file = None
        st.session_state.dataset_summary = None

# Placeholder for chat history above the input box
chat_placeholder = st.container()

# Chat interface
st.subheader("Step 2: Ask Anything")
query = st.text_input("Type your question about the dataset and press Enter 👇")

if query and st.session_state.dataset_summary:
    # Add user query to chat history
    st.session_state.chat_history.append({"role": "user", "content": query})

    # Combine input variables
    dataset_summary = st.session_state.dataset_summary
    history = "\n".join(
        [f"User: {msg['content']}" if msg["role"] == "user" else f"CharakBot: {msg['content']}" for msg in st.session_state.chat_history]
    )
    combined_input = f"Dataset summary:\n{dataset_summary}\n\nConversation history:\n{history}\n\nUser's question:\n{query}"

    # Use LangChain to process the query
    response = chain.run({"combined_input": combined_input})

    # Add bot's response to chat history
    st.session_state.chat_history.append({"role": "bot", "content": response})

# Display chat history above the input box
with chat_placeholder:
    st.subheader("Chat History")
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        elif message["role"] == "bot":
            st.markdown(f"**CharakBot:** {message['content']}")

# Clear chat button (optional)
if st.button("Clear Chat"):
    try:
        # Check and clear chat history
        if "chat_history" in st.session_state and st.session_state.chat_history:
            st.session_state.chat_history = []
            st.success("Chat history cleared!")
        else:
            st.warning("No chat history to clear.")

        # Clear uploaded file and dataset summary if they exist
        if "uploaded_file" in st.session_state:
            st.session_state.uploaded_file = None
        if "dataset_summary" in st.session_state:
            st.session_state.dataset_summary = None

        # Rerun the app to refresh the UI
        st.experimental_rerun()
    except Exception as e:
        # Log the error to console (optional)
        print(f"An error occurred during rerun: {e}")
        # Display a user-friendly message
        st.error("An unexpected error occurred. Please try again.")

# If no file is uploaded
if not st.session_state.dataset_summary:
    st.info("Please upload a dataset file first to start chatting.")





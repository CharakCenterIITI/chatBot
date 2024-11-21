import streamlit as st
import requests
import pandas as pd
from io import BytesIO
from PIL import Image

# Backend URL for the FastAPI app
BACKEND_URL = "http://4.247.162.170:8000//dataset/query"  # Update this to your backend server URL

# Function to handle dataset query and CSV upload with error handling
def get_dataset_analysis(file, query):
    files = {"file": ("uploaded_file.csv", file, "text/csv")}
    data = {"query": query}

    try:
        # Send the file and query to the backend
        response = requests.post(BACKEND_URL, files=files, data=data)

        # Check if the request was successful
        if response.status_code == 200:
            # If the response contains a PNG image (graph)
            if 'image/png' in response.headers['Content-Type']:
                img = Image.open(BytesIO(response.content))
                return img
            else:
                # Attempt to extract 'analysis' or fallback to 'result'
                response_data = response.json()
                return response_data.get("analysis", response_data.get("result", "No analysis available"))
        else:
            st.error(f"Error: Received status code {response.status_code}")
            st.write("Response content:", response.text)
            if response.status_code == 429:
                return "Error: OpenAI API rate limit exceeded or insufficient quota."
            return "Error: Failed to retrieve analysis. Please try again later."

    except requests.RequestException as e:
        st.error("Error: Failed to connect to the backend server.")
        st.write(f"Details: {e}")
        return "Error: Connection issue with the backend server."


# Initialize Streamlit UI
st.title("Charak Digital healthcare Masterclass on Artificial Intelligence")
st.write("Upload a CSV file and ask me anything about the dataset!")

# Create a session state to hold chat history and file
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

# File uploader
st.subheader("Step 1: Upload Your Dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
if uploaded_file:
    try:
        # Store uploaded file in session state
        st.session_state.uploaded_file = uploaded_file

        # Reset file pointer and display uploaded content
        st.session_state.uploaded_file.seek(0)
        dataframe = pd.read_csv(uploaded_file)
        st.write("Here's a preview of your uploaded dataset:")
        st.dataframe(dataframe)

    except pd.errors.EmptyDataError:
        st.error("The uploaded CSV file is empty. Please upload a valid CSV file.")
        st.session_state.uploaded_file = None

# Chat interface
st.subheader("Step 2: Chat with the CharakBot")
query = st.text_input("Type your question about the dataset and press Enter 👇")

if query and st.session_state.uploaded_file:
    # Store the query in chat history
    st.session_state.chat_history.append({"role": "user", "content": query})

    # Reset file pointer before sending it to the backend
    st.session_state.uploaded_file.seek(0)

    # Get analysis from the backend
    analysis_or_img = get_dataset_analysis(st.session_state.uploaded_file, query)

    # If the result is an image (PNG graph), display it
    if isinstance(analysis_or_img, Image.Image):
        st.image(analysis_or_img, caption="Generated Graph", use_column_width=True)
    else:
        # Otherwise, display the analysis text response
        st.session_state.chat_history.append({"role": "bot", "content": analysis_or_img})

# Display chat history
st.subheader("Chat History")
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    elif message["role"] == "bot":
        st.markdown(f"**CharakBot:** {message['content']}")

# If no file is uploaded, prompt the user
if not st.session_state.uploaded_file:
    st.info("Please upload a dataset file first to start chatting.")

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.session_state.uploaded_file = None
    # Simulate a rerun by setting query parameters
    st.experimental_set_query_params()




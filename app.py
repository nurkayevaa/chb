import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from xml.etree import ElementTree as ET

# Streamlit app setup
st.title("XML File to Chatbot")
st.markdown("Upload an XML file to create a chatbot.")

# Upload XML file
uploaded_file = st.file_uploader("Upload your XML file", type=["xml"])

if uploaded_file:
    try:
        # Parse the XML file
        st.info("Parsing the XML file...")
        tree = ET.parse(uploaded_file)
        root = tree.getroot()

        # Extract data from XML
        def parse_element(element):
            """
            Recursive function to extract text content from XML elements.
            """
            if element.text and element.text.strip():
                return element.text.strip()
            else:
                return " ".join(parse_element(child) for child in element)

        data = parse_element(root)
        st.success("XML file parsed successfully!")

        # Split data into smaller chunks for embedding
        st.info("Splitting data into chunks...")
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_text(data)
        st.success(f"Data split into {len(chunks)} chunks.")

        # Embed data using LangChain
        st.info("Embedding data using LangChain...")
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(chunks, embeddings)
        st.success("Data embedded successfully!")

        # Set up the chatbot
        st.info("Setting up the chatbot...")
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(model="gpt-4"),
            retriever=retriever,
            return_source_documents=True,
        )
        st.success("Chatbot is ready!")

        # Chatbot interface
        st.markdown("### Chat with your data")
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        user_input = st.text_input("Ask a question:")
        if user_input:
            response = qa_chain({"question": user_input, "chat_history": st.session_state["chat_history"]})
            answer = response["answer"]
            st.session_state["chat_history"].append((user_input, answer))

            # Display chat history
            for i, (question, answer) in enumerate(st.session_state["chat_history"]):
                st.markdown(f"**Q{i+1}:** {question}")
                st.markdown(f"**A{i+1}:** {answer}")

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Footer
st.markdown("---")
st.markdown("Powered by LangChain, OpenAI, and Streamlit.")

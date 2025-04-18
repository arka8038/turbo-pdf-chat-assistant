import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv
import tempfile

# === Constants ===
# Define path for the persistent vector database
CHROMA_PERSIST_DIR = "./chroma_db_persistent"

# === Load environment variables ===
load_dotenv()

# === Initialize HuggingFace embeddings ===
try:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
except Exception as e:
    st.error(f"Failed to load HuggingFace Embeddings model: {e}")
    st.info("Ensure you have internet connectivity and the 'sentence-transformers' library installed.")
    st.stop()


# === Streamlit UI Setup ===
st.set_page_config(
    page_title="PDF Chat Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Initialize session state ===
if 'store' not in st.session_state:
    st.session_state.store = {}
# Remove direct vectorstore object from initial state - we will load/create it
if 'vectorstore' in st.session_state: # Clean up old state if present
     del st.session_state['vectorstore']
# Flag to track if documents have been processed *and persisted*
if 'documents_processed_persisted' not in st.session_state:
    st.session_state.documents_processed_persisted = os.path.exists(CHROMA_PERSIST_DIR)
if 'current_session_id' not in st.session_state:
    st.session_state.current_session_id = "default_session"
# Track if the vectorstore is loaded into the current run
if 'vectorstore_loaded' not in st.session_state:
     st.session_state.vectorstore_loaded = None # Will hold the Chroma object


# === Sidebar Configuration ===
with st.sidebar:
    st.title("⚙️ Settings")

    env_api_key = os.getenv("GROQ_API_KEY")
    api_key_input = st.text_input(
        "Enter your GROQ API key:",
        type="password",
        value=env_api_key if env_api_key else "",
        help="Get your key from https://console.groq.com/keys"
    )
    api_key = api_key_input if api_key_input else env_api_key

    model = st.selectbox(
        "Select Model",
        ["llama-3.3-70b-versatile", "gemma2-9b-it"],
        index=0,
        help="Select the Groq model to use."
    )

    session_id_input = st.text_input(
        "Session ID",
        value=st.session_state.current_session_id,
        help="Unique ID for this chat session."
        )
    if session_id_input != st.session_state.current_session_id:
        st.session_state.current_session_id = session_id_input
        # Clear loaded store on session change if needed, or manage per session
        st.rerun()

    session_id = st.session_state.current_session_id

    if st.button("Clear Chat History"):
        if session_id in st.session_state.store:
            st.session_state.store[session_id].clear()
        st.success(f"Chat history for session '{session_id}' cleared.")
        st.rerun()

    st.divider()

# === Main Content Area ===
st.title("PDF Chat Assistant")
st.caption("Upload PDFs, ask questions, and get answers based on the document content.")

# === File Upload Section ===
upload_expander_expanded = not st.session_state.documents_processed_persisted
with st.expander("📄 Upload and Process PDF Files", expanded=upload_expander_expanded):
    uploaded_files = st.file_uploader(
        "Upload your PDF documents",
        type="pdf",
        accept_multiple_files=True,
        key=f"file_uploader_{session_id}"
    )

    if st.button("Process Uploaded Documents", key=f"process_btn_{session_id}", disabled=not uploaded_files):
        if uploaded_files:
            # Clear existing database if replacing documents (optional, default is to add)
            # Consider adding a checkbox for this behavior if desired
            # if os.path.exists(CHROMA_PERSIST_DIR):
            #     st.warning("Clearing existing database before processing new files.")
            #     shutil.rmtree(CHROMA_PERSIST_DIR)
            #     st.session_state.documents_processed_persisted = False
            #     st.session_state.vectorstore_loaded = None

            with st.spinner("Processing documents... This may take a moment."):
                documents = []
                temp_files = []

                for uploaded_file in uploaded_files:
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            temp_path = tmp_file.name
                            temp_files.append(temp_path)

                        st.write(f"Reading {uploaded_file.name}...")
                        loader = PyPDFLoader(temp_path)
                        docs = loader.load()
                        documents.extend(docs)
                        st.write(f"-> Loaded {len(docs)} pages.")
                    except Exception as e:
                        st.error(f"Error reading {uploaded_file.name}: {e}")
                    finally:
                         # Clean up temporary files immediately after loading attempt
                        if temp_path and os.path.exists(temp_path):
                             try:
                                 os.remove(temp_path)
                             except OSError as e:
                                 st.warning(f"Could not remove temp file {temp_path}: {e}")


                if documents:
                    st.write("Splitting documents into chunks...")
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )
                    splits = text_splitter.split_documents(documents)
                    st.write(f"-> Created {len(splits)} document chunks.")

                    st.write("Embedding chunks and creating/updating persistent vector store...")
                    try:
                        # Create vector store and PERSIST it to disk
                        vs = Chroma.from_documents(
                            documents=splits,
                            embedding=embeddings,
                            persist_directory=CHROMA_PERSIST_DIR # Key change!
                        )
                        # Ensure persistence call if needed (often automatic, but can be explicit)
                        # vs.persist() # Usually not required for from_documents

                        st.session_state.vectorstore_loaded = vs # Store the active object
                        st.session_state.documents_processed_persisted = True
                        st.success("✅ Documents processed and saved successfully!")
                        st.info("You can now ask questions below.")
                        # Rerun to update UI state (e.g., hide expander, enable chat)
                        st.rerun()
                    except Exception as e:
                         st.error(f"Error creating vector store: {e}")
                         st.session_state.documents_processed_persisted = False # Mark as failed

                else:
                    st.warning("No documents were successfully loaded to process.")

                # Final check for any remaining temp files (should be empty if finally worked)
                for f_path in temp_files:
                    if os.path.exists(f_path):
                        try:
                            os.remove(f_path)
                        except OSError as e:
                             st.warning(f"Could not remove leftover temp file {f_path}: {e}")

        else:
            st.warning("Please upload at least one PDF file.")


# === Chat Interface ===
if not api_key:
    st.warning("⚠️ Please enter your GROQ API key in the sidebar to enable the chat.")
    st.stop()

# --- Try to Load Persistent Vector Store ---
# This block attempts to load the vector store if documents were processed (possibly in a previous run)
# but it's not yet loaded into the current session_state object `vectorstore_loaded`.
if st.session_state.documents_processed_persisted and not st.session_state.vectorstore_loaded:
     if os.path.exists(CHROMA_PERSIST_DIR):
          with st.spinner(f"Loading existing document database from '{CHROMA_PERSIST_DIR}'..."):
               try:
                    st.session_state.vectorstore_loaded = Chroma(
                         persist_directory=CHROMA_PERSIST_DIR,
                         embedding_function=embeddings
                    )
                    # st.info("Existing database loaded.") # Optional success message
               except Exception as e:
                    st.error(f"Failed to load existing vector store: {e}")
                    st.warning("Please try processing the documents again.")
                    st.session_state.documents_processed_persisted = False # Reset flag
     else:
          st.warning("Document database directory not found, although processing was previously indicated. Please process documents again.")
          st.session_state.documents_processed_persisted = False # Reset flag


# --- Proceed only if Vector Store is loaded ---
if not st.session_state.documents_processed_persisted or not st.session_state.vectorstore_loaded:
    st.info("👆 Upload and process PDF files (or load existing database) to start chatting.")
    st.stop()

# Initialize LLM (if we have API key and loaded vector store)
try:
    llm = ChatGroq(groq_api_key=api_key, model_name=model)
except Exception as e:
    st.error(f"Failed to initialize the language model: {e}")
    st.warning("Please ensure your GROQ API key is correct and the model is available.")
    st.stop()


# ---- Langchain RAG Chain Setup (using st.session_state.vectorstore_loaded) ----

def get_session_history(session: str) -> BaseChatMessageHistory:
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]

# Create retriever from the LOADED vector store
retriever = st.session_state.vectorstore_loaded.as_retriever(search_kwargs={"k": 4})

# 1. Contextualize Question Chain
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])
history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    contextualize_q_prompt
)

# 2. Question Answering Chain
qa_system_prompt = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Keep the answer concise and based *only* on the provided context.
If the context does not contain the answer, state that explicitly.

Context:
{context}"""
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])
Youtube_chain = create_stuff_documents_chain(llm, qa_prompt)

# 3. Combine chains for Retrieval Chain
rag_chain = create_retrieval_chain(history_aware_retriever, Youtube_chain)

# 4. Add history management
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

# ---- Chat Interaction ----

st.subheader(f"Chat with your Documents (Session: {session_id})")

current_history = get_session_history(session_id).messages
for message in current_history:
    with st.chat_message("user" if message.type == "human" else "assistant"):
        st.markdown(message.content)

if prompt := st.chat_input("Ask a question about your PDFs..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = conversational_rag_chain.invoke(
                    {"input": prompt},
                    config={"configurable": {"session_id": session_id}}
                )
                st.markdown(response["answer"])

                # # Optional: Display retrieved context
                # with st.expander("Show Retrieved Context"):
                #    for i, doc in enumerate(response["context"]):
                #       st.write(f"**Context Chunk {i+1}:** Source: {doc.metadata.get('source', 'N/A')}, Page: {doc.metadata.get('page', 'N/A')}")
                #       st.caption(doc.page_content)
                #       st.divider()

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info("Please check API key, model selection, or try rephrasing. If the error persists, consider clearing the database and reprocessing documents.")

# Footer
st.divider()
st.caption("Built with LangChain, Groq & Streamlit | Persistent DB")
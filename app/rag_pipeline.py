import os
from langchain_community.document_loaders import CSVLoader, JSONLoader
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import jq
import time
 
load_dotenv()
 
def create_rag_pipeline():
    try:
        # load embeddings
        hf_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"local_files_only": True}
        )
        if os.path.exists("faiss_store"):
            print("Vector store already exists. Loading the existing store...")
            # Try to load the vector store
            faiss_store = FAISS.load_local("faiss_store", embeddings=hf_embeddings, allow_dangerous_deserialization=True)
            print("Vector store loaded successfully.")
        else:
            jq_schema = ".[] | {instruction: .instruction, input: .input, output: .output}"
 
            medical_json_loader = JSONLoader(file_path="data/chatdoctor5k.json",
                                        jq_schema = jq_schema,
                                        text_content = False
                                        )
 
            medical_json_docs = medical_json_loader.load()
 
            medical_csv_loader = CSVLoader(file_path="data/format_dataset.csv")
            medical_csv_docs = medical_csv_loader.load()
            final_medical_docs = medical_csv_docs + medical_json_docs
 
            # Create Tokens
            recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=60,
                                                            separators = ["\n\n", "\n", " ", "", ".",",", ";"])
 
            recursive_tokens = recursive_splitter.split_documents(final_medical_docs)
 
            faiss_store = FAISS.from_documents(documents = recursive_tokens, embedding=hf_embeddings)
 
            faiss_store.save_local("faiss_store")
 
        llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash-lite",max_output_tokens = 4000,temperature = 0.7)
        retriever = faiss_store.as_retriever(search_type="mmr", search_kwargs={"k": 2})
 
        prompt = PromptTemplate(
            input_variables = ["context", "chat_history", "question"],
            template = """
            You are medical consultant with expertise in understanding doctor-patient conversations, symptom descriptions and medical
            chat transcripts. Use only the information provided from the 'Medical Care and Chats' dataset to answer the user's question.
            Stay strictly within the given chats, symptoms, diagnosis and conversation notes. If the dataset contains the relevant information
            provide a clear, short and medically accurate response. If the answer is not present in the dataset, say "The answer is not available in provided context."
 
            Context:
            {context}
 
            Chat History:
            {chat_history}
 
            Question: {question}
 
            Answer:
            """
        )
 
        # Create RAG chain using LCEL (LangChain Expression Language)
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        get_question = RunnableLambda(lambda x: x.get("question", ""))
        get_chat_history = RunnableLambda(lambda x: x.get("chat_history", ""))
        
        QA_chain = (
            {
                "context": get_question | retriever | RunnableLambda(format_docs),
                "chat_history": get_chat_history,
                "question": get_question,
            }
            | prompt
            | llm
            | StrOutputParser()
        )
 
        print("Welcome to the Medical AI Chatbot!")
 
        return QA_chain
    except Exception as e:
        print(f"An error occurred while initializing the RAG pipeline: {str(e)}")
        raise e
 
 
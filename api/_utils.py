"""
Shared utilities for Vercel Functions
"""
import os
from typing import Dict, Optional
from datetime import datetime
import uuid

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Global session storage (in-memory for serverless)
sessions = {}

def get_session(session_id: str) -> Optional[Dict]:
    """Get session data"""
    return sessions.get(session_id)

def create_session(session_id: Optional[str] = None) -> str:
    """Create new session"""
    if not session_id:
        session_id = str(uuid.uuid4())
    
    sessions[session_id] = {
        "vectorstore": None,
        "qa_chain": None,
        "memory": ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        ),
        "pdf_name": None,
        "created_at": datetime.now().isoformat(),
        "messages_count": 0
    }
    
    return session_id

async def process_pdf(session_id: str, file_path: str, filename: str) -> Dict:
    """Process PDF and create vector store"""
    try:
        # Load PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create embeddings
        embeddings = OpenAIEmbeddings()
        
        # Create vector store
        vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings
        )
        
        # Create QA chain
        llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-3.5-turbo"
        )
        
        prompt_template = f"""You are an expert on the uploaded PDF document '{filename}'.
        
Use the following context to answer the question. Answer in Korean if the question is in Korean.
If you cannot find the answer in the context, say "I couldn't find that information in the uploaded document."

Context:
{{context}}

Chat History:
{{chat_history}}

Question: {{question}}

Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "chat_history", "question"]
        )
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            ),
            combine_docs_chain_kwargs={"prompt": PROMPT},
            return_source_documents=True,
            verbose=False
        )
        
        # Update session
        session = sessions[session_id]
        session["vectorstore"] = vectorstore
        session["qa_chain"] = qa_chain
        session["pdf_name"] = filename
        
        return {
            "session_id": session_id,
            "filename": filename,
            "page_count": len(documents),
            "chunk_count": len(chunks),
            "message": f"Successfully processed {filename}"
        }
        
    except Exception as e:
        raise Exception(f"PDF processing failed: {str(e)}")
"""
Query Endpoint for Vercel
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from _utils import get_session
from pydantic import BaseModel
from typing import List, Dict

class QueryRequest(BaseModel):
    session_id: str
    question: str

class QueryResponse(BaseModel):
    answer: str
    source_documents: List[Dict] = []

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """문서에 대한 질문 처리"""
    session = get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
    
    if not session.get("qa_chain"):
        raise HTTPException(status_code=400, detail="먼저 PDF를 업로드해주세요")
    
    try:
        # QA 체인 실행
        qa_chain = session["qa_chain"]
        memory = session["memory"]
        
        # 쿼리 실행
        result = qa_chain({"question": request.question, "chat_history": memory.chat_memory})
        
        # 메시지 카운트 증가
        session["messages_count"] += 1
        
        return QueryResponse(
            answer=result["answer"],
            source_documents=[{
                "page": doc.metadata.get("page", 0),
                "content": doc.page_content[:200] + "..."
            } for doc in result.get("source_documents", [])]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

handler = app
"""
PDF Upload Endpoint for Vercel
"""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import os

from _utils import create_session, process_pdf
from pydantic import BaseModel

class UploadResponse(BaseModel):
    session_id: str
    filename: str
    page_count: int
    chunk_count: int
    message: str

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/upload", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None)
):
    """PDF 파일 업로드 및 처리"""
    # 파일 확장자 확인
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다")
    
    # 세션 생성
    if not session_id:
        session_id = create_session(session_id)
    
    # 파일 저장
    os.makedirs("/tmp/uploads", exist_ok=True)
    file_path = f"/tmp/uploads/{session_id}_{file.filename}"
    
    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # PDF 처리
        result = await process_pdf(session_id, file_path, file.filename)
        
        return UploadResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 임시 파일 정리
        if os.path.exists(file_path):
            os.remove(file_path)

handler = app
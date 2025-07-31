"""
향상된 RAG 기반 챗봇 백엔드 서버 - 문서 업로드 기능 포함
"""
import os
import logging
import shutil
from typing import List, Dict, Optional
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
import uuid
import time
import re
from collections import Counter

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import pandas as pd
import csv
import json
import PyPDF2
import pdfplumber
from typing import Tuple
import tempfile
import subprocess

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Global variables
sessions = {}  # 세션별 벡터스토어 및 QA 체인 관리
session_history = {}  # 세션별 업로드 파일 히스토리
memory_usage = {  # 메모리 사용량 추적
    "total_size_bytes": 0,
    "session_sizes": {},
    "last_cleanup": datetime.now()
}
analytics_data = {  # 분석 데이터 저장
    "total_uploads": 0,
    "total_queries": 0,
    "query_types": {},
    "response_times": [],
    "keywords": {},
    "daily_usage": {},
    "session_stats": {}
}

# Production settings
IS_PRODUCTION = os.getenv("RENDER", "false").lower() == "true"
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {".pdf", ".csv"}
MAX_SESSIONS = 100  # Limit sessions in production
MAX_MEMORY_MB = 500  # 최대 메모리 사용량 (MB)
SESSION_CLEANUP_INTERVAL = 3600  # 1시간마다 클린업
MAX_FILES_PER_SESSION = 10  # 세션당 최대 파일 수
SESSION_EXPIRE_HOURS = 24  # 세션 만료 시간

# Request/Response models
class ChatRequest(BaseModel):
    message: str = Field(..., description="사용자 메시지")
    session_id: str = Field(..., description="세션 ID")

class ChatResponse(BaseModel):
    answer: str = Field(..., description="챗봇 응답")
    sources: List[Dict] = Field(default=[], description="참조된 소스")
    session_id: str = Field(..., description="세션 ID")
    pdf_name: Optional[str] = Field(None, description="현재 로드된 파일 이름")

class UploadResponse(BaseModel):
    session_id: str = Field(..., description="세션 ID")
    filename: str = Field(..., description="업로드된 파일명")
    file_type: str = Field(..., description="파일 타입")
    pages: Optional[int] = Field(None, description="PDF 페이지 수")
    rows: Optional[int] = Field(None, description="CSV 행 수")
    columns: Optional[int] = Field(None, description="CSV 열 수")
    chunks: int = Field(..., description="생성된 청크 수")
    status: str = Field(..., description="처리 상태")

class SessionInfo(BaseModel):
    session_id: str
    pdf_name: Optional[str]

class FileHistory(BaseModel):
    filename: str
    file_type: str
    upload_time: datetime
    file_size: int
    pages: Optional[int] = None
    rows: Optional[int] = None
    columns: Optional[int] = None
    chunks: int
    metadata: Optional[Dict] = None

class SessionHistory(BaseModel):
    session_id: str
    created_at: datetime
    last_accessed: datetime
    files: List[FileHistory]
    total_queries: int = 0
    memory_size_mb: float = 0.0

class MemoryManagementRequest(BaseModel):
    action: str = Field(..., description="관리 작업: cleanup_old, cleanup_session, get_stats")
    session_id: Optional[str] = Field(None, description="특정 세션 ID (선택사항)")
    hours_old: Optional[int] = Field(24, description="정리할 세션 나이 (시간)")

class SessionHistoryResponse(BaseModel):
    sessions: List[SessionHistory]
    total_memory_mb: float
    active_sessions: int
    created_at: str
    messages_count: int

class AnalyticsResponse(BaseModel):
    total_uploads: int
    total_queries: int
    query_types: Dict[str, int]
    response_times: List[float]
    keywords: Dict[str, int]
    daily_usage: Dict[str, int]
    average_response_time: float

# Session manager class
class SessionManager:
    def __init__(self):
        self.sessions = {}
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            length_function=len
        )
    
    def create_session(self, session_id: str = None) -> str:
        """새 세션 생성"""
        if not session_id:
            session_id = str(uuid.uuid4())
        
        self.sessions[session_id] = {
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
        
        logger.info(f"새 세션 생성: {session_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """세션 정보 가져오기"""
        session = self.sessions.get(session_id)
        if session:
            # 마지막 접근 시간 업데이트
            session["last_accessed"] = datetime.now().isoformat()
        return session
    
    def cleanup_session(self, session_id: str):
        """세션 정리"""
        if session_id in self.sessions:
            # 업로드된 파일 삭제
            try:
                for file in os.listdir("uploads"):
                    if file.startswith(f"{session_id}_"):
                        os.remove(os.path.join("uploads", file))
                        logger.info(f"파일 삭제: {file}")
            except Exception as e:
                logger.error(f"파일 삭제 실패: {e}")
            
            # Vector store 메모리 해제
            if self.sessions[session_id].get("vectorstore"):
                self.sessions[session_id]["vectorstore"] = None
            
            # 세션 데이터 삭제
            del self.sessions[session_id]
            logger.info(f"세션 삭제 완료: {session_id}")
    
    def cleanup_old_sessions(self, hours: int = 24):
        """오래된 세션 자동 정리"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        sessions_to_delete = []
        
        for session_id, session_data in self.sessions.items():
            last_accessed = session_data.get("last_accessed", session_data.get("created_at"))
            if last_accessed:
                access_time = datetime.fromisoformat(last_accessed)
                if access_time < cutoff_time:
                    sessions_to_delete.append(session_id)
        
        for session_id in sessions_to_delete:
            self.cleanup_session(session_id)
        
        if sessions_to_delete:
            logger.info(f"Cleaned up {len(sessions_to_delete)} old sessions")
    
    def extract_pdf_metadata(self, file_path: str) -> Dict:
        """PDF 메타데이터 추출"""
        metadata = {}
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                metadata['pages'] = len(pdf_reader.pages)
                metadata['encrypted'] = pdf_reader.is_encrypted
                
                if pdf_reader.metadata:
                    metadata['title'] = pdf_reader.metadata.get('/Title', '')
                    metadata['author'] = pdf_reader.metadata.get('/Author', '')
                    metadata['subject'] = pdf_reader.metadata.get('/Subject', '')
                    metadata['creator'] = pdf_reader.metadata.get('/Creator', '')
                    metadata['creation_date'] = str(pdf_reader.metadata.get('/CreationDate', ''))
                    metadata['modification_date'] = str(pdf_reader.metadata.get('/ModDate', ''))
        except Exception as e:
            logger.warning(f"메타데이터 추출 실패: {str(e)}")
        
        return metadata
    
    def try_multiple_pdf_loaders(self, file_path: str, filename: str) -> Tuple[List[Document], str]:
        """여러 PDF 로더를 시도하여 가장 좋은 결과 반환"""
        documents = []
        loader_used = ""
        
        # 1. 먼저 PyPDFLoader 시도 (가장 일반적)
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            if documents and any(doc.page_content.strip() for doc in documents):
                logger.info(f"PyPDFLoader로 성공적으로 로드")
                loader_used = "PyPDFLoader"
                return documents, loader_used
        except Exception as e:
            logger.warning(f"PyPDFLoader 실패: {str(e)}")
        
        # 2. pdfplumber 시도 (표와 복잡한 레이아웃에 좋음)
        try:
            documents = []
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    tables = page.extract_tables()
                    
                    # 표 데이터를 텍스트로 변환
                    if tables:
                        for table in tables:
                            table_text = "\n"
                            for row in table:
                                table_text += " | ".join(str(cell) if cell else "" for cell in row) + "\n"
                            text += table_text
                    
                    if text.strip():
                        doc = Document(
                            page_content=text,
                            metadata={
                                "source": filename,
                                "page": page_num,
                                "loader": "pdfplumber"
                            }
                        )
                        documents.append(doc)
            
            if documents:
                logger.info(f"pdfplumber로 성공적으로 로드")
                loader_used = "pdfplumber"
                return documents, loader_used
        except Exception as e:
            logger.warning(f"pdfplumber 실패: {str(e)}")
        
        # 3. PyPDF2 직접 사용 (기본 텍스트 추출)
        try:
            documents = []
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    if text.strip():
                        doc = Document(
                            page_content=text,
                            metadata={
                                "source": filename,
                                "page": page_num,
                                "loader": "PyPDF2"
                            }
                        )
                        documents.append(doc)
            
            if documents:
                logger.info(f"PyPDF2로 성공적으로 로드")
                loader_used = "PyPDF2"
                return documents, loader_used
        except Exception as e:
            logger.warning(f"PyPDF2 실패: {str(e)}")
        
        # 모든 방법이 실패한 경우
        if not documents:
            raise Exception("모든 PDF 로더가 실패했습니다. 파일이 손상되었거나 스캔된 이미지 PDF일 수 있습니다.")
        
        return documents, loader_used

    async def process_pdf(self, session_id: str, file_path: str, filename: str) -> Dict:
        """PDF 파일 처리 및 벡터스토어 생성"""
        try:
            # PDF 메타데이터 추출
            metadata = self.extract_pdf_metadata(file_path)
            logger.info(f"PDF 메타데이터: {metadata}")
            
            # 암호화된 PDF 확인
            if metadata.get('encrypted', False):
                raise HTTPException(
                    status_code=400, 
                    detail="암호로 보호된 PDF 파일입니다. 암호를 해제한 후 업로드해주세요."
                )
            
            # 여러 로더로 PDF 로드 시도
            documents, loader_used = self.try_multiple_pdf_loaders(file_path, filename)
            
            if not documents:
                raise HTTPException(
                    status_code=400,
                    detail="PDF 파일에서 텍스트를 추출할 수 없습니다. 스캔된 이미지 PDF인 경우 OCR 처리가 필요합니다."
                )
            
            logger.info(f"PDF 로드 완료: {filename}, 페이지 수: {len(documents)}, 사용된 로더: {loader_used}")
            
            # 문서가 너무 짧은 경우 청크 크기 조정
            avg_doc_length = sum(len(doc.page_content) for doc in documents) / len(documents)
            if avg_doc_length < 500:
                # 짧은 문서는 더 작은 청크로 분할
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=200,
                    chunk_overlap=50
                )
            else:
                text_splitter = self.text_splitter
            
            # 문서 분할
            chunks = text_splitter.split_documents(documents)
            
            # 메타데이터 추가
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'chunk_id': i,
                    'source_file': filename,
                    'upload_time': datetime.now().isoformat(),
                    'loader_used': loader_used,
                    'pdf_metadata': metadata
                })
            
            logger.info(f"청크 생성 완료: {len(chunks)}개")
            
            # 청크가 너무 적은 경우 경고
            if len(chunks) < 3:
                logger.warning(f"생성된 청크가 매우 적습니다 ({len(chunks)}개). PDF 내용이 제한적일 수 있습니다.")
            
            # 벡터스토어 생성
            vectorstore = FAISS.from_documents(
                documents=chunks,
                embedding=self.embeddings
            )
            
            # QA 체인 생성
            llm = ChatOpenAI(
                temperature=0.7,
                model_name="gpt-3.5-turbo",
                streaming=False
            )
            
            # 향상된 PDF 프롬프트 템플릿
            prompt_template = self.get_enhanced_prompt_template(filename, "pdf", {"pages": len(documents)})
            
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
            
            # 세션 업데이트
            session = self.sessions[session_id]
            session["vectorstore"] = vectorstore
            session["qa_chain"] = qa_chain
            session["pdf_name"] = filename
            session["num_chunks"] = len(chunks)  # 메모리 추정을 위해 청크 수 저장
            
            # 메모리 초기화 (새 파일 업로드 시)
            session["memory"].clear()
            session["messages_count"] = 0
            
            # 결과 정보
            result_info = {
                "status": "success",
                "filename": filename,
                "file_type": "pdf",
                "pages": len(documents),
                "chunks": len(chunks),
                "loader_used": loader_used,
                "metadata": metadata,
                "file_size": os.path.getsize(file_path)
            }
            
            # 히스토리에 추가
            self.add_file_to_history(session_id, result_info)
            
            return result_info
            
        except HTTPException:
            raise  # HTTPException은 그대로 전달
        except Exception as e:
            logger.error(f"PDF 처리 중 오류: {str(e)}", exc_info=True)
            
            # 사용자 친화적인 에러 메시지
            error_message = "PDF 처리 중 오류가 발생했습니다. "
            
            if "decrypt" in str(e).lower() or "encrypt" in str(e).lower():
                error_message = "암호로 보호된 PDF 파일입니다. 암호를 해제 후 업로드해주세요."
            elif "codec" in str(e).lower() or "decode" in str(e).lower():
                error_message = "PDF 파일의 인코딩 문제가 발생했습니다. 다른 PDF 뷰어에서 열어서 다시 저장 후 업로드해보세요."
            elif "eof" in str(e).lower() or "corrupt" in str(e).lower():
                error_message = "PDF 파일이 손상되었거나 불완전합니다. 파일을 확인해주세요."
            elif "image" in str(e).lower() or "scan" in str(e).lower():
                error_message = "스캔된 이미지 PDF로 보입니다. 텍스트가 포함된 PDF를 업로드해주세요."
            else:
                error_message += "파일을 확인하고 다시 시도해주세요."
                
            raise HTTPException(status_code=500, detail=error_message)
    
    async def process_csv(self, session_id: str, file_path: str, filename: str) -> Dict:
        """CSV 파일 처리 및 벡터스토어 생성"""
        try:
            # CSV 데이터 읽기 - 다양한 인코딩 시도
            df = None
            encodings = ['utf-8-sig', 'utf-8', 'cp949', 'euc-kr', 'latin1']
            last_error = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, sep=None, engine='python')
                    logger.info(f"CSV 로드 성공 - 인코딩: {encoding}")
                    break
                except Exception as e:
                    last_error = e
                    continue
            
            if df is None:
                raise HTTPException(
                    status_code=400, 
                    detail=f"CSV 파일을 읽을 수 없습니다. 파일 형식이나 인코딩을 확인해주세요. (시도한 인코딩: {', '.join(encodings)})"
                )
            
            # 데이터 검증
            if df.empty:
                raise HTTPException(
                    status_code=400,
                    detail="CSV 파일이 비어있습니다. 데이터가 포함된 파일을 업로드해주세요."
                )
            
            if len(df.columns) == 0:
                raise HTTPException(
                    status_code=400,
                    detail="CSV 파일에 컬럼이 없습니다. 올바른 형식의 CSV 파일을 업로드해주세요."
                )
            
            logger.info(f"CSV 로드 완료: {filename}, 행: {len(df)}, 열: {df.shape[1]}")
            
            # 데이터 요약 정보 생성
            data_summary = f"""
파일명: {filename}
총 행 수: {len(df)}
총 열 수: {df.shape[1]}
컬럼명: {', '.join(df.columns.tolist())}

데이터 타입:
{df.dtypes.to_string()}

기본 통계:
{df.describe(include='all').to_string()}

첫 5행 샘플:
{df.head().to_string()}
"""
            
            # CSV 데이터를 문서로 변환
            documents = []
            
            # 1. 전체 데이터 요약 문서
            summary_doc = Document(
                page_content=data_summary,
                metadata={
                    "source": filename,
                    "type": "csv_summary",
                    "row_count": len(df),
                    "column_count": df.shape[1]
                }
            )
            documents.append(summary_doc)
            
            # 2. 각 행을 개별 문서로 변환
            for idx, row in df.iterrows():
                # 행 데이터를 텍스트로 변환
                row_text = ""
                for col in df.columns:
                    value = row[col]
                    if pd.notna(value):
                        row_text += f"{col}: {value}\n"
                
                if row_text.strip():
                    doc = Document(
                        page_content=row_text,
                        metadata={
                            "source": filename,
                            "type": "csv_row",
                            "row_index": idx,
                            "row_number": idx + 1
                        }
                    )
                    documents.append(doc)
            
            # 3. 컬럼별 분석 문서 추가
            for col in df.columns:
                col_info = f"""
컬럼명: {col}
데이터 타입: {df[col].dtype}
고유값 개수: {df[col].nunique()}
결측값 개수: {df[col].isna().sum()}
"""
                if df[col].dtype in ['int64', 'float64']:
                    try:
                        col_info += f"""
평균: {df[col].mean():.2f}
중앙값: {df[col].median():.2f}
최솟값: {df[col].min()}
최댓값: {df[col].max()}
"""
                    except:
                        # 숫자 처리 실패 시 기본 정보만 표시
                        col_info += "\n숫자 통계를 계산할 수 없습니다.\n"
                else:
                    # 문자열 컬럼의 경우 상위 빈도값 표시
                    top_values = df[col].value_counts().head(10)
                    col_info += f"""
상위 10개 값:
{top_values.to_string()}
"""
                
                col_doc = Document(
                    page_content=col_info,
                    metadata={
                        "source": filename,
                        "type": "csv_column_analysis",
                        "column_name": col
                    }
                )
                documents.append(col_doc)
            
            logger.info(f"CSV 문서 변환 완료: {len(documents)}개")
            
            # 문서 분할 (필요시)
            chunks = []
            for doc in documents:
                if len(doc.page_content) > 1000:
                    # 긴 문서는 분할
                    split_docs = self.text_splitter.split_documents([doc])
                    chunks.extend(split_docs)
                else:
                    chunks.append(doc)
            
            # 메타데이터 추가
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'chunk_id': i,
                    'source_file': filename,
                    'file_type': 'csv',
                    'upload_time': datetime.now().isoformat()
                })
            
            logger.info(f"청크 생성 완료: {len(chunks)}개")
            
            # 벡터스토어 생성
            vectorstore = FAISS.from_documents(
                documents=chunks,
                embedding=self.embeddings
            )
            
            # 향상된 프롬프트 템플릿
            enhanced_prompt_template = f"""당신은 업로드된 파일 '{filename}'의 내용을 기반으로 답변하는 AI 전문가입니다.

파일 유형: CSV 데이터
총 행 수: {len(df)}
총 열 수: {df.shape[1]}
컬럼: {', '.join(df.columns.tolist())}

다음 지침을 따라 답변해주세요:

1. **정확성**: 제공된 데이터에 기반하여 정확한 답변을 제공하세요.
2. **구체성**: 가능한 한 구체적인 숫자, 날짜, 이름 등을 포함하세요.
3. **분석적 사고**: 데이터의 패턴, 추세, 이상치 등을 파악하여 통찰력 있는 답변을 제공하세요.
4. **계산 능력**: 필요시 합계, 평균, 비율 등의 계산을 수행하세요.
5. **시각화 제안**: 데이터를 더 잘 이해할 수 있는 차트나 그래프를 제안할 수 있습니다.
6. **한국어 답변**: 모든 답변은 명확하고 이해하기 쉬운 한국어로 작성하세요.

컨텍스트:
{{context}}

대화 기록:
{{chat_history}}

질문: {{question}}

답변 시 다음 사항을 고려하세요:
- 데이터에서 직접 확인할 수 있는 정보와 추론한 정보를 구분하세요.
- 불확실한 정보는 "추정", "약", "대략" 등의 표현을 사용하세요.
- 데이터에 없는 정보는 "제공된 데이터에서 확인할 수 없습니다"라고 명시하세요.
- 가능하다면 관련된 다른 데이터 포인트도 함께 제공하세요.

답변:"""
            
            # QA 체인 생성
            llm = ChatOpenAI(
                temperature=0.3,  # CSV 데이터는 더 정확한 답변을 위해 낮은 temperature
                model_name="gpt-3.5-turbo",
                streaming=False
            )
            
            PROMPT = PromptTemplate(
                template=enhanced_prompt_template,
                input_variables=["context", "chat_history", "question"]
            )
            
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 10}  # CSV는 더 많은 context 검색
                ),
                combine_docs_chain_kwargs={"prompt": PROMPT},
                return_source_documents=True,
                verbose=False
            )
            
            # 세션 업데이트
            session = self.sessions[session_id]
            session["vectorstore"] = vectorstore
            session["qa_chain"] = qa_chain
            session["pdf_name"] = filename
            session["file_type"] = "csv"
            session["num_chunks"] = len(chunks)  # 메모리 추정을 위해 청크 수 저장
            session["data_summary"] = {
                "rows": len(df),
                "columns": df.shape[1],
                "column_names": df.columns.tolist()
            }
            
            # 메모리 초기화
            session["memory"].clear()
            session["messages_count"] = 0
            
            # 결과 정보
            result_info = {
                "status": "success",
                "filename": filename,
                "file_type": "csv",
                "rows": len(df),
                "columns": df.shape[1],
                "chunks": len(chunks),
                "file_size": os.path.getsize(file_path)
            }
            
            # 히스토리에 추가
            self.add_file_to_history(session_id, result_info)
            
            return result_info
            
        except pd.errors.EmptyDataError:
            raise HTTPException(
                status_code=400,
                detail="CSV 파일이 비어있거나 형식이 올바르지 않습니다."
            )
        except pd.errors.ParserError as e:
            raise HTTPException(
                status_code=400,
                detail=f"CSV 파일 구문 분석 오류: 파일 형식을 확인해주세요. (구분자가 쉼표인지 확인)"
            )
        except MemoryError:
            raise HTTPException(
                status_code=413,
                detail="CSV 파일이 너무 큽니다. 더 작은 파일을 업로드해주세요."
            )
        except HTTPException:
            raise  # HTTPException은 그대로 전달
        except Exception as e:
            logger.error(f"CSV 처리 중 예상치 못한 오류: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"CSV 처리 중 오류가 발생했습니다. 파일 형식을 확인하고 다시 시도해주세요."
            )
    
    async def process_file(self, session_id: str, file_path: str, filename: str) -> Dict:
        """파일 타입에 따라 적절한 처리 메서드 호출"""
        file_extension = os.path.splitext(filename)[1].lower()
        
        if file_extension == ".pdf":
            return await self.process_pdf(session_id, file_path, filename)
        elif file_extension == ".csv":
            return await self.process_csv(session_id, file_path, filename)
        else:
            raise HTTPException(status_code=400, detail=f"지원하지 않는 파일 형식: {file_extension}")
    
    def get_enhanced_prompt_template(self, filename: str, file_type: str, session_data: Dict) -> str:
        """파일 타입과 컨텍스트에 맞는 향상된 프롬프트 템플릿 생성"""
        base_template = f"""당신은 업로드된 파일 '{filename}'의 내용을 기반으로 답변하는 AI 전문가입니다.

다음 지침을 따라 최상의 답변을 제공하세요:

1. **정확성과 신뢰성**
   - 제공된 문서/데이터에 기반한 정확한 정보만 제공
   - 추측이나 가정은 명확히 구분하여 표시
   - 불확실한 정보는 "추정", "대략", "약" 등의 표현 사용

2. **포괄적이고 상세한 답변**
   - 질문의 모든 측면을 다루도록 노력
   - 관련된 배경 정보나 맥락도 함께 제공
   - 필요시 단계별 설명이나 예시 포함

3. **분석적이고 통찰력 있는 접근**
   - 단순 정보 전달을 넘어 패턴, 관계, 의미 분석
   - 데이터 간의 연결점과 시사점 도출
   - 실용적이고 actionable한 인사이트 제공

4. **명확하고 구조화된 커뮤니케이션**
   - 논리적인 구조로 답변 구성
   - 필요시 번호, 불릿포인트, 소제목 활용
   - 전문용어는 쉽게 설명하거나 정의 제공

5. **맥락 인식과 적응**
   - 이전 대화 내용을 고려한 일관된 답변
   - 질문자의 의도와 필요를 파악하여 맞춤형 답변
   - 추가 정보가 도움될 경우 proactive하게 제공

컨텍스트:
{{context}}

대화 기록:
{{chat_history}}

질문: {{question}}

답변 시 추가 고려사항:
- 답변은 한국어로 작성하되, 전문성과 친근함의 균형 유지
- 데이터/문서에서 직접 확인 가능한 사실과 추론/분석 내용을 구분
- 질문에 직접적으로 답하지 못할 경우, 관련된 유용한 정보 제공
- 복잡한 개념은 단계적으로 설명하고 필요시 비유나 예시 활용

답변:"""
        
        return base_template
    
    def chat(self, session_id: str, message: str) -> Dict:
        """채팅 응답 생성"""
        session = self.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
        
        if not session["qa_chain"]:
            raise HTTPException(status_code=400, detail="먼저 파일(PDF 또는 CSV)을 업로드해주세요")
        
        try:
            # 응답 시간 측정 시작
            start_time = time.time()
            
            # 세션 히스토리 업데이트
            if session_id in session_history:
                session_history[session_id]["last_accessed"] = datetime.now()
                session_history[session_id]["total_queries"] += 1
            
            # QA 체인 실행
            result = session["qa_chain"]({
                "question": message,
                "chat_history": session["memory"].chat_memory.messages
            })
            
            # 응답 시간 기록
            response_time = time.time() - start_time
            self.track_analytics(message, result["answer"], response_time)
            
            # 소스 문서 정리
            sources = []
            if "source_documents" in result:
                for doc in result["source_documents"][:3]:
                    sources.append({
                        "content": doc.page_content[:200] + "...",
                        "metadata": doc.metadata
                    })
            
            # 메모리에 대화 저장
            session["memory"].chat_memory.add_user_message(message)
            session["memory"].chat_memory.add_ai_message(result["answer"])
            session["messages_count"] += 1
            
            return {
                "answer": result["answer"],
                "sources": sources,
                "session_id": session_id,
                "pdf_name": session["pdf_name"]
            }
            
        except Exception as e:
            logger.error(f"채팅 처리 중 오류: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def track_analytics(self, question: str, answer: str, response_time: float):
        """분석 데이터 추적"""
        global analytics_data
        
        # 총 쿼리 수 증가
        analytics_data["total_queries"] += 1
        
        # 응답 시간 기록
        analytics_data["response_times"].append(response_time)
        # 최대 1000개까지만 저장
        if len(analytics_data["response_times"]) > 1000:
            analytics_data["response_times"] = analytics_data["response_times"][-1000:]
        
        # 오늘 날짜의 사용량 증가
        today = datetime.now().strftime("%Y-%m-%d")
        analytics_data["daily_usage"][today] = analytics_data["daily_usage"].get(today, 0) + 1
        
        # 질문 유형 분류
        query_type = self.classify_query(question)
        analytics_data["query_types"][query_type] = analytics_data["query_types"].get(query_type, 0) + 1
        
        # 키워드 추출 및 빈도 업데이트
        keywords = self.extract_keywords(question + " " + answer)
        for keyword in keywords:
            analytics_data["keywords"][keyword] = analytics_data["keywords"].get(keyword, 0) + 1
    
    def classify_query(self, question: str) -> str:
        """질문 유형 분류"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["요약", "정리", "summary", "summarize"]):
            return "Summary Requests"
        elif any(word in question_lower for word in ["기술", "technical", "구현", "코드", "방법"]):
            return "Technical Details"
        elif any(word in question_lower for word in ["분석", "analyze", "평가", "비교"]):
            return "Analysis"
        elif any(word in question_lower for word in ["정의", "뜻", "meaning", "what is", "무엇"]):
            return "Definitions"
        else:
            return "General Questions"
    
    def extract_keywords(self, text: str) -> List[str]:
        """텍스트에서 키워드 추출"""
        # 한글, 영문만 추출
        words = re.findall(r'[가-힣]+|[a-zA-Z]+', text.lower())
        # 불용어 제거
        stop_words = {"the", "is", "at", "which", "on", "and", "a", "an", "을", "를", "이", "가", "은", "는", "의", "에", "와", "과"}
        words = [w for w in words if len(w) > 2 and w not in stop_words]
        # 빈도수 상위 10개 추출
        word_counts = Counter(words)
        return [word for word, _ in word_counts.most_common(10)]
    
    def add_file_to_history(self, session_id: str, file_info: Dict) -> None:
        """세션 히스토리에 파일 정보 추가"""
        if session_id not in session_history:
            session_history[session_id] = {
                "session_id": session_id,
                "created_at": datetime.now(),
                "last_accessed": datetime.now(),
                "files": [],
                "total_queries": 0,
                "memory_size_mb": 0.0
            }
        
        # 파일 히스토리 생성
        file_history = {
            "filename": file_info["filename"],
            "file_type": file_info["file_type"],
            "upload_time": datetime.now(),
            "file_size": file_info.get("file_size", 0),
            "pages": file_info.get("pages"),
            "rows": file_info.get("rows"),
            "columns": file_info.get("columns"),
            "chunks": file_info["chunks"],
            "metadata": file_info.get("metadata")
        }
        
        # 세션당 최대 파일 수 체크
        if len(session_history[session_id]["files"]) >= MAX_FILES_PER_SESSION:
            # 가장 오래된 파일 제거
            session_history[session_id]["files"].pop(0)
        
        session_history[session_id]["files"].append(file_history)
        session_history[session_id]["last_accessed"] = datetime.now()
        
        # 메모리 사용량 업데이트
        self.update_memory_usage(session_id)
    
    def update_memory_usage(self, session_id: str) -> None:
        """세션의 메모리 사용량 업데이트"""
        if session_id in sessions:
            # 간단한 추정: 청크 수 * 평균 청크 크기
            estimated_size = 0
            session = sessions[session_id]
            if session.get("vector_store"):
                # 벡터스토어 크기 추정 (청크당 약 2KB)
                num_chunks = session.get("num_chunks", 0)
                estimated_size = num_chunks * 2048  # bytes
            
            memory_usage["session_sizes"][session_id] = estimated_size
            memory_usage["total_size_bytes"] = sum(memory_usage["session_sizes"].values())
            
            # MB로 변환하여 히스토리에 저장
            if session_id in session_history:
                session_history[session_id]["memory_size_mb"] = estimated_size / (1024 * 1024)
    
    def get_session_history(self, session_id: str = None) -> List[Dict]:
        """세션 히스토리 조회"""
        if session_id:
            return [session_history[session_id]] if session_id in session_history else []
        else:
            return list(session_history.values())
    
    def cleanup_old_sessions(self, hours: int = SESSION_EXPIRE_HOURS) -> Dict:
        """오래된 세션 정리"""
        now = datetime.now()
        cutoff_time = now - timedelta(hours=hours)
        cleaned_sessions = []
        
        # 만료된 세션 찾기
        expired_sessions = []
        for sid, history in session_history.items():
            if history["last_accessed"] < cutoff_time:
                expired_sessions.append(sid)
        
        # 세션 제거
        for sid in expired_sessions:
            self.remove_session(sid)
            if sid in session_history:
                del session_history[sid]
            if sid in memory_usage["session_sizes"]:
                del memory_usage["session_sizes"][sid]
            cleaned_sessions.append(sid)
        
        # 메모리 사용량 업데이트
        memory_usage["total_size_bytes"] = sum(memory_usage["session_sizes"].values())
        memory_usage["last_cleanup"] = now
        
        return {
            "cleaned_sessions": len(cleaned_sessions),
            "remaining_sessions": len(sessions),
            "freed_memory_mb": sum(memory_usage["session_sizes"].get(sid, 0) for sid in cleaned_sessions) / (1024 * 1024)
        }
    
    def cleanup_by_memory_limit(self) -> Dict:
        """메모리 제한에 따른 세션 정리"""
        current_memory_mb = memory_usage["total_size_bytes"] / (1024 * 1024)
        
        if current_memory_mb <= MAX_MEMORY_MB:
            return {"status": "ok", "current_memory_mb": current_memory_mb}
        
        # 가장 오래 접근하지 않은 세션부터 정리
        sorted_sessions = sorted(
            session_history.items(),
            key=lambda x: x[1]["last_accessed"]
        )
        
        cleaned_sessions = []
        for sid, history in sorted_sessions:
            if current_memory_mb <= MAX_MEMORY_MB * 0.8:  # 80%까지만 사용
                break
            
            session_size_mb = memory_usage["session_sizes"].get(sid, 0) / (1024 * 1024)
            self.remove_session(sid)
            if sid in session_history:
                del session_history[sid]
            if sid in memory_usage["session_sizes"]:
                del memory_usage["session_sizes"][sid]
            
            cleaned_sessions.append(sid)
            current_memory_mb -= session_size_mb
        
        memory_usage["total_size_bytes"] = sum(memory_usage["session_sizes"].values())
        
        return {
            "cleaned_sessions": len(cleaned_sessions),
            "current_memory_mb": current_memory_mb,
            "status": "cleaned"
        }
    
    def get_memory_stats(self) -> Dict:
        """메모리 사용 통계 반환"""
        return {
            "total_memory_mb": memory_usage["total_size_bytes"] / (1024 * 1024),
            "max_memory_mb": MAX_MEMORY_MB,
            "usage_percent": (memory_usage["total_size_bytes"] / (1024 * 1024)) / MAX_MEMORY_MB * 100,
            "active_sessions": len(sessions),
            "total_sessions": len(session_history),
            "last_cleanup": memory_usage["last_cleanup"].isoformat() if memory_usage["last_cleanup"] else None,
            "session_details": [
                {
                    "session_id": sid,
                    "memory_mb": size / (1024 * 1024),
                    "last_accessed": session_history[sid]["last_accessed"].isoformat() if sid in session_history else None
                }
                for sid, size in sorted(
                    memory_usage["session_sizes"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]  # Top 10 memory users
            ]
        }

# Create session manager
session_manager = SessionManager()

# FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 실행되는 라이프사이클 관리"""
    # 시작 시
    logger.info("향상된 RAG 챗봇 서버 시작됨")
    
    # 업로드 디렉토리 생성
    os.makedirs("uploads", exist_ok=True)
    
    yield
    
    # 종료 시
    # 업로드된 파일 정리
    if os.path.exists("uploads"):
        shutil.rmtree("uploads")
    logger.info("서버 종료됨")

app = FastAPI(
    title="Enhanced RAG Chatbot API",
    description="문서(PDF/CSV) 업로드 기능이 포함된 RAG 챗봇",
    version="2.0.0",
    lifespan=lifespan
)

# CORS 설정
allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000")
allowed_origins = [origin.strip() for origin in allowed_origins_str.split(",")]

# Production에서는 GitHub Pages URL 추가
if IS_PRODUCTION:
    allowed_origins.extend([
        "https://humanist96.github.io",
        "https://rag-web-svc.onrender.com",
        "https://rag-web-svc.vercel.app",
        "https://rag-web-svc-humanist96s-projects.vercel.app",  # 실제 배포 도메인
    ])
    # Vercel preview deployments를 위한 동적 처리
    # 와일드카드는 CORSMiddleware에서 지원하지 않으므로 allow_origin_regex 사용

# 모든 origin 허용 (개발 단계에서만 사용)
if IS_PRODUCTION:
    # Production에서도 일단 모든 origin 허용 (디버깅용)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 모든 origin 허용
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"]
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"]
    )

logger.info(f"CORS configured for {'all origins (debugging)' if IS_PRODUCTION else 'specific origins'}")

# Error handlers
@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    logger.error(f"HTTP error: {exc.status_code} - {exc.detail}")
    return await http_exception_handler(request, exc)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error: {exc}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error" if IS_PRODUCTION else str(exc)}
    )

# Periodic cleanup task
import asyncio
from typing import Optional

async def periodic_cleanup():
    """주기적인 세션 정리"""
    while True:
        try:
            await asyncio.sleep(3600)  # 1시간마다
            if IS_PRODUCTION:
                session_manager.cleanup_old_sessions(hours=12)  # 12시간 이상 된 세션 삭제
                logger.info("Periodic cleanup completed")
        except Exception as e:
            logger.error(f"Periodic cleanup failed: {e}")

# Background task
background_tasks = set()

@app.on_event("startup")
async def startup_event():
    """앱 시작 시 이벤트"""
    logger.info(f"Starting AI Nexus backend (Production: {IS_PRODUCTION})")
    if IS_PRODUCTION:
        task = asyncio.create_task(periodic_cleanup())
        background_tasks.add(task)

@app.on_event("shutdown")
async def shutdown_event():
    """앱 종료 시 이벤트"""
    logger.info("Shutting down AI Nexus backend")
    for task in background_tasks:
        task.cancel()

# API 엔드포인트
@app.get("/")
@app.head("/")
async def root():
    """헬스 체크"""
    return {
        "message": "Enhanced RAG Chatbot API is running",
        "status": "healthy",
        "environment": "production" if IS_PRODUCTION else "development",
        "sessions_count": len(session_manager.sessions),
        "version": "1.0.0"
    }

@app.get("/favicon.ico")
async def favicon():
    """파비콘 더미 응답"""
    return JSONResponse(status_code=204)  # No Content

@app.post("/test-upload")
async def test_upload(request: Request):
    """업로드 테스트 엔드포인트"""
    headers = dict(request.headers)
    logger.info(f"Test upload - Headers: {headers}")
    return {
        "message": "Upload test endpoint working",
        "headers": headers,
        "method": request.method,
        "url": str(request.url)
    }

@app.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None)
):
    """파일(PDF/CSV) 업로드 및 처리"""
    logger.info(f"Upload request received - File: {file.filename}, Session: {session_id}")
    
    # 파일이 없는 경우
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="파일이 선택되지 않았습니다. PDF 또는 CSV 파일을 선택해주세요.")
    
    # 파일 확장자 확인
    file_extension = os.path.splitext(file.filename)[1].lower()
    if not file_extension:
        raise HTTPException(status_code=400, detail="파일 확장자가 없습니다. .pdf 또는 .csv 파일을 업로드해주세요.")
    
    if file_extension not in ALLOWED_EXTENSIONS:
        logger.error(f"Invalid file type: {file.filename}")
        allowed_list = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(
            status_code=400, 
            detail=f"지원하지 않는 파일 형식입니다. 지원 형식: {allowed_list} (업로드된 파일: {file_extension})"
        )
    
    # Content-Type 검증 (추가 보안)
    content_type = file.content_type
    if file_extension == ".pdf" and content_type not in ["application/pdf", "application/x-pdf"]:
        logger.warning(f"Content-Type mismatch for PDF: {content_type}")
    elif file_extension == ".csv" and content_type not in ["text/csv", "application/csv", "text/plain", "application/vnd.ms-excel"]:
        logger.warning(f"Content-Type mismatch for CSV: {content_type}")
    
    # 파일 크기 확인
    file_content = await file.read()
    file_size = len(file_content)
    
    if file_size == 0:
        raise HTTPException(status_code=400, detail="파일이 비어있습니다. 내용이 있는 파일을 업로드해주세요.")
    
    if file_size > MAX_FILE_SIZE:
        size_mb = MAX_FILE_SIZE // 1024 // 1024
        file_size_mb = file_size // 1024 // 1024
        raise HTTPException(
            status_code=413, 
            detail=f"파일 크기가 너무 큽니다. 최대 {size_mb}MB까지 업로드 가능합니다. (현재 파일: 약 {file_size_mb}MB)"
        )
    
    # 파일명 보안 검증
    safe_filename = re.sub(r'[^\w\s.-]', '', file.filename)
    if not safe_filename:
        safe_filename = f"document_{uuid.uuid4().hex[:8]}{file_extension}"
    
    await file.seek(0)  # Reset file pointer
    
    # 세션 생성 또는 가져오기
    if not session_id or session_id not in session_manager.sessions:
        session_id = session_manager.create_session(session_id)
    
    # 파일 저장
    os.makedirs("uploads", exist_ok=True)
    file_path = f"uploads/{session_id}_{safe_filename}"
    try:
        logger.info(f"파일 저장 중: {file_path}")
        with open(file_path, "wb") as buffer:
            buffer.write(file_content)
        logger.info(f"파일 저장 완료: {file_path}, 크기: {os.path.getsize(file_path)} bytes")
        
        # 파일 처리
        result = await session_manager.process_file(session_id, file_path, file.filename)
        
        # 업로드 통계 증가
        analytics_data["total_uploads"] += 1
        
        return UploadResponse(
            session_id=session_id,
            filename=file.filename,
            file_type=result.get("file_type", "pdf"),
            pages=result.get("pages"),
            rows=result.get("rows"),
            columns=result.get("columns"),
            chunks=result["chunks"],
            status="success"
        )
        
    except HTTPException as e:
        # HTTPException은 그대로 전달 (이미 사용자 친화적인 메시지)
        if os.path.exists(file_path):
            os.remove(file_path)
        raise
    except Exception as e:
        logger.error(f"업로드 처리 중 오류: {str(e)}", exc_info=True)
        # 오류 시 파일 삭제
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # 사용자 친화적인 에러 메시지
        error_message = "파일 처리 중 오류가 발생했습니다. "
        if "memory" in str(e).lower():
            error_message = "파일이 너무 큽니다. 더 작은 파일을 업로드해주세요."
        elif "encoding" in str(e).lower() or "decode" in str(e).lower():
            error_message = "파일 인코딩 문제가 발생했습니다. UTF-8 또는 한글 인코딩 파일을 사용해주세요."
        elif "format" in str(e).lower():
            error_message = "파일 형식이 올바르지 않습니다. PDF 또는 CSV 파일을 업로드해주세요."
        else:
            error_message += "파일을 확인하고 다시 시도해주세요."
            
        raise HTTPException(status_code=500, detail=error_message)
    
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """채팅 엔드포인트"""
    try:
        result = session_manager.chat(
            session_id=request.session_id,
            message=request.message
        )
        return ChatResponse(**result)
    except Exception as e:
        logger.error(f"채팅 처리 중 오류: {str(e)}")
        raise

@app.get("/sessions/{session_id}", response_model=SessionInfo)
async def get_session_info(session_id: str):
    """세션 정보 조회"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
    
    return SessionInfo(
        session_id=session_id,
        pdf_name=session["pdf_name"],
        created_at=session["created_at"],
        messages_count=session["messages_count"]
    )

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """세션 삭제"""
    if session_id in session_manager.sessions:
        # 업로드된 파일 삭제
        for file in os.listdir("uploads"):
            if file.startswith(session_id):
                os.remove(os.path.join("uploads", file))
        
        del session_manager.sessions[session_id]
        return {"message": f"세션 {session_id}가 삭제되었습니다"}
    else:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")

@app.get("/sessions")
async def list_sessions():
    """모든 세션 목록 조회"""
    sessions_info = []
    for session_id, session in session_manager.sessions.items():
        sessions_info.append({
            "session_id": session_id,
            "pdf_name": session["pdf_name"],
            "created_at": session["created_at"],
            "messages_count": session["messages_count"]
        })
    return {"sessions": sessions_info}

@app.get("/session-history")
async def get_session_history(session_id: Optional[str] = None):
    """세션 히스토리 조회"""
    history = session_manager.get_session_history(session_id)
    
    # datetime을 문자열로 변환
    formatted_history = []
    for h in history:
        formatted_session = {
            "session_id": h["session_id"],
            "created_at": h["created_at"].isoformat(),
            "last_accessed": h["last_accessed"].isoformat(),
            "files": [],
            "total_queries": h["total_queries"],
            "memory_size_mb": h["memory_size_mb"]
        }
        
        for f in h["files"]:
            formatted_file = {
                "filename": f["filename"],
                "file_type": f["file_type"],
                "upload_time": f["upload_time"].isoformat(),
                "file_size": f["file_size"],
                "pages": f.get("pages"),
                "rows": f.get("rows"),
                "columns": f.get("columns"),
                "chunks": f["chunks"],
                "metadata": f.get("metadata")
            }
            formatted_session["files"].append(formatted_file)
        
        formatted_history.append(formatted_session)
    
    total_memory = sum(h["memory_size_mb"] for h in history)
    
    return {
        "sessions": formatted_history,
        "total_memory_mb": total_memory,
        "active_sessions": len(session_manager.sessions)
    }

@app.post("/memory-management")
async def manage_memory(request: MemoryManagementRequest):
    """메모리 관리 작업 수행"""
    action = request.action
    
    if action == "cleanup_old":
        result = session_manager.cleanup_old_sessions(hours=request.hours_old)
        return {
            "status": "success",
            "action": action,
            "result": result
        }
    
    elif action == "cleanup_session":
        if not request.session_id:
            raise HTTPException(status_code=400, detail="세션 ID가 필요합니다")
        
        if request.session_id in session_manager.sessions:
            session_manager.remove_session(request.session_id)
            if request.session_id in session_history:
                del session_history[request.session_id]
            if request.session_id in memory_usage["session_sizes"]:
                del memory_usage["session_sizes"][request.session_id]
            
            return {
                "status": "success",
                "action": action,
                "message": f"세션 {request.session_id}가 정리되었습니다"
            }
        else:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
    
    elif action == "get_stats":
        stats = session_manager.get_memory_stats()
        return {
            "status": "success",
            "action": action,
            "stats": stats
        }
    
    else:
        raise HTTPException(status_code=400, detail=f"알 수 없는 작업: {action}")

@app.get("/memory-stats")
async def get_memory_stats():
    """메모리 사용 통계 조회"""
    stats = session_manager.get_memory_stats()
    
    # 메모리 제한 체크
    if stats["usage_percent"] > 80:
        session_manager.cleanup_by_memory_limit()
        stats = session_manager.get_memory_stats()  # 정리 후 다시 조회
    
    return stats

@app.get("/analytics", response_model=AnalyticsResponse)
async def get_analytics():
    """분석 데이터 조회"""
    # 평균 응답 시간 계산
    avg_response_time = 0
    if analytics_data["response_times"]:
        avg_response_time = sum(analytics_data["response_times"]) / len(analytics_data["response_times"])
    
    # 최근 7일간의 데이터만 반환
    today = datetime.now()
    daily_usage = {}
    for i in range(7):
        date = (today - timedelta(days=i)).strftime("%Y-%m-%d")
        daily_usage[date] = analytics_data["daily_usage"].get(date, 0)
    
    # 상위 20개 키워드만 반환
    top_keywords = dict(sorted(analytics_data["keywords"].items(), key=lambda x: x[1], reverse=True)[:20])
    
    return AnalyticsResponse(
        total_uploads=analytics_data["total_uploads"],
        total_queries=analytics_data["total_queries"],
        query_types=analytics_data["query_types"],
        response_times=analytics_data["response_times"][-100:],  # 최근 100개만
        keywords=top_keywords,
        daily_usage=daily_usage,
        average_response_time=avg_response_time
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
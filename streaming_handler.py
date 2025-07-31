"""
Enhanced Streaming Response Handler for Multiple LLM Providers
"""

import asyncio
import json
import logging
from typing import AsyncIterator, Optional, Dict, Any
from queue import Queue
import time

from langchain.callbacks.base import AsyncCallbackHandler
from langchain.schema import LLMResult

logger = logging.getLogger(__name__)

class EnhancedStreamingHandler(AsyncCallbackHandler):
    """향상된 스트리밍 핸들러 - OpenAI와 Claude 모두 지원"""
    
    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue()
        self.streaming_complete = False
        self.accumulated_content = ""
        self.token_count = 0
        self.start_time = None
        self.provider = None
        
    async def on_llm_start(self, serialized: Dict[str, Any], prompts: list[str], **kwargs) -> None:
        """LLM 시작 시 호출"""
        self.start_time = time.time()
        self.streaming_complete = False
        self.accumulated_content = ""
        self.token_count = 0
        
        # 프로바이더 감지
        if serialized and "name" in serialized:
            if "openai" in serialized["name"].lower():
                self.provider = "openai"
            elif "anthropic" in serialized["name"].lower() or "claude" in serialized["name"].lower():
                self.provider = "claude"
        
        logger.info(f"스트리밍 시작 - 프로바이더: {self.provider}")
        
    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        """새 토큰이 생성될 때마다 호출"""
        if not self.streaming_complete:
            self.accumulated_content += token
            self.token_count += 1
            
            # 토큰을 JSON 형식으로 큐에 추가
            await self.queue.put(json.dumps({
                "type": "token",
                "content": token,
                "accumulated": self.accumulated_content,
                "token_count": self.token_count,
                "provider": self.provider
            }))
    
    async def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """LLM 응답 완료 시 호출"""
        self.streaming_complete = True
        
        # 완료 메시지 전송
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        await self.queue.put(json.dumps({
            "type": "complete",
            "content": self.accumulated_content,
            "token_count": self.token_count,
            "elapsed_time": elapsed_time,
            "provider": self.provider,
            "metadata": {
                "tokens_per_second": self.token_count / elapsed_time if elapsed_time > 0 else 0
            }
        }))
        
        # 종료 신호
        await self.queue.put("[DONE]")
        
    async def on_llm_error(self, error: Exception, **kwargs) -> None:
        """에러 발생 시 호출"""
        logger.error(f"스트리밍 중 에러 발생: {error}")
        await self.queue.put(json.dumps({
            "type": "error",
            "error": str(error),
            "provider": self.provider
        }))
        await self.queue.put("[DONE]")
    
    async def aiter(self) -> AsyncIterator[str]:
        """비동기 이터레이터로 스트리밍 데이터 반환"""
        while True:
            item = await self.queue.get()
            if item == "[DONE]":
                break
            yield item

class StreamingResponseFormatter:
    """스트리밍 응답 포맷터"""
    
    @staticmethod
    def format_sse_message(data: Dict[str, Any]) -> str:
        """Server-Sent Events 형식으로 메시지 포맷"""
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
    
    @staticmethod
    def format_chunk_message(content: str, metadata: Optional[Dict] = None) -> str:
        """청크 메시지 포맷"""
        message = {
            "content": content,
            "timestamp": time.time()
        }
        if metadata:
            message["metadata"] = metadata
        return StreamingResponseFormatter.format_sse_message(message)
    
    @staticmethod
    def format_error_message(error: str) -> str:
        """에러 메시지 포맷"""
        return StreamingResponseFormatter.format_sse_message({
            "type": "error",
            "error": error,
            "timestamp": time.time()
        })
    
    @staticmethod
    def format_complete_message() -> str:
        """완료 메시지 포맷"""
        return "data: [DONE]\n\n"

class ModelSpecificOptimizer:
    """모델별 스트리밍 최적화"""
    
    @staticmethod
    def get_optimal_chunk_size(provider: str, model: str) -> int:
        """모델별 최적 청크 크기 반환"""
        chunk_sizes = {
            "openai": {
                "gpt-3.5-turbo": 50,
                "gpt-3.5-turbo-16k": 100,
                "gpt-4": 30,
                "gpt-4-turbo-preview": 50
            },
            "claude": {
                "claude-3-opus-20240229": 30,
                "claude-3-sonnet-20240229": 50,
                "claude-3-haiku-20240307": 100,
                "claude-2.1": 50,
                "claude-instant-1.2": 100
            }
        }
        
        provider_sizes = chunk_sizes.get(provider, {})
        return provider_sizes.get(model, 50)  # 기본값 50
    
    @staticmethod
    def should_buffer_tokens(provider: str, model: str) -> bool:
        """토큰 버퍼링 필요 여부"""
        # Claude 모델은 더 긴 토큰을 생성하는 경향이 있어 버퍼링이 유용
        if provider == "claude":
            return True
        # GPT-4는 느리므로 버퍼링 없이 즉시 전송
        if provider == "openai" and "gpt-4" in model:
            return False
        return False

class TokenBuffer:
    """토큰 버퍼링 및 청킹"""
    
    def __init__(self, chunk_size: int = 50, buffer_time: float = 0.1):
        self.chunk_size = chunk_size
        self.buffer_time = buffer_time
        self.buffer = []
        self.last_flush = time.time()
    
    def add_token(self, token: str) -> Optional[str]:
        """토큰 추가 및 필요시 플러시"""
        self.buffer.append(token)
        
        current_time = time.time()
        should_flush = (
            len(self.buffer) >= self.chunk_size or
            (current_time - self.last_flush) >= self.buffer_time
        )
        
        if should_flush:
            return self.flush()
        return None
    
    def flush(self) -> str:
        """버퍼 플러시"""
        if not self.buffer:
            return ""
        
        content = "".join(self.buffer)
        self.buffer = []
        self.last_flush = time.time()
        return content

async def create_streaming_response(
    qa_chain,
    question: str,
    chat_history: list,
    provider: str,
    model: str
) -> AsyncIterator[str]:
    """스트리밍 응답 생성"""
    
    # 스트리밍 핸들러 생성
    handler = EnhancedStreamingHandler()
    
    # 모델별 최적화 설정
    chunk_size = ModelSpecificOptimizer.get_optimal_chunk_size(provider, model)
    should_buffer = ModelSpecificOptimizer.should_buffer_tokens(provider, model)
    
    # 토큰 버퍼 생성 (필요한 경우)
    buffer = TokenBuffer(chunk_size=chunk_size) if should_buffer else None
    
    # 비동기 태스크로 QA 체인 실행
    task = asyncio.create_task(
        qa_chain.acall({
            "question": question,
            "chat_history": chat_history
        }, callbacks=[handler])
    )
    
    formatter = StreamingResponseFormatter()
    
    try:
        async for message in handler.aiter():
            if message == "[DONE]":
                break
                
            data = json.loads(message)
            
            if data["type"] == "token":
                if buffer:
                    # 버퍼링 사용
                    content = buffer.add_token(data["content"])
                    if content:
                        yield formatter.format_chunk_message(content, {
                            "provider": provider,
                            "model": model
                        })
                else:
                    # 즉시 전송
                    yield formatter.format_chunk_message(data["content"], {
                        "provider": provider,
                        "model": model
                    })
            
            elif data["type"] == "error":
                yield formatter.format_error_message(data["error"])
                break
        
        # 버퍼에 남은 토큰 플러시
        if buffer:
            remaining = buffer.flush()
            if remaining:
                yield formatter.format_chunk_message(remaining, {
                    "provider": provider,
                    "model": model
                })
        
        # 완료 신호
        yield formatter.format_complete_message()
        
        # 태스크 완료 대기
        await task
        
    except Exception as e:
        logger.error(f"스트리밍 중 오류: {e}")
        yield formatter.format_error_message(str(e))
        yield formatter.format_complete_message()
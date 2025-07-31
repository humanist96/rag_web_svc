"""
LLM Provider Abstraction
OpenAI와 Claude API를 통합 관리하는 시스템
"""

import os
from typing import Dict, Optional, List, Any, AsyncIterator
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass
import json

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """모델 설정"""
    provider: str  # "openai" or "claude"
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 2000
    streaming: bool = True
    api_key: Optional[str] = None

class LLMProvider(ABC):
    """LLM 프로바이더 추상 클래스"""
    
    @abstractmethod
    def get_chat_model(self, config: ModelConfig, callbacks: Optional[List] = None):
        """채팅 모델 인스턴스 반환"""
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[Dict[str, str]]:
        """사용 가능한 모델 목록 반환"""
        pass
    
    @abstractmethod
    def validate_api_key(self, api_key: str) -> bool:
        """API 키 유효성 검증"""
        pass

class OpenAIProvider(LLMProvider):
    """OpenAI 프로바이더"""
    
    def __init__(self):
        self.models = [
            {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo", "description": "빠르고 효율적인 모델"},
            {"id": "gpt-3.5-turbo-16k", "name": "GPT-3.5 Turbo 16K", "description": "긴 컨텍스트 처리"},
            {"id": "gpt-4", "name": "GPT-4", "description": "가장 강력한 모델"},
            {"id": "gpt-4-turbo-preview", "name": "GPT-4 Turbo", "description": "최신 GPT-4 모델"},
        ]
    
    def get_chat_model(self, config: ModelConfig, callbacks: Optional[List] = None):
        """OpenAI 채팅 모델 반환"""
        api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        
        return ChatOpenAI(
            model_name=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            streaming=config.streaming,
            api_key=api_key,
            callbacks=callbacks or []
        )
    
    def get_available_models(self) -> List[Dict[str, str]]:
        return self.models
    
    def validate_api_key(self, api_key: str) -> bool:
        """OpenAI API 키 검증"""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            # 간단한 API 호출로 검증
            client.models.list()
            return True
        except Exception as e:
            logger.error(f"OpenAI API 키 검증 실패: {e}")
            return False

class ClaudeProvider(LLMProvider):
    """Claude (Anthropic) 프로바이더"""
    
    def __init__(self):
        self.models = [
            {"id": "claude-3-opus-20240229", "name": "Claude 3 Opus", "description": "가장 강력한 Claude 모델"},
            {"id": "claude-3-sonnet-20240229", "name": "Claude 3 Sonnet", "description": "균형잡힌 성능과 속도"},
            {"id": "claude-3-haiku-20240307", "name": "Claude 3 Haiku", "description": "빠르고 효율적인 모델"},
            {"id": "claude-2.1", "name": "Claude 2.1", "description": "이전 세대 모델"},
            {"id": "claude-instant-1.2", "name": "Claude Instant", "description": "가장 빠른 응답"},
        ]
    
    def get_chat_model(self, config: ModelConfig, callbacks: Optional[List] = None):
        """Claude 채팅 모델 반환"""
        api_key = config.api_key or os.getenv("ANTHROPIC_API_KEY")
        
        return ChatAnthropic(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            streaming=config.streaming,
            anthropic_api_key=api_key,
            callbacks=callbacks or []
        )
    
    def get_available_models(self) -> List[Dict[str, str]]:
        return self.models
    
    def validate_api_key(self, api_key: str) -> bool:
        """Claude API 키 검증"""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            # 간단한 API 호출로 검증
            client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}]
            )
            return True
        except Exception as e:
            logger.error(f"Claude API 키 검증 실패: {e}")
            return False

class LLMProviderFactory:
    """LLM 프로바이더 팩토리"""
    
    _providers = {
        "openai": OpenAIProvider(),
        "claude": ClaudeProvider()
    }
    
    @classmethod
    def get_provider(cls, provider_name: str) -> LLMProvider:
        """프로바이더 인스턴스 반환"""
        provider = cls._providers.get(provider_name.lower())
        if not provider:
            raise ValueError(f"지원하지 않는 프로바이더: {provider_name}")
        return provider
    
    @classmethod
    def get_available_providers(cls) -> List[str]:
        """사용 가능한 프로바이더 목록"""
        return list(cls._providers.keys())
    
    @classmethod
    def get_all_models(cls) -> Dict[str, List[Dict[str, str]]]:
        """모든 프로바이더의 모델 목록"""
        return {
            provider_name: provider.get_available_models()
            for provider_name, provider in cls._providers.items()
        }

class StreamingCallbackHandler(BaseCallbackHandler):
    """스트리밍 응답을 위한 콜백 핸들러"""
    
    def __init__(self, queue):
        self.queue = queue
        self.streaming_complete = False
    
    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        """새 토큰이 생성될 때마다 호출"""
        if not self.streaming_complete:
            await self.queue.put(json.dumps({"content": token}))
    
    async def on_llm_end(self, response, **kwargs) -> None:
        """LLM 응답 완료 시 호출"""
        self.streaming_complete = True
        await self.queue.put("[DONE]")

class LLMManager:
    """LLM 관리자 - 프로바이더와 모델을 통합 관리"""
    
    def __init__(self):
        self.current_config: Optional[ModelConfig] = None
        self.current_provider: Optional[LLMProvider] = None
    
    def set_config(self, config: ModelConfig):
        """현재 설정 업데이트"""
        self.current_config = config
        self.current_provider = LLMProviderFactory.get_provider(config.provider)
    
    def get_chat_model(self, callbacks: Optional[List] = None):
        """현재 설정에 따른 채팅 모델 반환"""
        if not self.current_config or not self.current_provider:
            # 기본값: OpenAI GPT-3.5-turbo
            self.set_config(ModelConfig(
                provider="openai",
                model_name="gpt-3.5-turbo-16k",
                temperature=0.5,
                max_tokens=2000,
                streaming=True
            ))
        
        return self.current_provider.get_chat_model(self.current_config, callbacks)
    
    def update_api_key(self, provider: str, api_key: str) -> bool:
        """API 키 업데이트 및 검증"""
        provider_instance = LLMProviderFactory.get_provider(provider)
        
        if provider_instance.validate_api_key(api_key):
            # 환경 변수 업데이트
            if provider == "openai":
                os.environ["OPENAI_API_KEY"] = api_key
            elif provider == "claude":
                os.environ["ANTHROPIC_API_KEY"] = api_key
            
            return True
        return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """현재 모델 정보 반환"""
        if not self.current_config:
            return {"provider": None, "model": None}
        
        return {
            "provider": self.current_config.provider,
            "model": self.current_config.model_name,
            "temperature": self.current_config.temperature,
            "max_tokens": self.current_config.max_tokens
        }

# 전역 LLM 매니저 인스턴스
llm_manager = LLMManager()
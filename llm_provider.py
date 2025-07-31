"""
LLM Provider Manager
Handles different LLM providers and configurations
"""

import logging
import os
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseLLMProvider(ABC):
    """Base class for LLM providers"""
    
    def __init__(self, model_name: str, provider: str, context_window: int):
        self.model_name = model_name
        self.provider = provider
        self.context_window = context_window
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from prompt"""
        pass
    
    @abstractmethod
    async def stream_generate(self, prompt: str, **kwargs):
        """Generate streaming response from prompt"""
        pass

class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM Provider"""
    
    def __init__(self, model_name: str, api_key: str):
        model_configs = {
            'gpt-3.5-turbo': {'context_window': 4096},
            'gpt-3.5-turbo-16k': {'context_window': 16384},
            'gpt-4': {'context_window': 8192},
            'gpt-4-turbo-preview': {'context_window': 128000}
        }
        
        config = model_configs.get(model_name, {'context_window': 4096})
        super().__init__(model_name, 'OpenAI', config['context_window'])
        
        self.api_key = api_key
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client"""
        try:
            import openai
            self.client = openai.AsyncOpenAI(api_key=self.api_key)
            logger.info(f"OpenAI client initialized for {self.model_name}")
        except ImportError:
            logger.error("OpenAI package not installed")
            raise
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from prompt"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 2000)
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI generation failed: {str(e)}")
            raise
    
    async def stream_generate(self, prompt: str, **kwargs):
        """Generate streaming response from prompt"""
        try:
            stream = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 2000),
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"OpenAI streaming failed: {str(e)}")
            raise

class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude LLM Provider"""
    
    def __init__(self, model_name: str, api_key: str):
        model_configs = {
            'claude-3-opus-20240229': {'context_window': 200000},
            'claude-3-sonnet-20240229': {'context_window': 200000},
            'claude-3-haiku-20240307': {'context_window': 200000},
            'claude-2.1': {'context_window': 100000},
            'claude-instant-1.2': {'context_window': 100000}
        }
        
        config = model_configs.get(model_name, {'context_window': 100000})
        super().__init__(model_name, 'Anthropic', config['context_window'])
        
        self.api_key = api_key
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Anthropic client"""
        try:
            import anthropic
            self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
            logger.info(f"Anthropic client initialized for {self.model_name}")
        except ImportError:
            logger.error("Anthropic package not installed")
            raise
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from prompt"""
        try:
            response = await self.client.messages.create(
                model=self.model_name,
                max_tokens=kwargs.get('max_tokens', 2000),
                temperature=kwargs.get('temperature', 0.7),
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Anthropic generation failed: {str(e)}")
            raise
    
    async def stream_generate(self, prompt: str, **kwargs):
        """Generate streaming response from prompt"""
        try:
            async with self.client.messages.stream(
                model=self.model_name,
                max_tokens=kwargs.get('max_tokens', 2000),
                temperature=kwargs.get('temperature', 0.7),
                messages=[{"role": "user", "content": prompt}]
            ) as stream:
                async for text in stream.text_stream:
                    yield text
                    
        except Exception as e:
            logger.error(f"Anthropic streaming failed: {str(e)}")
            raise

class LLMProviderManager:
    """Manager for different LLM providers"""
    
    def __init__(self):
        self.providers: Dict[str, BaseLLMProvider] = {}
        self.provider_configs = {
            # OpenAI Models
            'gpt-3.5-turbo': {
                'class': OpenAIProvider,
                'provider': 'OpenAI',
                'requires_api_key': True,
                'env_var': 'OPENAI_API_KEY'
            },
            'gpt-3.5-turbo-16k': {
                'class': OpenAIProvider,
                'provider': 'OpenAI',
                'requires_api_key': True,
                'env_var': 'OPENAI_API_KEY'
            },
            'gpt-4': {
                'class': OpenAIProvider,
                'provider': 'OpenAI',
                'requires_api_key': True,
                'env_var': 'OPENAI_API_KEY'
            },
            'gpt-4-turbo-preview': {
                'class': OpenAIProvider,
                'provider': 'OpenAI',
                'requires_api_key': True,
                'env_var': 'OPENAI_API_KEY'
            },
            
            # Anthropic Models
            'claude-3-opus-20240229': {
                'class': AnthropicProvider,
                'provider': 'Anthropic',
                'requires_api_key': True,
                'env_var': 'ANTHROPIC_API_KEY'
            },
            'claude-3-sonnet-20240229': {
                'class': AnthropicProvider,
                'provider': 'Anthropic',
                'requires_api_key': True,
                'env_var': 'ANTHROPIC_API_KEY'
            },
            'claude-3-haiku-20240307': {
                'class': AnthropicProvider,
                'provider': 'Anthropic',
                'requires_api_key': True,
                'env_var': 'ANTHROPIC_API_KEY'
            },
            'claude-2.1': {
                'class': AnthropicProvider,
                'provider': 'Anthropic',
                'requires_api_key': True,
                'env_var': 'ANTHROPIC_API_KEY'
            },
            'claude-instant-1.2': {
                'class': AnthropicProvider,
                'provider': 'Anthropic',
                'requires_api_key': True,
                'env_var': 'ANTHROPIC_API_KEY'
            }
        }
    
    async def get_model(self, model_name: str) -> BaseLLMProvider:
        """Get or create LLM provider instance"""
        if model_name in self.providers:
            return self.providers[model_name]
        
        if model_name not in self.provider_configs:
            raise ValueError(f"Unknown LLM model: {model_name}")
        
        config = self.provider_configs[model_name]
        provider_class = config['class']
        
        try:
            if config.get('requires_api_key'):
                api_key = self._get_api_key(config['env_var'])
                provider_instance = provider_class(model_name, api_key)
            else:
                provider_instance = provider_class(model_name)
            
            self.providers[model_name] = provider_instance
            logger.info(f"LLM provider loaded: {model_name}")
            return provider_instance
            
        except Exception as e:
            logger.error(f"Failed to load LLM provider {model_name}: {str(e)}")
            raise
    
    def _get_api_key(self, env_var: str) -> str:
        """Get API key from environment variables"""
        api_key = os.getenv(env_var)
        if not api_key:
            raise ValueError(f"API key not found in environment variable: {env_var}")
        return api_key
    
    def is_model_available(self, model_name: str) -> bool:
        """Check if a model is available"""
        if model_name not in self.provider_configs:
            return False
        
        config = self.provider_configs[model_name]
        
        # Check if API key is available (for models that require it)
        if config.get('requires_api_key'):
            return bool(os.getenv(config['env_var']))
        
        return True
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        available = []
        for model_name in self.provider_configs:
            if self.is_model_available(model_name):
                available.append(model_name)
        return available
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model"""
        if model_name not in self.provider_configs:
            return None
        
        config = self.provider_configs[model_name]
        
        return {
            'name': model_name,
            'provider': config['provider'],
            'requires_api_key': config.get('requires_api_key', False),
            'available': self.is_model_available(model_name),
            'env_var': config.get('env_var')
        }
    
    def get_models_by_provider(self, provider: str) -> List[str]:
        """Get models for a specific provider"""
        models = []
        for model_name, config in self.provider_configs.items():
            if config['provider'].lower() == provider.lower():
                models.append(model_name)
        return models

# Global instance
llm_manager = LLMProviderManager()
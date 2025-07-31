"""
Embedding Models Manager
Handles different embedding model providers and configurations
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseEmbeddingModel(ABC):
    """Base class for embedding models"""
    
    def __init__(self, model_name: str, dimensions: int, max_tokens: int):
        self.model_name = model_name
        self.dimensions = dimensions
        self.max_tokens = max_tokens
    
    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        pass
    
    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        pass

class OpenAIEmbeddingModel(BaseEmbeddingModel):
    """OpenAI Embedding Model Implementation"""
    
    def __init__(self, model_name: str, api_key: str):
        model_configs = {
            'openai-ada-002': {'dimensions': 1536, 'max_tokens': 8191},
            'openai-embedding-3-small': {'dimensions': 1536, 'max_tokens': 8191},
            'openai-embedding-3-large': {'dimensions': 3072, 'max_tokens': 8191}
        }
        
        config = model_configs.get(model_name, {'dimensions': 1536, 'max_tokens': 8191})
        super().__init__(model_name, config['dimensions'], config['max_tokens'])
        
        self.api_key = api_key
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client"""
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
            logger.info(f"OpenAI client initialized for {self.model_name}")
        except ImportError:
            logger.error("OpenAI package not installed")
            raise
    
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        try:
            # OpenAI API has batch limits, so we process in chunks
            batch_size = 100
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                # Create embeddings for batch
                response = await asyncio.to_thread(
                    self.client.embeddings.create,
                    model=self._get_api_model_name(),
                    input=batch
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            
            logger.info(f"Created {len(all_embeddings)} embeddings using {self.model_name}")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {str(e)}")
            raise
    
    async def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        try:
            response = await asyncio.to_thread(
                self.client.embeddings.create,
                model=self._get_api_model_name(),
                input=[text]
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"OpenAI query embedding failed: {str(e)}")
            raise
    
    def _get_api_model_name(self) -> str:
        """Map internal model name to OpenAI API model name"""
        mapping = {
            'openai-ada-002': 'text-embedding-ada-002',
            'openai-embedding-3-small': 'text-embedding-3-small',
            'openai-embedding-3-large': 'text-embedding-3-large'
        }
        return mapping.get(self.model_name, 'text-embedding-ada-002')

class HuggingFaceEmbeddingModel(BaseEmbeddingModel):
    """HuggingFace Sentence Transformers Implementation"""
    
    def __init__(self, model_name: str = 'huggingface-sentence-transformers'):
        super().__init__(model_name, 768, 512)
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize Sentence Transformers model"""
        try:
            from sentence_transformers import SentenceTransformer
            # Use a multilingual model for better Korean support
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            logger.info(f"HuggingFace model initialized: {self.model_name}")
        except ImportError:
            logger.error("sentence-transformers package not installed")
            raise
    
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        try:
            # Run embedding in thread to avoid blocking
            embeddings = await asyncio.to_thread(
                self.model.encode,
                texts,
                convert_to_tensor=False,
                show_progress_bar=True
            )
            
            # Convert to list of lists
            embeddings_list = [emb.tolist() for emb in embeddings]
            
            logger.info(f"Created {len(embeddings_list)} embeddings using {self.model_name}")
            return embeddings_list
            
        except Exception as e:
            logger.error(f"HuggingFace embedding failed: {str(e)}")
            raise
    
    async def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        try:
            embedding = await asyncio.to_thread(
                self.model.encode,
                [text],
                convert_to_tensor=False
            )
            
            return embedding[0].tolist()
            
        except Exception as e:
            logger.error(f"HuggingFace query embedding failed: {str(e)}")
            raise

class CohereEmbeddingModel(BaseEmbeddingModel):
    """Cohere Embedding Model Implementation"""
    
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name, 1024, 2048)
        self.api_key = api_key
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Cohere client"""
        try:
            import cohere
            self.client = cohere.Client(self.api_key)
            logger.info(f"Cohere client initialized for {self.model_name}")
        except ImportError:
            logger.error("cohere package not installed")
            raise
    
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        try:
            # Cohere has batch limits
            batch_size = 96
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                response = await asyncio.to_thread(
                    self.client.embed,
                    texts=batch,
                    model='embed-multilingual-v2.0'
                )
                
                all_embeddings.extend(response.embeddings)
            
            logger.info(f"Created {len(all_embeddings)} embeddings using {self.model_name}")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Cohere embedding failed: {str(e)}")
            raise
    
    async def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        try:
            response = await asyncio.to_thread(
                self.client.embed,
                texts=[text],
                model='embed-multilingual-v2.0'
            )
            
            return response.embeddings[0]
            
        except Exception as e:
            logger.error(f"Cohere query embedding failed: {str(e)}")
            raise

class EmbeddingModelManager:
    """Manager for different embedding model providers"""
    
    def __init__(self):
        self.models: Dict[str, BaseEmbeddingModel] = {}
        self.model_configs = {
            'openai-ada-002': {
                'class': OpenAIEmbeddingModel,
                'requires_api_key': True,
                'env_var': 'OPENAI_API_KEY'
            },
            'openai-embedding-3-small': {
                'class': OpenAIEmbeddingModel,
                'requires_api_key': True,
                'env_var': 'OPENAI_API_KEY'
            },
            'openai-embedding-3-large': {
                'class': OpenAIEmbeddingModel,
                'requires_api_key': True,
                'env_var': 'OPENAI_API_KEY'
            },
            'huggingface-sentence-transformers': {
                'class': HuggingFaceEmbeddingModel,
                'requires_api_key': False
            },
            'cohere-embed-multilingual': {
                'class': CohereEmbeddingModel,
                'requires_api_key': True,
                'env_var': 'COHERE_API_KEY'
            }
        }
    
    async def get_model(self, model_name: str) -> BaseEmbeddingModel:
        """Get or create embedding model instance"""
        if model_name in self.models:
            return self.models[model_name]
        
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown embedding model: {model_name}")
        
        config = self.model_configs[model_name]
        model_class = config['class']
        
        try:
            if config.get('requires_api_key'):
                api_key = self._get_api_key(config['env_var'])
                model_instance = model_class(model_name, api_key)
            else:
                model_instance = model_class(model_name)
            
            self.models[model_name] = model_instance
            logger.info(f"Embedding model loaded: {model_name}")
            return model_instance
            
        except Exception as e:
            logger.error(f"Failed to load embedding model {model_name}: {str(e)}")
            raise
    
    def _get_api_key(self, env_var: str) -> str:
        """Get API key from environment variables"""
        import os
        api_key = os.getenv(env_var)
        if not api_key:
            raise ValueError(f"API key not found in environment variable: {env_var}")
        return api_key
    
    def is_model_available(self, model_name: str) -> bool:
        """Check if a model is available"""
        if model_name not in self.model_configs:
            return False
        
        config = self.model_configs[model_name]
        
        # Check if API key is available (for models that require it)
        if config.get('requires_api_key'):
            import os
            return bool(os.getenv(config['env_var']))
        
        return True
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        available = []
        for model_name in self.model_configs:
            if self.is_model_available(model_name):
                available.append(model_name)
        return available
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model"""
        if model_name not in self.model_configs:
            return None
        
        config = self.model_configs[model_name]
        
        # Create temporary instance to get dimensions info
        try:
            if config.get('requires_api_key'):
                api_key = self._get_api_key(config['env_var'])
                temp_instance = config['class'](model_name, api_key)
            else:
                temp_instance = config['class'](model_name)
            
            return {
                'name': model_name,
                'dimensions': temp_instance.dimensions,
                'max_tokens': temp_instance.max_tokens,
                'requires_api_key': config.get('requires_api_key', False),
                'available': self.is_model_available(model_name)
            }
            
        except Exception:
            return {
                'name': model_name,
                'available': False,
                'error': 'Failed to initialize'
            }

# Global instance
embedding_manager = EmbeddingModelManager()
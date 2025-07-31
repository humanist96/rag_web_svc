"""
Enhanced Upload Handler with Model Selection Support
Handles file uploads with embedding model and LLM selection
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Tuple
from fastapi import UploadFile, Form, HTTPException
from pathlib import Path

from model_specific_prompts import prompt_optimizer
from embedding_models import EmbeddingModelManager
from llm_provider import LLMProviderManager

logger = logging.getLogger(__name__)

class EnhancedUploadHandler:
    """Enhanced file upload handler with model selection support"""
    
    def __init__(self):
        self.embedding_manager = EmbeddingModelManager()
        self.llm_manager = LLMProviderManager()
        self.upload_dir = Path("uploads")
        self.upload_dir.mkdir(exist_ok=True)
        
    async def handle_upload_with_models(
        self,
        file: UploadFile,
        session_id: str,
        embedding_model: str,
        llm_model: str,
        model_config: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Handle file upload with selected models
        
        Args:
            file: Uploaded file
            session_id: Session ID
            embedding_model: Selected embedding model key
            llm_model: Selected LLM model key
            model_config: JSON string with model configurations
            
        Returns:
            Upload result with model information
        """
        try:
            # Validate models
            self._validate_model_selection(embedding_model, llm_model)
            
            # Parse model config
            config = {}
            if model_config:
                try:
                    config = json.loads(model_config)
                except json.JSONDecodeError:
                    logger.warning("Invalid model_config JSON, using defaults")
            
            # Save uploaded file
            file_path = await self._save_uploaded_file(file, session_id)
            
            # Initialize models
            embedding_instance = await self.embedding_manager.get_model(embedding_model)
            llm_instance = await self.llm_manager.get_model(llm_model)
            
            # Process file based on type
            file_info = await self._process_file(file_path, file, embedding_instance)
            
            # Create optimized prompt template
            task_type = self._detect_file_task_type(file_info)
            prompt_template = prompt_optimizer.get_optimized_prompt(
                llm_model, 
                task_type, 
                file_info,
                metadata=config.get('metadata')
            )
            
            # Store session configuration
            session_config = {
                'session_id': session_id,
                'file_info': file_info,
                'embedding_model': {
                    'key': embedding_model,
                    'config': config.get('embedding', {})
                },
                'llm_model': {
                    'key': llm_model,
                    'config': config.get('llm', {})
                },
                'prompt_template': prompt_template.template,
                'task_type': task_type
            }
            
            await self._save_session_config(session_id, session_config)
            
            # Prepare response
            response = {
                'session_id': session_id,
                'filename': file.filename,
                'file_type': file_info['file_type'],
                'file_path': str(file_path),
                'chunks': file_info.get('chunks', 0),
                'embedding_model': embedding_model,
                'llm_model': llm_model,
                'task_type': task_type,
                'processing_stats': {
                    'embedding_dimensions': embedding_instance.dimensions if hasattr(embedding_instance, 'dimensions') else None,
                    'estimated_cost': self._calculate_estimated_cost(file_info, embedding_model, llm_model)
                }
            }
            
            # Add file-specific information
            if file_info['file_type'] == 'pdf':
                response.update({
                    'pages': file_info.get('pages', 0),
                    'metadata': file_info.get('metadata', {}),
                    'loader_used': file_info.get('loader_used')
                })
            elif file_info['file_type'] == 'csv':
                response.update({
                    'rows': file_info.get('rows', 0),
                    'columns': file_info.get('columns', 0),
                    'column_names': file_info.get('column_names', [])
                })
            
            logger.info(f"Successfully processed upload with models - Embedding: {embedding_model}, LLM: {llm_model}")
            return response
            
        except Exception as e:
            logger.error(f"Upload processing failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Upload processing failed: {str(e)}")
    
    def _validate_model_selection(self, embedding_model: str, llm_model: str) -> None:
        """Validate model selection and compatibility"""
        # Check if models exist
        if not self.embedding_manager.is_model_available(embedding_model):
            raise HTTPException(status_code=400, detail=f"Embedding model '{embedding_model}' not available")
        
        if not self.llm_manager.is_model_available(llm_model):
            raise HTTPException(status_code=400, detail=f"LLM model '{llm_model}' not available")
        
        # Check model compatibility
        compatibility = self._check_model_compatibility(embedding_model, llm_model)
        if not compatibility['compatible']:
            raise HTTPException(
                status_code=400, 
                detail=f"Models not compatible: {compatibility['reason']}"
            )
    
    def _check_model_compatibility(self, embedding_model: str, llm_model: str) -> Dict[str, Any]:
        """Check if selected models are compatible"""
        # Define compatibility matrix
        compatibility_matrix = {
            'openai-ada-002': ['gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-4', 'gpt-4-turbo-preview'],
            'openai-embedding-3-small': ['gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-4', 'gpt-4-turbo-preview'],
            'openai-embedding-3-large': ['gpt-4', 'gpt-4-turbo-preview'],
            'huggingface-sentence-transformers': ['claude-3-haiku-20240307', 'claude-instant-1.2'],
            'cohere-embed-multilingual': ['claude-3-sonnet-20240229', 'claude-3-haiku-20240307']
        }
        
        compatible_llms = compatibility_matrix.get(embedding_model, [])
        
        if llm_model in compatible_llms:
            return {'compatible': True, 'reason': 'Models are compatible'}
        else:
            return {
                'compatible': False, 
                'reason': f"LLM '{llm_model}' not compatible with embedding model '{embedding_model}'"
            }
    
    async def _save_uploaded_file(self, file: UploadFile, session_id: str) -> Path:
        """Save uploaded file to disk"""
        # Create session directory
        session_dir = self.upload_dir / session_id
        session_dir.mkdir(exist_ok=True)
        
        # Save file
        file_path = session_dir / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"File saved: {file_path}")
        return file_path
    
    async def _process_file(self, file_path: Path, file: UploadFile, embedding_model) -> Dict[str, Any]:
        """Process uploaded file and extract information"""
        file_extension = file_path.suffix.lower()
        file_info = {
            'file_path': str(file_path),
            'filename': file.filename,
            'size': file_path.stat().st_size,
            'file_type': 'pdf' if file_extension == '.pdf' else 'csv'
        }
        
        if file_extension == '.pdf':
            file_info.update(await self._process_pdf(file_path, embedding_model))
        elif file_extension == '.csv':
            file_info.update(await self._process_csv(file_path, embedding_model))
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}")
        
        return file_info
    
    async def _process_pdf(self, file_path: Path, embedding_model) -> Dict[str, Any]:
        """Process PDF file"""
        try:
            from PyPDF2 import PdfReader
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            
            # Read PDF
            reader = PdfReader(str(file_path))
            pages = len(reader.pages)
            
            # Extract text
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            
            # Extract metadata
            metadata = {}
            if reader.metadata:
                metadata = {
                    'title': reader.metadata.get('/Title', ''),
                    'author': reader.metadata.get('/Author', ''),
                    'subject': reader.metadata.get('/Subject', ''),
                    'creator': reader.metadata.get('/Creator', '')
                }
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text)
            
            # Create embeddings
            embeddings = await embedding_model.embed_documents(chunks)
            
            return {
                'pages': pages,
                'chunks': len(chunks),
                'metadata': metadata,
                'text_length': len(text),
                'embeddings_created': len(embeddings),
                'loader_used': 'PyPDF2'
            }
            
        except Exception as e:
            logger.error(f"PDF processing failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")
    
    async def _process_csv(self, file_path: Path, embedding_model) -> Dict[str, Any]:
        """Process CSV file"""
        try:
            import pandas as pd
            
            # Read CSV
            df = pd.read_csv(file_path)
            rows, columns = df.shape
            column_names = df.columns.tolist()
            
            # Convert to text for embedding
            text_data = []
            for _, row in df.iterrows():
                row_text = " | ".join([f"{col}: {val}" for col, val in row.items()])
                text_data.append(row_text)
            
            # Create embeddings for sample of data (first 100 rows to avoid excessive processing)
            sample_data = text_data[:min(100, len(text_data))]
            embeddings = await embedding_model.embed_documents(sample_data)
            
            return {
                'rows': rows,
                'columns': columns,
                'column_names': column_names,
                'chunks': len(sample_data),
                'embeddings_created': len(embeddings),
                'sample_rows': min(100, rows)
            }
            
        except Exception as e:
            logger.error(f"CSV processing failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"CSV processing failed: {str(e)}")
    
    def _detect_file_task_type(self, file_info: Dict[str, Any]) -> str:
        """Detect the primary task type based on file characteristics"""
        file_type = file_info.get('file_type')
        
        if file_type == 'csv':
            return 'analysis'  # CSV files are typically for data analysis
        elif file_type == 'pdf':
            # Determine based on file size and content
            pages = file_info.get('pages', 0)
            if pages > 50:
                return 'summary'  # Large documents likely need summarization
            else:
                return 'qa'  # Smaller documents good for Q&A
        
        return 'general'
    
    def _calculate_estimated_cost(self, file_info: Dict[str, Any], embedding_model: str, llm_model: str) -> Dict[str, float]:
        """Calculate estimated processing costs"""
        # Cost estimates per 1K tokens (simplified)
        embedding_costs = {
            'openai-ada-002': 0.0001,
            'openai-embedding-3-small': 0.00002,
            'openai-embedding-3-large': 0.00013,
            'huggingface-sentence-transformers': 0.0,  # Free
            'cohere-embed-multilingual': 0.0001
        }
        
        llm_costs = {
            'gpt-3.5-turbo': 0.002,
            'gpt-3.5-turbo-16k': 0.004,
            'gpt-4': 0.03,
            'gpt-4-turbo-preview': 0.01,
            'claude-3-opus-20240229': 0.015,
            'claude-3-sonnet-20240229': 0.003,
            'claude-3-haiku-20240307': 0.00025
        }
        
        # Estimate tokens
        chunks = file_info.get('chunks', 0)
        estimated_tokens = chunks * 250  # Average tokens per chunk
        
        embedding_cost = (estimated_tokens / 1000) * embedding_costs.get(embedding_model, 0.0001)
        llm_cost_per_query = (estimated_tokens / 1000) * llm_costs.get(llm_model, 0.002)
        
        return {
            'embedding_cost': round(embedding_cost, 4),
            'llm_cost_per_query': round(llm_cost_per_query, 4),
            'estimated_tokens': estimated_tokens
        }
    
    async def _save_session_config(self, session_id: str, config: Dict[str, Any]) -> None:
        """Save session configuration to disk"""
        config_dir = Path("sessions")
        config_dir.mkdir(exist_ok=True)
        
        config_file = config_dir / f"{session_id}.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Session config saved: {config_file}")
    
    async def load_session_config(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load session configuration from disk"""
        config_file = Path("sessions") / f"{session_id}.json"
        
        if not config_file.exists():
            return None
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load session config: {str(e)}")
            return None

# Global instance
upload_handler = EnhancedUploadHandler()
"""
Model-Specific Prompt Templates and Optimization
각 LLM 모델의 특성에 맞춘 프롬프트 최적화
"""

from typing import Dict, Optional
from langchain.prompts import PromptTemplate

class ModelPromptOptimizer:
    """모델별 프롬프트 최적화 클래스"""
    
    def __init__(self):
        self.model_characteristics = {
            # OpenAI Models
            "gpt-3.5-turbo": {
                "strengths": ["빠른 응답", "일반적인 질문", "요약"],
                "context_window": 4096,
                "style": "concise",
                "temperature_range": (0.3, 0.7)
            },
            "gpt-3.5-turbo-16k": {
                "strengths": ["긴 문서 처리", "상세한 분석", "대용량 데이터"],
                "context_window": 16384,
                "style": "detailed",
                "temperature_range": (0.3, 0.7)
            },
            "gpt-4": {
                "strengths": ["복잡한 추론", "창의적 해결", "정확성"],
                "context_window": 8192,
                "style": "analytical",
                "temperature_range": (0.2, 0.6)
            },
            "gpt-4-turbo-preview": {
                "strengths": ["최신 정보", "멀티모달", "긴 컨텍스트"],
                "context_window": 128000,
                "style": "comprehensive",
                "temperature_range": (0.2, 0.6)
            },
            
            # Claude Models
            "claude-3-opus-20240229": {
                "strengths": ["깊은 분석", "창의적 사고", "복잡한 작업"],
                "context_window": 200000,
                "style": "thorough",
                "temperature_range": (0.2, 0.5)
            },
            "claude-3-sonnet-20240229": {
                "strengths": ["균형잡힌 성능", "다양한 작업", "효율성"],
                "context_window": 200000,
                "style": "balanced",
                "temperature_range": (0.3, 0.7)
            },
            "claude-3-haiku-20240307": {
                "strengths": ["빠른 응답", "간단한 작업", "효율성"],
                "context_window": 200000,
                "style": "efficient",
                "temperature_range": (0.3, 0.7)
            },
            "claude-2.1": {
                "strengths": ["긴 문서", "상세한 분석", "정확성"],
                "context_window": 100000,
                "style": "detailed",
                "temperature_range": (0.2, 0.6)
            },
            "claude-instant-1.2": {
                "strengths": ["즉각적인 응답", "간단한 질문", "속도"],
                "context_window": 100000,
                "style": "quick",
                "temperature_range": (0.3, 0.7)
            }
        }
    
    def get_optimized_prompt(self, model_name: str, task_type: str, 
                           file_info: Dict, metadata: Optional[Dict] = None) -> PromptTemplate:
        """모델과 작업 유형에 맞는 최적화된 프롬프트 반환"""
        
        model_info = self.model_characteristics.get(model_name, {})
        style = model_info.get("style", "balanced")
        
        # 기본 프롬프트 구조
        base_prompt = self._get_base_prompt(model_name, file_info, metadata)
        
        # 작업 유형별 최적화
        if task_type == "qa":
            return self._get_qa_prompt(model_name, style, base_prompt)
        elif task_type == "analysis":
            return self._get_analysis_prompt(model_name, style, base_prompt)
        elif task_type == "summary":
            return self._get_summary_prompt(model_name, style, base_prompt)
        else:
            return self._get_general_prompt(model_name, style, base_prompt)
    
    def _get_base_prompt(self, model_name: str, file_info: Dict, metadata: Optional[Dict]) -> str:
        """기본 프롬프트 정보 구성"""
        file_type = file_info.get("file_type", "document")
        filename = file_info.get("filename", "uploaded file")
        
        # 모델별 특성 반영
        if "claude" in model_name.lower():
            # Claude는 더 직접적이고 명확한 지시를 선호
            intro = f"You are analyzing the {file_type} '{filename}'."
            instruction_style = "Please follow these guidelines precisely:"
        else:
            # GPT는 더 맥락적인 설명을 선호
            intro = f"You are an AI assistant analyzing the {file_type} '{filename}'."
            instruction_style = "Guidelines for your response:"
        
        # 메타데이터 정보
        metadata_info = ""
        if metadata:
            if file_type == "pdf" and metadata.get("pages"):
                metadata_info += f"\n- Document pages: {metadata.get('pages', 'N/A')}"
            elif file_type == "csv" and metadata.get("rows"):
                metadata_info += f"\n- Data rows: {metadata.get('rows', 'N/A')}"
                metadata_info += f"\n- Data columns: {metadata.get('columns', 'N/A')}"
        
        return f"{intro}{metadata_info}\n\n{instruction_style}"
    
    def _get_qa_prompt(self, model_name: str, style: str, base_prompt: str) -> PromptTemplate:
        """Q&A 스타일 프롬프트"""
        
        # 스타일별 응답 지침
        style_guidelines = {
            "concise": """
- Provide direct, concise answers (1-2 sentences for simple questions)
- Focus on the most relevant information
- Use bullet points for multiple items""",
            
            "detailed": """
- Provide comprehensive answers with context
- Include relevant details and examples from the document
- Structure long answers with clear sections""",
            
            "analytical": """
- Analyze the question deeply before answering
- Consider multiple perspectives if applicable
- Provide evidence-based reasoning""",
            
            "comprehensive": """
- Address all aspects of the question thoroughly
- Include background context when helpful
- Synthesize information from multiple parts of the document""",
            
            "thorough": """
- Examine the question from multiple angles
- Provide exhaustive coverage of the topic
- Include nuanced understanding and edge cases""",
            
            "balanced": """
- Provide complete yet accessible answers
- Balance detail with clarity
- Focus on practical understanding""",
            
            "efficient": """
- Get straight to the point
- Prioritize the most important information
- Keep answers brief but complete""",
            
            "quick": """
- Provide immediate, actionable answers
- Focus on the essential information only
- Use simple, clear language"""
        }
        
        guidelines = style_guidelines.get(style, style_guidelines["balanced"])
        
        # Claude 특화 최적화
        if "claude" in model_name.lower():
            template = f"""{base_prompt}

{guidelines}

Important: Base all answers strictly on the provided document content. If the information is not in the document, clearly state this.

Context from document:
{{context}}

Previous conversation:
{{chat_history}}

Question: {{question}}

Answer:"""
        # GPT 특화 최적화
        else:
            template = f"""{base_prompt}

{guidelines}

## Response Format:
- For factual questions: Start with the direct answer
- For analysis questions: Begin with a brief overview
- For comparison questions: Use structured format

Remember to cite specific parts of the document when relevant.

Context from document:
{{context}}

Chat History:
{{chat_history}}

Human Question: {{question}}

AI Response:"""
        
        return PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template=template
        )
    
    def _get_analysis_prompt(self, model_name: str, style: str, base_prompt: str) -> PromptTemplate:
        """분석 작업용 프롬프트"""
        
        # Claude는 구조화된 분석을 잘 수행
        if "claude" in model_name.lower():
            if "opus" in model_name:
                # Opus는 깊은 분석에 최적화
                analysis_structure = """
Perform a comprehensive analysis following this structure:
1. **Overview**: High-level summary of findings
2. **Detailed Analysis**: 
   - Key patterns and trends
   - Significant data points
   - Relationships and correlations
3. **Insights**: Deep insights and implications
4. **Recommendations**: Actionable suggestions based on analysis
5. **Limitations**: Any constraints or caveats"""
            else:
                # 다른 Claude 모델은 더 간결한 구조
                analysis_structure = """
Structure your analysis as follows:
1. **Key Findings**: Main discoveries from the document
2. **Supporting Evidence**: Specific examples and data
3. **Implications**: What this means practically
4. **Next Steps**: Suggested actions or considerations"""
        
        # GPT는 더 유연한 구조 선호
        else:
            if "gpt-4" in model_name:
                # GPT-4는 복잡한 추론 가능
                analysis_structure = """
Provide an analytical response that includes:
- Executive Summary
- Detailed findings with evidence
- Critical evaluation of the information
- Practical applications and recommendations
- Potential risks or considerations"""
            else:
                # GPT-3.5는 더 직접적인 구조
                analysis_structure = """
Analyze the document focusing on:
- Main themes and patterns
- Important details and data points
- Practical implications
- Clear recommendations"""
        
        template = f"""{base_prompt}

{analysis_structure}

Context from document:
{{context}}

Analysis request: {{question}}

Comprehensive Analysis:"""
        
        return PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )
    
    def _get_summary_prompt(self, model_name: str, style: str, base_prompt: str) -> PromptTemplate:
        """요약 작업용 프롬프트"""
        
        # 모델별 요약 스타일
        if "claude" in model_name.lower():
            if "haiku" in model_name or "instant" in model_name:
                # 빠른 모델은 핵심만
                summary_style = """
Create a concise summary focusing on:
- Core message (2-3 sentences)
- Key points (3-5 bullet points)
- Essential takeaways"""
            else:
                # 더 강력한 모델은 구조화된 요약
                summary_style = """
Provide a structured summary:
1. **Executive Summary** (1 paragraph)
2. **Key Points** (organized by theme)
3. **Important Details** (supporting information)
4. **Conclusions** (main takeaways)"""
        
        else:  # GPT models
            if "16k" in model_name or "turbo" in model_name:
                # 긴 컨텍스트 모델은 상세 요약
                summary_style = """
Create a comprehensive summary including:
- Overview (2-3 paragraphs)
- Detailed key points with context
- Supporting details and examples
- Conclusions and implications"""
            else:
                # 일반 모델은 표준 요약
                summary_style = """
Summarize the document with:
- Brief overview (1-2 paragraphs)
- Main points (bullet list)
- Key takeaways"""
        
        template = f"""{base_prompt}

{summary_style}

Document content:
{{context}}

Summary request: {{question}}

Summary:"""
        
        return PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )
    
    def _get_general_prompt(self, model_name: str, style: str, base_prompt: str) -> PromptTemplate:
        """일반 작업용 프롬프트"""
        
        # 모델 특성에 따른 일반 지침
        model_instructions = {
            "claude": """
You are a helpful AI assistant. Provide clear, accurate, and useful responses based on the document content.
Focus on being helpful while maintaining precision and clarity.""",
            
            "gpt": """
You are a knowledgeable AI assistant analyzing documents. Provide thoughtful, well-structured responses that directly address the user's needs.
Maintain a professional yet approachable tone."""
        }
        
        instruction = model_instructions.get(
            "claude" if "claude" in model_name.lower() else "gpt"
        )
        
        template = f"""{base_prompt}

{instruction}

Context from document:
{{context}}

Chat History:
{{chat_history}}

Question: {{question}}

Response:"""
        
        return PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template=template
        )
    
    def get_optimal_temperature(self, model_name: str, task_type: str) -> float:
        """작업 유형에 따른 최적 temperature 반환"""
        
        model_info = self.model_characteristics.get(model_name, {})
        temp_range = model_info.get("temperature_range", (0.3, 0.7))
        
        # 작업별 temperature 조정
        task_temps = {
            "qa": temp_range[0],  # 낮은 temperature로 정확성 향상
            "analysis": (temp_range[0] + temp_range[1]) / 2,  # 중간값
            "summary": temp_range[0] + 0.1,  # 약간 높여서 자연스러움
            "creative": temp_range[1],  # 높은 temperature로 창의성
            "general": (temp_range[0] + temp_range[1]) / 2  # 중간값
        }
        
        return task_temps.get(task_type, 0.5)
    
    def get_model_specific_params(self, model_name: str, task_type: str) -> Dict:
        """모델별 최적 파라미터 반환"""
        
        params = {
            "temperature": self.get_optimal_temperature(model_name, task_type),
            "max_tokens": 2000,  # 기본값
        }
        
        # 모델별 특수 설정
        if "gpt-4" in model_name:
            params["max_tokens"] = 3000  # GPT-4는 더 긴 응답 가능
        elif "claude-3-opus" in model_name:
            params["max_tokens"] = 4000  # Opus는 매우 상세한 응답 가능
        elif "haiku" in model_name or "instant" in model_name:
            params["max_tokens"] = 1500  # 빠른 모델은 짧게
        
        # 작업별 조정
        if task_type == "summary":
            params["max_tokens"] = min(params["max_tokens"], 1000)
        elif task_type == "analysis":
            params["max_tokens"] = max(params["max_tokens"], 2500)
        
        return params

    def detect_task_type(self, question: str) -> str:
        """질문에서 작업 유형 감지"""
        question_lower = question.lower()
        
        # 요약 관련 키워드
        summary_keywords = ["요약", "정리", "핵심", "summary", "summarize", "주요 내용"]
        if any(keyword in question_lower for keyword in summary_keywords):
            return "summary"
        
        # 분석 관련 키워드
        analysis_keywords = ["분석", "비교", "추세", "패턴", "analyze", "compare", "trend", "통계"]
        if any(keyword in question_lower for keyword in analysis_keywords):
            return "analysis"
        
        # 질문 형태 패턴
        qa_patterns = ["무엇", "언제", "어디", "누가", "왜", "어떻게", "what", "when", "where", "who", "why", "how"]
        if any(pattern in question_lower for pattern in qa_patterns):
            return "qa"
        
        # 기본값은 일반 질의응답
        return "general"

# 전역 인스턴스
prompt_optimizer = ModelPromptOptimizer()
"""
Q&A 스타일 응답 포맷터
간단명료한 답변과 부연설명을 분리
"""

import re
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)

class QAResponseFormatter:
    """Q&A 봇 스타일의 간단명료한 응답 포맷터"""
    
    def __init__(self):
        self.answer_patterns = {
            "definition": r"(.*?는\s+.*?(?:입니다|이다|임))",
            "yes_no": r"^(네|아니오|예|아니요|맞습니다|틀립니다|그렇습니다|그렇지 않습니다)",
            "number": r"^(\d+(?:\.\d+)?(?:\s*[가-힣]+)?)",
            "list": r"^(\d+\..*?)(?:\n|$)",
            "date": r"(\d{4}년\s*\d{1,2}월\s*\d{1,2}일|\d{4}-\d{2}-\d{2})",
        }
        
        self.question_types = {
            "what": ["무엇", "뭐", "what", "어떤"],
            "why": ["왜", "이유", "why", "어째서"],
            "how": ["어떻게", "방법", "how", "절차"],
            "when": ["언제", "시기", "when", "날짜"],
            "where": ["어디", "장소", "where", "위치"],
            "who": ["누구", "누가", "who", "사람"],
            "yes_no": ["입니까", "인가요", "맞나요", "맞습니까", "있나요", "있습니까"],
            "how_many": ["얼마나", "몇", "개수", "수량"],
        }
    
    def identify_question_type(self, question: str) -> str:
        """질문 유형 식별"""
        question_lower = question.lower()
        
        for q_type, keywords in self.question_types.items():
            for keyword in keywords:
                if keyword in question_lower:
                    return q_type
        
        return "general"
    
    def extract_core_answer(self, full_response: str, question_type: str) -> str:
        """전체 응답에서 핵심 답변만 추출"""
        try:
            # 줄바꿈으로 분리
            sentences = full_response.split('\n')
            first_paragraph = sentences[0] if sentences else full_response
            
            # 질문 유형별 처리
            if question_type == "yes_no":
                # Yes/No 질문의 경우 첫 문장에서 답 추출
                for pattern in ["네", "아니오", "예", "아니요", "맞습니다", "틀립니다"]:
                    if pattern in first_paragraph:
                        return self._extract_first_complete_sentence(first_paragraph)
            
            elif question_type == "what":
                # 정의형 질문의 경우 첫 번째 완전한 문장
                return self._extract_first_complete_sentence(first_paragraph)
            
            elif question_type in ["how_many", "when"]:
                # 숫자나 날짜가 포함된 첫 문장
                for sentence in sentences[:3]:
                    if re.search(r'\d+', sentence):
                        return self._extract_first_complete_sentence(sentence)
            
            elif question_type == "why":
                # 이유를 설명하는 첫 문장 (때문, 이유는 등)
                for sentence in sentences[:3]:
                    if any(word in sentence for word in ["때문", "이유", "원인", "왜냐하면"]):
                        return self._extract_first_complete_sentence(sentence)
            
            # 기본값: 첫 번째 완전한 문장
            return self._extract_first_complete_sentence(first_paragraph)
            
        except Exception as e:
            logger.error(f"핵심 답변 추출 실패: {e}")
            return full_response.split('.')[0] + '.' if '.' in full_response else full_response
    
    def _extract_first_complete_sentence(self, text: str) -> str:
        """첫 번째 완전한 문장 추출"""
        # 문장 종결 패턴
        sentence_endings = ['. ', '.\n', '다.', '요.', '니다.', '까.', '죠.']
        
        min_pos = len(text)
        for ending in sentence_endings:
            pos = text.find(ending)
            if pos != -1 and pos < min_pos:
                min_pos = pos + len(ending) - 1
        
        if min_pos < len(text):
            return text[:min_pos + 1].strip()
        
        return text.strip()
    
    def extract_supplementary_info(self, full_response: str, core_answer: str) -> Optional[str]:
        """전체 응답에서 부연설명 추출"""
        try:
            # 핵심 답변 이후의 내용 추출
            if core_answer in full_response:
                remaining = full_response[full_response.index(core_answer) + len(core_answer):].strip()
                
                if remaining and len(remaining) > 20:
                    # 부연설명 정리
                    cleaned = self._clean_supplementary(remaining)
                    if cleaned:
                        return cleaned
            
            # 핵심 답변과 다른 내용이 있는 경우
            if len(full_response) > len(core_answer) * 2:
                # 전체 응답에서 핵심 답변을 제외한 부분
                remaining = full_response.replace(core_answer, '', 1).strip()
                if remaining:
                    return self._clean_supplementary(remaining)
            
            return None
            
        except Exception as e:
            logger.error(f"부연설명 추출 실패: {e}")
            return None
    
    def _clean_supplementary(self, text: str) -> str:
        """부연설명 정리"""
        # 불필요한 접속사 제거
        text = re.sub(r'^(그리고|또한|한편|따라서|그러나|하지만)\s*', '', text)
        
        # 중복된 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        # 빈 줄 제거
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        return '\n'.join(lines).strip()
    
    def format_qa_response(self, question: str, full_response: str) -> Dict[str, str]:
        """Q&A 스타일로 응답 포맷팅"""
        try:
            # 질문 유형 식별
            question_type = self.identify_question_type(question)
            
            # 핵심 답변 추출
            core_answer = self.extract_core_answer(full_response, question_type)
            
            # 부연설명 추출
            supplementary = self.extract_supplementary_info(full_response, core_answer)
            
            # 포맷팅된 응답 생성
            formatted_response = {
                "core_answer": core_answer,
                "supplementary": supplementary,
                "question_type": question_type,
                "full_formatted": self._create_formatted_response(core_answer, supplementary)
            }
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Q&A 응답 포맷팅 실패: {e}")
            return {
                "core_answer": full_response,
                "supplementary": None,
                "question_type": "general",
                "full_formatted": full_response
            }
    
    def _create_formatted_response(self, core_answer: str, supplementary: Optional[str]) -> str:
        """최종 포맷팅된 응답 생성"""
        if supplementary:
            return f"{core_answer}\n\n**부연설명:**\n{supplementary}"
        else:
            return core_answer
    
    def enhance_answer_clarity(self, answer: str, question_type: str) -> str:
        """답변의 명확성 향상"""
        # 숫자 답변 강조
        if question_type == "how_many":
            answer = re.sub(r'(\d+(?:\.\d+)?)', r'**\1**', answer)
        
        # 날짜 답변 강조
        elif question_type == "when":
            answer = re.sub(r'(\d{4}년\s*\d{1,2}월\s*\d{1,2}일)', r'**\1**', answer)
        
        # Yes/No 답변 강조
        elif question_type == "yes_no":
            for word in ["네", "아니오", "예", "아니요", "맞습니다", "틀립니다"]:
                if word in answer:
                    answer = answer.replace(word, f"**{word}**", 1)
                    break
        
        return answer
    
    def create_bullet_points(self, items: List[str]) -> str:
        """리스트 항목을 불릿 포인트로 변환"""
        if not items:
            return ""
        
        formatted_items = []
        for i, item in enumerate(items, 1):
            formatted_items.append(f"{i}. {item.strip()}")
        
        return "\n".join(formatted_items)
    
    def is_answer_complete(self, answer: str) -> bool:
        """답변이 완전한지 확인"""
        # 최소 길이 체크
        if len(answer.strip()) < 10:
            return False
        
        # 문장 종결 체크
        sentence_endings = ['다', '요', '니다', '까', '죠', '.', '!', '?']
        return any(answer.strip().endswith(ending) for ending in sentence_endings)
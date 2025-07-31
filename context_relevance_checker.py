"""
컨텍스트 관련성 검사 시스템
질문이 업로드된 문서와 관련이 있는지 검증
"""

import re
from typing import List, Dict, Tuple, Optional
import logging

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    
try:
    from konlpy.tag import Okt
    KONLPY_AVAILABLE = True
except ImportError:
    KONLPY_AVAILABLE = False

logger = logging.getLogger(__name__)

class ContextRelevanceChecker:
    """질문과 문서 컨텍스트의 관련성을 검사하는 클래스"""
    
    def __init__(self):
        self.okt = Okt() if KONLPY_AVAILABLE else None
        self.general_questions = [
            # 일반적인 질문 패턴
            r"안녕|반가|하이|헬로|hello|hi",
            r"날씨|기온|비|눈|맑|흐림",
            r"오늘|내일|어제|요일|날짜|시간",
            r"누구|당신|너|AI|인공지능|챗봇",
            r"뭐해|무엇을|뭘|어떻게 지내",
            r"고마워|감사|땡큐|thanks",
            r"미안|죄송|sorry",
            r"농담|재미있|웃긴",
            r"사랑|좋아|싫어|감정",
            r"게임|영화|드라마|음악|연예인",
            r"요리|음식|레시피|맛집",
            r"운동|스포츠|축구|야구|농구",
            r"주식|코인|비트코인|투자",
            r"쇼핑|구매|판매|가격",
        ]
        
        self.document_related_keywords = [
            # 문서 관련 키워드
            "문서", "파일", "내용", "자료", "데이터",
            "페이지", "단락", "섹션", "챕터", "부분",
            "설명", "정의", "의미", "뜻", "해석",
            "분석", "요약", "정리", "핵심", "중요",
            "찾아", "검색", "확인", "알려", "보여",
            "어디", "언급", "나와", "있는", "관련"
        ]
        
        self.off_topic_responses = [
            "죄송합니다. 업로드하신 문서의 내용과 관련된 질문을 해주세요. 🔍",
            "이 질문은 현재 업로드된 문서와 관련이 없는 것 같습니다. 문서 내용에 대해 궁금하신 점을 물어봐 주세요. 📄",
            "제가 답변드릴 수 있는 범위는 업로드하신 문서의 내용으로 제한됩니다. 문서와 관련된 질문을 해주시면 도움을 드리겠습니다. 📚",
            "현재 제공된 문서 외의 일반적인 질문에는 답변드리기 어렵습니다. 문서 내용에 대해 질문해 주세요. 💡",
            "업로드된 문서를 기반으로 답변을 제공하고 있습니다. 문서와 관련된 구체적인 질문을 해주시면 감사하겠습니다. 🎯"
        ]
    
    def extract_keywords(self, text: str) -> List[str]:
        """텍스트에서 핵심 키워드 추출"""
        try:
            keywords = []
            
            if self.okt and KONLPY_AVAILABLE:
                # 형태소 분석
                tokens = self.okt.nouns(text)
                # 2글자 이상인 명사만 추출
                keywords = [token for token in tokens if len(token) >= 2]
            else:
                # 간단한 단어 추출 (konlpy 없을 때)
                # 한글 단어 추출 (2글자 이상)
                korean_words = re.findall(r'[가-힣]{2,}', text)
                keywords.extend(korean_words)
            
            # 영어 단어 추출
            english_words = re.findall(r'\b[A-Za-z]{3,}\b', text)
            keywords.extend([word.lower() for word in english_words])
            
            return list(set(keywords))
        except Exception as e:
            logger.error(f"키워드 추출 실패: {e}")
            return []
    
    def is_general_question(self, question: str) -> bool:
        """일반적인 대화성 질문인지 확인"""
        question_lower = question.lower()
        
        for pattern in self.general_questions:
            if re.search(pattern, question_lower):
                return True
        
        # 질문이 너무 짧으면 일반 질문으로 간주
        if len(question.strip()) < 10:
            return True
        
        # 문서 관련 키워드가 없으면 일반 질문으로 간주
        has_doc_keyword = any(keyword in question for keyword in self.document_related_keywords)
        if not has_doc_keyword and len(question.split()) < 5:
            return True
        
        return False
    
    def calculate_relevance_score(self, question: str, document_chunks: List[str]) -> float:
        """질문과 문서 청크들 간의 관련성 점수 계산"""
        try:
            if not document_chunks:
                return 0.0
            
            # 질문 키워드 추출
            question_keywords = self.extract_keywords(question)
            if not question_keywords:
                return 0.0
            
            # 문서 전체에서 키워드 추출 (상위 100개 청크만 사용)
            sample_chunks = document_chunks[:100]
            doc_text = " ".join(sample_chunks)
            doc_keywords = self.extract_keywords(doc_text)
            
            if not doc_keywords:
                return 0.0
            
            # 키워드 교집합 비율 계산
            common_keywords = set(question_keywords) & set(doc_keywords)
            relevance_score = len(common_keywords) / len(question_keywords) if question_keywords else 0
            
            # TF-IDF 기반 유사도 계산
            if SKLEARN_AVAILABLE:
                try:
                    all_texts = [question] + sample_chunks[:20]  # 질문 + 상위 20개 청크
                    vectorizer = TfidfVectorizer(max_features=100)
                    tfidf_matrix = vectorizer.fit_transform(all_texts)
                    
                    # 질문과 각 청크 간의 코사인 유사도
                    question_vector = tfidf_matrix[0:1]
                    chunk_vectors = tfidf_matrix[1:]
                    similarities = cosine_similarity(question_vector, chunk_vectors).flatten()
                    
                    # 최대 유사도
                    max_similarity = np.max(similarities) if len(similarities) > 0 else 0
                    
                    # 종합 점수 (키워드 매칭 40% + 유사도 60%)
                    final_score = (relevance_score * 0.4) + (max_similarity * 0.6)
                    
                    return float(final_score)
                    
                except Exception as e:
                    logger.error(f"TF-IDF 계산 실패: {e}")
                    return relevance_score
            else:
                # sklearn 없을 때는 키워드 매칭만 사용
                return relevance_score
                
        except Exception as e:
            logger.error(f"관련성 점수 계산 실패: {e}")
            return 0.0
    
    def check_relevance(self, question: str, document_chunks: List[str], 
                       threshold: float = 0.15) -> Tuple[bool, float, Optional[str]]:
        """
        질문의 관련성 검사
        
        Returns:
            - is_relevant: 관련성 여부
            - score: 관련성 점수
            - rejection_message: 거부 메시지 (관련 없을 경우)
        """
        # 빈 질문 처리
        if not question or not question.strip():
            return False, 0.0, "질문을 입력해주세요."
        
        # 일반적인 대화성 질문 필터링
        if self.is_general_question(question):
            import random
            return False, 0.0, random.choice(self.off_topic_responses)
        
        # 문서가 없는 경우
        if not document_chunks:
            return False, 0.0, "먼저 문서를 업로드해주세요. 📎"
        
        # 관련성 점수 계산
        relevance_score = self.calculate_relevance_score(question, document_chunks)
        
        # 임계값 기반 판단
        is_relevant = relevance_score >= threshold
        
        if not is_relevant:
            # 점수에 따른 차별화된 응답
            import random
            if relevance_score < 0.05:
                # 매우 낮은 관련성
                message = random.choice(self.off_topic_responses)
            elif relevance_score < 0.1:
                # 낮은 관련성
                message = "이 질문은 업로드된 문서의 주제와 다소 거리가 있어 보입니다. 문서 내용과 더 직접적으로 관련된 질문을 해주세요. 🤔"
            else:
                # 경계선상의 관련성
                message = "질문이 문서 내용과 약간의 연관성은 있지만, 더 구체적으로 문서 내용에 대해 질문해주시면 정확한 답변을 드릴 수 있습니다. 📝"
            
            return False, relevance_score, message
        
        return True, relevance_score, None
    
    def enhance_question_with_context(self, question: str, is_relevant: bool, score: float) -> str:
        """질문에 컨텍스트 정보 추가"""
        if is_relevant:
            if score > 0.5:
                prefix = "[문서 관련 질문] "
            elif score > 0.3:
                prefix = "[부분적 관련 질문] "
            else:
                prefix = "[확인 필요 질문] "
            
            return f"{prefix}{question}"
        
        return question
    
    def get_contextual_hints(self, document_chunks: List[str], max_hints: int = 5) -> List[str]:
        """문서 기반 질문 힌트 생성"""
        if not document_chunks:
            return []
        
        # 문서에서 주요 키워드 추출
        sample_text = " ".join(document_chunks[:30])
        keywords = self.extract_keywords(sample_text)
        
        # 상위 키워드 기반 힌트 생성
        hints = []
        top_keywords = keywords[:10]
        
        if top_keywords:
            hints.extend([
                f"{keyword}에 대해 설명해주세요" for keyword in top_keywords[:2]
            ])
            hints.extend([
                f"{keyword}의 특징은 무엇인가요?" for keyword in top_keywords[2:4]
            ])
            hints.append(f"문서에서 {top_keywords[0]}와 {top_keywords[1]}의 관계를 설명해주세요")
        
        # 일반적인 문서 관련 질문 추가
        general_hints = [
            "이 문서의 핵심 내용을 요약해주세요",
            "문서에서 가장 중요한 포인트는 무엇인가요?",
            "이 자료의 주요 결론을 알려주세요"
        ]
        
        hints.extend(general_hints)
        
        return hints[:max_hints]
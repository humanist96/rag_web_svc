"""
세션 히스토리 저장 및 로드 시스템
JSON 파일 기반 영구 저장소
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class SessionStorage:
    """세션 데이터를 JSON 파일로 저장/로드하는 클래스"""
    
    def __init__(self, storage_path: str = "session_data"):
        self.storage_path = storage_path
        self.history_file = os.path.join(storage_path, "session_history.json")
        self.memory_file = os.path.join(storage_path, "memory_usage.json")
        
        # 저장 디렉토리 생성
        os.makedirs(storage_path, exist_ok=True)
        
        # 기존 데이터 로드
        self.load_data()
    
    def load_data(self) -> Dict:
        """저장된 데이터 로드"""
        session_history = {}
        memory_usage = {
            "total_size_bytes": 0,
            "session_sizes": {},
            "last_cleanup": None
        }
        
        # 세션 히스토리 로드
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # datetime 문자열을 datetime 객체로 변환
                    for session_id, session_data in data.items():
                        session_data["created_at"] = datetime.fromisoformat(session_data["created_at"])
                        session_data["last_accessed"] = datetime.fromisoformat(session_data["last_accessed"])
                        
                        for file in session_data.get("files", []):
                            file["upload_time"] = datetime.fromisoformat(file["upload_time"])
                        
                        session_history[session_id] = session_data
                
                logger.info(f"세션 히스토리 로드 완료: {len(session_history)}개 세션")
            except Exception as e:
                logger.error(f"세션 히스토리 로드 실패: {e}")
        
        # 메모리 사용량 로드
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    memory_usage = data
                    if memory_usage.get("last_cleanup"):
                        memory_usage["last_cleanup"] = datetime.fromisoformat(memory_usage["last_cleanup"])
                
                logger.info(f"메모리 사용량 로드 완료: {memory_usage['total_size_bytes']} bytes")
            except Exception as e:
                logger.error(f"메모리 사용량 로드 실패: {e}")
        
        return {
            "session_history": session_history,
            "memory_usage": memory_usage
        }
    
    def save_session_history(self, session_history: Dict) -> bool:
        """세션 히스토리 저장"""
        try:
            # datetime 객체를 문자열로 변환
            serializable_data = {}
            
            for session_id, session_data in session_history.items():
                serializable_session = session_data.copy()
                serializable_session["created_at"] = session_data["created_at"].isoformat()
                serializable_session["last_accessed"] = session_data["last_accessed"].isoformat()
                
                serializable_files = []
                for file in session_data.get("files", []):
                    serializable_file = file.copy()
                    serializable_file["upload_time"] = file["upload_time"].isoformat()
                    serializable_files.append(serializable_file)
                
                serializable_session["files"] = serializable_files
                serializable_data[session_id] = serializable_session
            
            # JSON 파일로 저장
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"세션 히스토리 저장 완료: {len(session_history)}개 세션")
            return True
            
        except Exception as e:
            logger.error(f"세션 히스토리 저장 실패: {e}")
            return False
    
    def save_memory_usage(self, memory_usage: Dict) -> bool:
        """메모리 사용량 저장"""
        try:
            serializable_data = memory_usage.copy()
            if memory_usage.get("last_cleanup"):
                serializable_data["last_cleanup"] = memory_usage["last_cleanup"].isoformat()
            
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"메모리 사용량 저장 완료: {memory_usage['total_size_bytes']} bytes")
            return True
            
        except Exception as e:
            logger.error(f"메모리 사용량 저장 실패: {e}")
            return False
    
    def add_session_log(self, session_id: str, action: str, details: Dict) -> bool:
        """세션 활동 로그 추가"""
        try:
            log_file = os.path.join(self.storage_path, f"session_{session_id}_log.json")
            
            # 기존 로그 로드
            logs = []
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
            
            # 새 로그 추가
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "action": action,
                "details": details
            }
            logs.append(log_entry)
            
            # 로그 저장
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(logs, f, ensure_ascii=False, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"세션 로그 저장 실패: {e}")
            return False
    
    def get_session_logs(self, session_id: str) -> List[Dict]:
        """세션 활동 로그 조회"""
        try:
            log_file = os.path.join(self.storage_path, f"session_{session_id}_log.json")
            
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
                    # timestamp를 datetime 객체로 변환
                    for log in logs:
                        log["timestamp"] = datetime.fromisoformat(log["timestamp"])
                    return logs
            
            return []
            
        except Exception as e:
            logger.error(f"세션 로그 조회 실패: {e}")
            return []
    
    def cleanup_old_logs(self, days: int = 7) -> int:
        """오래된 로그 파일 정리"""
        try:
            cleaned_count = 0
            cutoff_date = datetime.now().timestamp() - (days * 24 * 60 * 60)
            
            for filename in os.listdir(self.storage_path):
                if filename.startswith("session_") and filename.endswith("_log.json"):
                    file_path = os.path.join(self.storage_path, filename)
                    
                    # 파일 수정 시간 확인
                    if os.path.getmtime(file_path) < cutoff_date:
                        os.remove(file_path)
                        cleaned_count += 1
                        logger.info(f"오래된 로그 삭제: {filename}")
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"로그 정리 실패: {e}")
            return 0
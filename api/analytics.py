"""
Analytics Endpoint for Vercel
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
from typing import Dict, List
from pydantic import BaseModel

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 간단한 메모리 기반 분석 데이터 (실제로는 데이터베이스 사용 권장)
analytics_data = {
    "uploads": 0,
    "queries": 0,
    "query_types": {},
    "response_times": [],
    "keywords": {}
}

class AnalyticsResponse(BaseModel):
    total_uploads: int
    total_queries: int
    query_types: Dict[str, int]
    response_times: List[float]
    keywords: Dict[str, int]
    daily_usage: Dict[str, int]
    average_response_time: float

@app.get("/api/analytics", response_model=AnalyticsResponse)
async def get_analytics():
    """분석 데이터 조회"""
    # 최근 7일 사용량
    daily_usage = {}
    today = datetime.now().date()
    for i in range(7):
        date = (today - timedelta(days=i)).isoformat()
        daily_usage[date] = 0  # 실제로는 날짜별 데이터를 저장해야 함
    
    avg_response_time = (
        sum(analytics_data["response_times"]) / len(analytics_data["response_times"])
        if analytics_data["response_times"] else 0.0
    )
    
    return AnalyticsResponse(
        total_uploads=analytics_data["uploads"],
        total_queries=analytics_data["queries"],
        query_types=analytics_data["query_types"],
        response_times=analytics_data["response_times"][-10:],  # 최근 10개
        keywords=dict(sorted(
            analytics_data["keywords"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]),  # 상위 10개
        daily_usage=daily_usage,
        average_response_time=avg_response_time
    )

handler = app
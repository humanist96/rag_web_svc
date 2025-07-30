"""
Health Check Endpoint for Vercel
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
async def health_check():
    """헬스 체크"""
    return {
        "message": "AI Nexus API is running on Vercel",
        "status": "healthy",
        "environment": "vercel",
        "version": "2.0.0"
    }

handler = app
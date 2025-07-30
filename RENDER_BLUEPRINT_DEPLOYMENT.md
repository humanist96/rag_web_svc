# Render Blueprint 배포 가이드

## 🚀 Blueprint를 통한 전체 서비스 배포

### 1. Render Dashboard에서 Blueprint 배포

1. **Render Dashboard 로그인**
   - https://dashboard.render.com
   - humanist96@gmail.com 계정 사용

2. **Blueprint 생성**
   - Dashboard에서 **"New +"** 클릭
   - **"Blueprint"** 선택

3. **GitHub 리포지토리 연결**
   - `humanist96/rag_web_svc` 리포지토리 선택
   - Branch: `master` 선택

4. **자동 감지 및 배포**
   - Render가 `render.yaml` 파일을 자동으로 감지
   - 다음 서비스들이 자동으로 생성됨:
     - `rag-web-svc-backend` (Python Web Service)
     - `rag-web-svc-frontend` (Static Site)

5. **환경 변수 설정**
   - Backend 서비스 클릭
   - Environment 탭에서 **OPENAI_API_KEY** 추가
   - 값: 실제 OpenAI API 키 입력

### 2. render.yaml 구조 (업데이트됨)

**참고**: Render가 `type: static`을 지원하지 않아 `type: web`으로 변경했습니다.

```yaml
services:
  # Backend API Service
  - type: web
    name: rag-web-svc-backend
    env: python
    plan: free
    buildCommand: "python -m pip install -r requirements_minimal.txt"
    startCommand: "python -m uvicorn enhanced_rag_chatbot:app --host 0.0.0.0 --port $PORT"
    envVars:
      - key: OPENAI_API_KEY
        sync: false  # 수동으로 설정 필요
      - key: ALLOWED_ORIGINS
        value: "https://humanist96.github.io,https://rag-web-svc-frontend.onrender.com"
  
  # Frontend Web Service
  - type: web
    name: rag-web-svc-frontend
    env: python
    plan: free
    startCommand: "python frontend_server.py"
```

### 3. 배포 상태 확인

배포 진행 상황:
1. Services 탭에서 각 서비스 상태 확인
2. 빌드 로그 실시간 모니터링
3. 배포 완료 후 URL 확인

### 4. 예상 URL

- **Backend API**: https://rag-web-svc-backend.onrender.com
- **Frontend**: https://rag-web-svc-frontend.onrender.com

### 5. API를 통한 배포 (대안)

Render API를 사용한 자동화:

```bash
# Render API Key 필요 (Dashboard → Account Settings → API Keys)
export RENDER_API_KEY="your-api-key"

# Blueprint 배포
curl -X POST https://api.render.com/v1/blueprints \
  -H "Authorization: Bearer $RENDER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "rag-web-svc",
    "repo": {
      "provider": "github",
      "owner": "humanist96",
      "name": "rag_web_svc",
      "branch": "master"
    }
  }'
```

### 6. 배포 후 테스트

1. **Backend 헬스체크**
   ```bash
   curl https://rag-web-svc-backend.onrender.com/
   ```

2. **Frontend 접속**
   - 브라우저에서 https://rag-web-svc-frontend.onrender.com 열기

3. **기능 테스트**
   - PDF 업로드
   - AI 채팅
   - 분석 대시보드

### 7. 트러블슈팅

#### Backend 빌드 실패
- requirements_minimal.txt 확인
- Python 버전 호환성 확인
- 환경 변수 설정 확인

#### Frontend 404 오류
- _redirects 파일 확인
- premium_index.html 경로 확인
- staticPublishPath 설정 확인

#### CORS 오류
- Backend ALLOWED_ORIGINS 환경 변수 확인
- Frontend URL이 정확한지 확인

### 8. 모니터링

- **Logs**: 각 서비스의 Logs 탭에서 실시간 로그 확인
- **Metrics**: CPU, Memory 사용량 모니터링
- **Events**: 배포 이벤트 및 상태 변경 추적

## 📌 중요 사항

- Blueprint 배포는 render.yaml의 모든 서비스를 한번에 생성
- 환경 변수는 수동으로 설정 필요 (특히 OPENAI_API_KEY)
- 첫 배포는 5-10분 정도 소요될 수 있음
- Free tier 제한: 15분 비활성 시 스핀다운
# Render.com 배포 가이드

## 백엔드 배포 단계별 가이드

### 1. Render.com 설정
1. https://render.com 에서 humanist96@gmail.com으로 로그인
2. Dashboard에서 "New +" → "Web Service" 클릭

### 2. GitHub 연결
1. "Connect a repository" 선택
2. GitHub 계정 연결 (아직 안 했다면)
3. `humanist96/rag_web_svc` 리포지토리 선택

### 3. 서비스 설정

**중요: Render가 Node.js 프로젝트로 자동 감지할 수 있으므로 Python을 명시적으로 선택해야 합니다.**

**Basic Settings:**
- Name: `rag-web-svc-backend`
- Region: `Oregon (US West)`
- Branch: `master`
- Root Directory: (비워두기 - 루트 디렉토리 사용)
- **Environment: `Python 3` (중요! Docker가 아닌 Python 선택)**
- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn enhanced_rag_chatbot:app --host 0.0.0.0 --port $PORT`

**만약 Environment 선택이 없다면:**
1. "You can use Render's native environments" 섹션에서
2. "Python 3" 선택
3. 또는 Settings > Environment에서 Python으로 변경

### 4. 환경 변수 설정
"Advanced" 섹션을 열고 다음 환경 변수를 추가:

1. **OPENAI_API_KEY**
   - Key: `OPENAI_API_KEY`
   - Value: `[여기에 OpenAI API 키 입력]`

2. **ALLOWED_ORIGINS**
   - Key: `ALLOWED_ORIGINS`
   - Value: `https://humanist96.github.io`

3. **PYTHON_VERSION** (선택사항)
   - Key: `PYTHON_VERSION`
   - Value: `3.11.9`

4. **RENDER** (자동 환경 감지용)
   - Key: `RENDER`
   - Value: `true`

### 5. 배포
1. "Create Web Service" 클릭
2. 배포가 시작되면 로그를 확인하여 진행 상황 모니터링
3. 배포가 완료되면 제공된 URL 확인 (예: `https://rag-web-svc-backend.onrender.com`)

### 6. 프론트엔드 연결
백엔드가 성공적으로 배포되면:

1. 백엔드 URL 복사 (예: `https://rag-web-svc-backend.onrender.com`)
2. `config.js` 파일 업데이트:
   ```javascript
   production: {
       API_URL: 'https://rag-web-svc-backend.onrender.com'  // 실제 URL로 변경
   }
   ```
3. 변경사항 커밋 및 푸시:
   ```bash
   git add config.js
   git commit -m "Update production API URL"
   git push
   ```

### 7. 테스트
1. 백엔드 API 문서 확인: `https://rag-web-svc-backend.onrender.com/docs`
2. 헬스체크: `https://rag-web-svc-backend.onrender.com/`
3. 프론트엔드에서 전체 기능 테스트:
   - https://humanist96.github.io/rag_web_svc/premium_index.html
   - PDF 업로드
   - 채팅 기능
   - 분석 대시보드

## 주의사항

### Free Tier 제한사항
- 15분 동안 요청이 없으면 서버가 스핀다운됨
- 첫 요청 시 30초 정도의 콜드 스타트 시간 필요
- 월 750시간 무료 사용 가능

### 성능 최적화
- 세션 자동 정리 기능이 활성화됨 (12시간 이상 미사용 세션)
- 최대 100개 세션으로 제한
- 파일 크기 50MB로 제한

### 모니터링
- Render 대시보드에서 로그 확인
- 메트릭 탭에서 메모리/CPU 사용량 모니터링
- 에러 발생 시 로그 확인

## 문제 해결

### "yarn" 또는 "package.json" 오류 발생 시
**이것은 Render가 프로젝트를 Node.js로 잘못 인식했을 때 발생합니다.**

해결 방법:
1. Render Dashboard에서 현재 서비스의 Settings 탭으로 이동
2. "Environment" 섹션에서 "Python 3"로 변경
3. 또는 서비스를 삭제하고 다시 생성할 때 Python 선택

### 배포 실패 시
1. Build logs에서 에러 메시지 확인
2. requirements.txt 파일이 있는지 확인
3. Python 버전 호환성 확인

### CORS 에러
1. ALLOWED_ORIGINS 환경 변수 확인
2. 프론트엔드 URL이 정확한지 확인 (trailing slash 없어야 함)

### API 키 에러
1. OPENAI_API_KEY가 올바르게 설정되었는지 확인
2. API 키의 사용량 한도 확인
# Render.com 수동 설정 가이드 (Python 강제 설정)

## 🚨 중요: Render가 Node.js로 잘못 인식하는 문제 해결

### 방법 1: Render Dashboard에서 직접 설정 (권장)

1. **Render Dashboard 로그인**
   - https://dashboard.render.com 접속
   - humanist96@gmail.com으로 로그인

2. **기존 서비스 삭제**
   - 잘못 생성된 서비스 선택
   - Settings → Delete Service

3. **새 서비스 생성 (Python 명시)**
   - "New +" 클릭
   - **"Web Service"** 선택
   - GitHub 리포지토리 연결
   - **중요: "Language" 드롭다운에서 반드시 "Python 3" 선택**

4. **서비스 설정**
   ```
   Name: rag-web-svc-backend
   Region: Oregon (US West)
   Branch: master
   Root Directory: (비워두기)
   
   Build Command: pip install -r requirements.txt
   Start Command: uvicorn enhanced_rag_chatbot:app --host 0.0.0.0 --port $PORT
   ```

5. **환경 변수 추가**
   - OPENAI_API_KEY: [당신의 API 키]
   - ALLOWED_ORIGINS: https://humanist96.github.io
   - PYTHON_VERSION: 3.11

### 방법 2: Render Blueprint 사용

1. **render.yaml 대신 UI 사용**
   - render.yaml을 삭제하거나 이름 변경
   - Render UI에서 모든 설정 수동 입력

### 방법 3: 빌드 스크립트 우회

render.yaml을 다음과 같이 수정:

```yaml
services:
  - type: web
    name: rag-web-svc-backend
    env: python
    buildCommand: |
      echo "Starting Python build..."
      python --version
      pip install -r requirements.txt
    startCommand: python -m uvicorn enhanced_rag_chatbot:app --host 0.0.0.0 --port $PORT
```

### 방법 4: Docker 배포 (최후의 수단)

Dockerfile을 생성하여 환경을 완전히 제어:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "enhanced_rag_chatbot:app", "--host", "0.0.0.0", "--port", "$PORT"]
```

## 🎯 즉시 해결 방법

### Render Dashboard에서:

1. **Services** 탭에서 현재 서비스 클릭
2. **Settings** 탭으로 이동
3. **Build & Deploy** 섹션에서:
   - Environment: **Docker** → **Python 3**로 변경
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn enhanced_rag_chatbot:app --host 0.0.0.0 --port $PORT`
4. **Save Changes**
5. **Manual Deploy** → **Deploy latest commit**

### 디버깅 명령어:

Render Shell에서 실행 (있다면):
```bash
ls -la
cat runtime.txt
python --version
which python
pip --version
```

## ⚠️ 주의사항

- render.yaml이 있어도 Render UI 설정이 우선됨
- 첫 배포 시 환경 자동 감지가 잘못될 수 있음
- Python 환경이 확실히 선택되었는지 재확인 필요

## 📞 추가 지원

여전히 문제가 있다면:
1. Render Support에 문의
2. 프로젝트가 Python임을 명시
3. 스크린샷과 함께 오류 로그 제공
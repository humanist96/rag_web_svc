# Vercel 환경 변수 설정 가이드

## 🔐 OPENAI_API_KEY 설정 방법

### 방법 1: Vercel Dashboard에서 설정 (권장)

1. **Vercel Dashboard 로그인**
   - https://vercel.com/dashboard

2. **프로젝트 선택**
   - `rag-web-svc` 프로젝트 클릭

3. **Settings 탭**
   - 상단 메뉴에서 "Settings" 클릭

4. **Environment Variables**
   - 왼쪽 메뉴에서 "Environment Variables" 클릭

5. **환경 변수 추가**
   - Key: `OPENAI_API_KEY`
   - Value: 실제 OpenAI API 키 입력
   - Environment: Production, Preview, Development 모두 체크
   - "Save" 클릭

### 방법 2: Vercel CLI로 설정

```bash
# 환경 변수 추가
vercel env add OPENAI_API_KEY

# Production에만 추가
vercel env add OPENAI_API_KEY production

# 모든 환경에 추가
vercel env add OPENAI_API_KEY production preview development
```

## 🚀 재배포

환경 변수 설정 후 재배포가 필요합니다:

```bash
# Production 재배포
vercel --prod

# 또는 Vercel Dashboard에서:
# 1. Deployments 탭
# 2. 최신 배포의 ... 메뉴 클릭
# 3. "Redeploy" 선택
```

## ✅ 배포 확인

### 1. Health Check
```bash
curl https://rag-web-svc.vercel.app/api/health
```

성공 응답:
```json
{"status": "healthy", "message": "PDF RAG Chatbot is running"}
```

### 2. 웹사이트 접속
- Production: https://rag-web-svc.vercel.app
- Preview: https://rag-web-svc-[hash].vercel.app

### 3. 기능 테스트
1. PDF 파일 업로드
2. 질문 입력 및 AI 응답 확인
3. Analytics 대시보드 확인

## 🔍 문제 해결

### 환경 변수 오류
- 오류: "Environment Variable 'OPENAI_API_KEY' not found"
- 해결: Dashboard에서 환경 변수 설정 후 재배포

### API 오류
- 오류: "Failed to upload" 또는 "Failed to get response"
- 해결: 
  1. 브라우저 개발자 도구(F12) → Network 탭 확인
  2. Console 에러 메시지 확인
  3. OPENAI_API_KEY가 올바른지 확인

### 로그 확인
```bash
# 최신 배포 로그
vercel logs

# 특정 URL의 로그
vercel logs https://rag-web-svc.vercel.app
```

## 📝 주의사항

- 환경 변수는 민감한 정보이므로 절대 코드에 직접 포함하지 않습니다
- `.env.local` 파일은 로컬 개발용입니다 (`.gitignore`에 포함)
- Vercel Dashboard에서 설정한 환경 변수는 서버(API)에서만 접근 가능합니다
- 프론트엔드에서는 환경 변수에 직접 접근할 수 없습니다
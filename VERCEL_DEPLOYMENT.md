# Vercel 배포 가이드

## 🚀 프론트엔드 Vercel 배포

### 사전 준비
- Vercel CLI 설치됨: `npm install -g vercel`
- GitHub 계정: humanist96@gmail.com

### 배포 명령어

```bash
# 1. Vercel 로그인 (처음 한 번만)
vercel login

# 2. 프로젝트 배포
vercel

# 3. Production 배포
vercel --prod
```

### 첫 배포 시 설정

Vercel CLI가 물어보는 설정:
1. **Set up and deploy?** → Y
2. **Which scope?** → 개인 계정 선택
3. **Link to existing project?** → N (새 프로젝트)
4. **Project name?** → rag-web-svc
5. **In which directory?** → ./ (현재 디렉토리)
6. **Override settings?** → N

### 배포 후 URL

- **Preview**: https://rag-web-svc-[hash].vercel.app
- **Production**: https://rag-web-svc.vercel.app

### 환경 변수 설정

Vercel Dashboard에서 설정 가능:
1. https://vercel.com/dashboard
2. 프로젝트 선택
3. Settings → Environment Variables

현재는 프론트엔드만 배포하므로 환경 변수 불필요

### 커스텀 도메인 (선택사항)

1. Settings → Domains
2. Add Domain
3. DNS 설정 안내 따르기

### 자동 배포

GitHub 연결 시 자동 배포 가능:
1. Settings → Git
2. Connect GitHub repository
3. `humanist96/rag_web_svc` 선택

### vercel.json 설정

```json
{
  "version": 2,
  "name": "rag-web-svc",
  "builds": [
    {
      "src": "premium_index.html",
      "use": "@vercel/static"
    }
  ],
  "routes": [
    {
      "src": "/",
      "dest": "/premium_index.html"
    }
  ]
}
```

## 백엔드 정보

백엔드는 여전히 Render.com에서 실행 중:
- **URL**: https://rag-web-svc.onrender.com
- **API Docs**: https://rag-web-svc.onrender.com/docs

## 테스트

1. 배포된 URL 접속
2. PDF 업로드 테스트
3. AI 채팅 기능 확인
4. Analytics 대시보드 확인

## 명령어 요약

```bash
# 개발 배포 (Preview)
vercel

# 프로덕션 배포
vercel --prod

# 로컬 개발 서버
vercel dev

# 배포 목록 확인
vercel ls

# 로그 확인
vercel logs [url]
```

## 장점

- **무료 호스팅**: 개인 프로젝트 무료
- **글로벌 CDN**: 빠른 로딩 속도
- **자동 HTTPS**: SSL 인증서 자동 제공
- **Preview 배포**: PR마다 미리보기 URL
- **간편한 배포**: 한 명령어로 배포
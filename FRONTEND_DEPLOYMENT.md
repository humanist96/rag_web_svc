# 프론트엔드 Render.com 배포 가이드

## 배포 방법

### 방법 1: render.yaml 사용 (권장)
1. GitHub에 코드가 푸시됨
2. Render Dashboard에서 "New +" → "Blueprint"
3. 리포지토리 선택 후 자동으로 frontend와 backend 모두 배포

### 방법 2: 수동 설정
1. Render Dashboard 로그인
2. "New +" → "Static Site" 선택
3. 설정 입력:
   - **Name**: rag-web-svc-frontend
   - **Branch**: master
   - **Build Command**: (비워두기)
   - **Publish Directory**: ./
   - **Auto-Deploy**: Yes

## 프론트엔드 구조

```
/
├── premium_index.html      # 메인 앱
├── premium_script.js       # JavaScript 로직
├── premium_styles.css      # 스타일시트
├── config.js              # API 설정
├── _headers               # HTTP 헤더 설정
└── _redirects             # URL 리다이렉트 규칙
```

## 배포 후 설정

### 1. 환경 변수
프론트엔드는 정적 사이트이므로 환경 변수가 필요 없습니다.
API URL은 config.js에서 자동으로 감지됩니다.

### 2. 커스텀 도메인 (선택사항)
- Settings → Custom Domains
- 도메인 추가 및 DNS 설정

### 3. HTTPS
- Render가 자동으로 SSL 인증서 제공
- 모든 트래픽이 HTTPS로 자동 리다이렉트

## 주요 기능

- **자동 배포**: GitHub 푸시 시 자동 재배포
- **글로벌 CDN**: 빠른 로딩 속도
- **무료 SSL**: HTTPS 자동 제공
- **커스텀 헤더**: 보안 및 캐싱 설정
- **SPA 지원**: 모든 경로가 index.html로 리다이렉트

## 접속 URL

배포 완료 후:
- https://rag-web-svc-frontend.onrender.com

## 테스트 체크리스트

- [ ] 메인 페이지 로딩
- [ ] 3D 배경 애니메이션
- [ ] PDF 업로드 기능
- [ ] AI 채팅 기능
- [ ] 세션 관리
- [ ] 분석 대시보드
- [ ] 반응형 디자인

## 문제 해결

### 페이지가 로드되지 않음
1. 빌드 로그 확인
2. 파일 경로 확인
3. Console 에러 확인

### API 연결 실패
1. config.js의 API URL 확인
2. CORS 설정 확인
3. 백엔드 서버 상태 확인
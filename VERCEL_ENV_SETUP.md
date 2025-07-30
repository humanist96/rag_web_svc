# Vercel 환경 변수 설정 가이드

## 1. Vercel Dashboard에서 설정

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

## 2. 재배포

환경 변수 설정 후:
```bash
vercel --prod
```

## 3. 확인

배포 후 테스트:
```bash
# Health check
curl https://rag-web-svc.vercel.app/api/health
```

## 주의사항

- 환경 변수는 민감한 정보이므로 절대 코드에 직접 포함하지 않습니다
- `.env.local` 파일은 로컬 개발용입니다
- Vercel Dashboard에서 설정한 환경 변수는 서버에서만 접근 가능합니다
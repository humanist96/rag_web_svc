import requests

# 테스트 PDF 파일 경로
pdf_file_path = "samilpwc_ai-agent-in-action.pdf"

# 업로드 URL
upload_url = "https://rag-web-svc.onrender.com/upload"

# 파일 업로드
with open(pdf_file_path, 'rb') as f:
    files = {'file': (pdf_file_path, f, 'application/pdf')}
    
    print(f"Uploading {pdf_file_path} to {upload_url}")
    
    response = requests.post(upload_url, files=files)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response Headers: {dict(response.headers)}")
    print(f"Response: {response.text}")
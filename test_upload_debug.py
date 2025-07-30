"""
Upload test and debug script
"""
import requests
import sys
import json
import os

# UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')

API_URL = "http://localhost:8001"

def test_server():
    """Test server health"""
    try:
        response = requests.get(f"{API_URL}/")
        print("✅ Server Status:", response.json())
        return True
    except Exception as e:
        print("❌ Server Connection Failed:", e)
        return False

def test_upload():
    """Test PDF upload"""
    # Check if PDF exists
    pdf_path = "samilpwc_ai-agent-in-action.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return False
    
    print(f"✅ PDF file found: {pdf_path} ({os.path.getsize(pdf_path)} bytes)")
    
    try:
        # Prepare file upload
        with open(pdf_path, 'rb') as f:
            files = {'file': (pdf_path, f, 'application/pdf')}
            
            print("\n📤 Uploading PDF...")
            response = requests.post(
                f"{API_URL}/upload",
                files=files
            )
            
            print(f"Response Status: {response.status_code}")
            print(f"Response Headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                result = response.json()
                print("\n✅ Upload Successful!")
                print(json.dumps(result, indent=2, ensure_ascii=False))
                return result.get('session_id')
            else:
                print("\n❌ Upload Failed!")
                print(f"Status Code: {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
    except Exception as e:
        print(f"\n❌ Upload Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_chat(session_id):
    """Test chat with uploaded PDF"""
    if not session_id:
        print("❌ No session ID provided")
        return
    
    try:
        data = {
            "session_id": session_id,
            "message": "이 문서는 무엇에 관한 내용입니까?"
        }
        
        print(f"\n💬 Testing chat with session: {session_id}")
        response = requests.post(
            f"{API_URL}/chat",
            json=data
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ Chat Response:")
            print(f"Answer: {result['answer'][:200]}...")
            if result.get('sources'):
                print(f"Sources: {len(result['sources'])} found")
        else:
            print(f"\n❌ Chat Failed: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"\n❌ Chat Error: {e}")
        import traceback
        traceback.print_exc()

def check_environment():
    """Check environment setup"""
    print("🔍 Checking Environment...")
    
    # Check OpenAI API key
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        print(f"✅ OpenAI API Key: {'*' * 10}{api_key[-4:]}")
    else:
        print("❌ OpenAI API Key: Not found")
    
    # Check uploads directory
    if os.path.exists("uploads"):
        print("✅ Uploads directory exists")
    else:
        print("❌ Uploads directory missing")
        os.makedirs("uploads", exist_ok=True)
        print("✅ Created uploads directory")
    
    # Check Python packages
    required_packages = [
        'fastapi', 'uvicorn', 'langchain', 'langchain-openai', 
        'langchain-community', 'faiss-cpu', 'pypdf'
    ]
    
    print("\n📦 Checking packages:")
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - Not installed")

def main():
    print("=== Premium RAG Chatbot Upload Test ===\n")
    
    # Check environment first
    check_environment()
    
    print("\n" + "="*50 + "\n")
    
    # Test server
    if not test_server():
        print("\n⚠️  Please make sure the server is running:")
        print("   python enhanced_rag_chatbot.py")
        return
    
    # Test upload
    print("\n" + "="*50 + "\n")
    session_id = test_upload()
    
    # Test chat if upload successful
    if session_id:
        print("\n" + "="*50 + "\n")
        test_chat(session_id)
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    main()
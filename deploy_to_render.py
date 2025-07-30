#!/usr/bin/env python3
"""
Render Blueprint Deployment Helper Script
Render API를 사용하여 Blueprint 배포를 자동화합니다.
"""

import os
import sys
import json
import requests
from typing import Dict, Optional

class RenderDeployer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.render.com/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def create_blueprint(self, repo_owner: str, repo_name: str, branch: str = "master") -> Dict:
        """Blueprint 생성"""
        endpoint = f"{self.base_url}/blueprints"
        payload = {
            "name": repo_name,
            "repo": {
                "provider": "github",
                "owner": repo_owner,
                "name": repo_name,
                "branch": branch
            }
        }
        
        response = requests.post(endpoint, json=payload, headers=self.headers)
        
        if response.status_code == 201:
            print(f"✅ Blueprint created successfully!")
            return response.json()
        else:
            print(f"❌ Failed to create blueprint: {response.status_code}")
            print(response.text)
            return {}
    
    def list_services(self) -> Dict:
        """서비스 목록 조회"""
        endpoint = f"{self.base_url}/services"
        response = requests.get(endpoint, headers=self.headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Failed to list services: {response.status_code}")
            return {}
    
    def get_service_status(self, service_id: str) -> Dict:
        """서비스 상태 조회"""
        endpoint = f"{self.base_url}/services/{service_id}"
        response = requests.get(endpoint, headers=self.headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Failed to get service status: {response.status_code}")
            return {}
    
    def update_env_vars(self, service_id: str, env_vars: Dict[str, str]) -> bool:
        """환경 변수 업데이트"""
        endpoint = f"{self.base_url}/services/{service_id}/env-vars"
        
        payload = []
        for key, value in env_vars.items():
            payload.append({
                "key": key,
                "value": value
            })
        
        response = requests.put(endpoint, json=payload, headers=self.headers)
        
        if response.status_code == 200:
            print(f"✅ Environment variables updated successfully!")
            return True
        else:
            print(f"❌ Failed to update env vars: {response.status_code}")
            return False

def main():
    """메인 함수"""
    # API 키 확인
    api_key = os.environ.get("RENDER_API_KEY")
    if not api_key:
        print("❌ RENDER_API_KEY environment variable not set!")
        print("Get your API key from: https://dashboard.render.com/account/api-keys")
        sys.exit(1)
    
    deployer = RenderDeployer(api_key)
    
    # GitHub 정보
    repo_owner = "humanist96"
    repo_name = "rag_web_svc"
    
    print(f"🚀 Deploying {repo_owner}/{repo_name} to Render...")
    
    # Blueprint 생성
    blueprint = deployer.create_blueprint(repo_owner, repo_name)
    
    if blueprint:
        print("\n📋 Blueprint Details:")
        print(f"ID: {blueprint.get('id')}")
        print(f"Name: {blueprint.get('name')}")
        print(f"Status: {blueprint.get('status')}")
        
        print("\n⏳ Waiting for services to be created...")
        print("This may take a few minutes...")
        
        # 서비스 목록 확인
        import time
        time.sleep(10)  # 서비스 생성을 위해 대기
        
        services = deployer.list_services()
        if services:
            print("\n📊 Services:")
            for service in services.get('items', []):
                print(f"- {service['name']} ({service['type']}): {service['serviceDetails']['url']}")
                
                # Backend 서비스에 OPENAI_API_KEY 설정 필요
                if 'backend' in service['name']:
                    print(f"\n⚠️  Remember to set OPENAI_API_KEY for {service['name']}!")
                    print(f"Go to: https://dashboard.render.com/services/{service['id']}/env")
        
        print("\n✅ Deployment initiated successfully!")
        print("\n📍 Next steps:")
        print("1. Go to https://dashboard.render.com")
        print("2. Monitor the build progress")
        print("3. Set OPENAI_API_KEY for the backend service")
        print("4. Test your deployed services")

if __name__ == "__main__":
    main()
// Model Selector UI Component

class ModelSelector {
    constructor() {
        this.apiUrl = window.API_URL || 'http://localhost:8001';
        this.currentModel = null;
        this.availableModels = {};
        this.providers = [];
        this.init();
    }

    async init() {
        await this.loadModels();
        this.createUI();
        this.setupEventListeners();
    }

    async loadModels() {
        try {
            // Use embedded model list instead of fetching from backend
            this.currentModel = {
                provider: 'Anthropic',
                model: 'claude-3-sonnet-20240229',
                display_name: 'Claude 3 Sonnet'
            };
            
            this.availableModels = {
                'OpenAI': [
                    { id: 'gpt-3.5-turbo', name: 'GPT-3.5 Turbo', description: '빠르고 효율적인 모델' },
                    { id: 'gpt-3.5-turbo-16k', name: 'GPT-3.5 Turbo 16K', description: '긴 컨텍스트 지원' },
                    { id: 'gpt-4', name: 'GPT-4', description: '가장 강력한 모델' },
                    { id: 'gpt-4-turbo-preview', name: 'GPT-4 Turbo', description: '최신 GPT-4 모델' }
                ],
                'Anthropic': [
                    { id: 'claude-3-opus-20240229', name: 'Claude 3 Opus', description: '가장 강력한 Claude 모델' },
                    { id: 'claude-3-sonnet-20240229', name: 'Claude 3 Sonnet', description: '균형잡힌 성능 (기본)' },
                    { id: 'claude-3-haiku-20240307', name: 'Claude 3 Haiku', description: '빠른 응답 속도' },
                    { id: 'claude-2.1', name: 'Claude 2.1', description: '이전 버전' },
                    { id: 'claude-instant-1.2', name: 'Claude Instant', description: '매우 빠른 응답' }
                ]
            };
            
            this.providers = ['OpenAI', 'Anthropic'];
            
            // Skip fetching from backend
            console.log('Using embedded model configuration');
        } catch (error) {
            console.error('Error in model configuration:', error);
        }
    }

    createUI() {
        // Model selector container
        const container = document.createElement('div');
        container.className = 'model-selector-container';
        container.innerHTML = `
            <div class="model-selector-header">
                <h3>AI 모델 설정</h3>
                <button class="close-btn" id="close-model-selector">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            
            <div class="model-selector-content">
                <!-- 현재 모델 정보 -->
                <div class="current-model-info">
                    <h4>현재 모델</h4>
                    <p class="current-model-display">
                        <span class="provider">${this.currentModel?.provider || 'OpenAI'}</span> - 
                        <span class="model">${this.currentModel?.model || 'gpt-3.5-turbo-16k'}</span>
                    </p>
                </div>

                <!-- 프로바이더 선택 -->
                <div class="form-group">
                    <label for="provider-select">AI 프로바이더</label>
                    <select id="provider-select" class="form-control">
                        ${this.providers.map(provider => 
                            `<option value="${provider}" ${provider === this.currentModel?.provider ? 'selected' : ''}>
                                ${this.getProviderName(provider)}
                            </option>`
                        ).join('')}
                    </select>
                </div>

                <!-- 모델 선택 -->
                <div class="form-group">
                    <label for="model-select">모델</label>
                    <select id="model-select" class="form-control">
                        <!-- 동적으로 채워집니다 -->
                    </select>
                    <small class="model-description"></small>
                </div>

                <!-- 고급 설정 -->
                <div class="advanced-settings">
                    <h4>고급 설정</h4>
                    
                    <div class="form-group">
                        <label for="temperature-slider">
                            Temperature: <span id="temperature-value">0.7</span>
                        </label>
                        <input type="range" id="temperature-slider" 
                               min="0" max="1" step="0.1" value="0.7"
                               class="slider">
                        <small>낮을수록 일관된 답변, 높을수록 창의적인 답변</small>
                    </div>

                    <div class="form-group">
                        <label for="max-tokens">최대 토큰 수</label>
                        <input type="number" id="max-tokens" 
                               value="2000" min="100" max="4000" 
                               class="form-control">
                        <small>응답의 최대 길이를 제한합니다</small>
                    </div>
                </div>

                <!-- API 키 설정 -->
                <div class="api-key-section">
                    <h4>API 키 설정</h4>
                    <div class="form-group">
                        <label for="api-key-input">
                            <span id="api-key-label">API 키</span>
                        </label>
                        <div class="api-key-input-group">
                            <input type="password" id="api-key-input" 
                                   class="form-control" 
                                   placeholder="API 키를 입력하세요">
                            <button id="toggle-api-key" class="btn-icon">
                                <i class="fas fa-eye"></i>
                            </button>
                        </div>
                        <button id="update-api-key" class="btn btn-secondary">
                            API 키 업데이트
                        </button>
                    </div>
                </div>

                <!-- 액션 버튼 -->
                <div class="model-selector-actions">
                    <button id="test-model" class="btn btn-secondary">
                        모델 테스트
                    </button>
                    <button id="apply-model" class="btn btn-primary">
                        적용
                    </button>
                </div>
            </div>
        `;

        // Add to page
        document.body.appendChild(container);
        this.container = container;

        // Update model dropdown
        this.updateModelDropdown();
    }

    setupEventListeners() {
        // Provider change
        document.getElementById('provider-select').addEventListener('change', (e) => {
            this.updateModelDropdown();
            this.updateApiKeyLabel(e.target.value);
        });

        // Temperature slider
        document.getElementById('temperature-slider').addEventListener('input', (e) => {
            document.getElementById('temperature-value').textContent = e.target.value;
        });

        // Toggle API key visibility
        document.getElementById('toggle-api-key').addEventListener('click', () => {
            const input = document.getElementById('api-key-input');
            const icon = document.querySelector('#toggle-api-key i');
            
            if (input.type === 'password') {
                input.type = 'text';
                icon.className = 'fas fa-eye-slash';
            } else {
                input.type = 'password';
                icon.className = 'fas fa-eye';
            }
        });

        // Update API key
        document.getElementById('update-api-key').addEventListener('click', () => {
            this.updateApiKey();
        });

        // Test model
        document.getElementById('test-model').addEventListener('click', () => {
            this.testModel();
        });

        // Apply model
        document.getElementById('apply-model').addEventListener('click', () => {
            this.applyModel();
        });

        // Close button
        document.getElementById('close-model-selector').addEventListener('click', () => {
            this.hide();
        });

        // Model description update
        document.getElementById('model-select').addEventListener('change', (e) => {
            this.updateModelDescription(e.target.value);
        });
    }

    updateModelDropdown() {
        const provider = document.getElementById('provider-select').value;
        const modelSelect = document.getElementById('model-select');
        const models = this.availableModels[provider] || [];

        modelSelect.innerHTML = models.map(model => 
            `<option value="${model.id}" ${model.id === this.currentModel?.model ? 'selected' : ''}>
                ${model.name}
            </option>`
        ).join('');

        // Update description for selected model
        if (models.length > 0) {
            this.updateModelDescription(modelSelect.value);
        }
    }

    updateModelDescription(modelId) {
        const provider = document.getElementById('provider-select').value;
        const models = this.availableModels[provider] || [];
        const model = models.find(m => m.id === modelId);
        
        const descElement = document.querySelector('.model-description');
        if (model && descElement) {
            descElement.textContent = model.description || '';
        }
    }

    updateApiKeyLabel(provider) {
        const label = document.getElementById('api-key-label');
        if (provider === 'openai') {
            label.textContent = 'OpenAI API 키';
        } else if (provider === 'claude') {
            label.textContent = 'Claude API 키';
        } else {
            label.textContent = 'API 키';
        }
    }

    async updateApiKey() {
        const provider = document.getElementById('provider-select').value;
        const apiKey = document.getElementById('api-key-input').value;

        if (!apiKey) {
            this.showNotification('API 키를 입력해주세요', 'warning');
            return;
        }

        try {
            const response = await fetch(`${this.apiUrl}/models/api-key`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    provider: provider,
                    api_key: apiKey
                })
            });

            const data = await response.json();
            
            if (response.ok) {
                this.showNotification(data.message, 'success');
                document.getElementById('api-key-input').value = '';
            } else {
                this.showNotification(data.detail || 'API 키 업데이트 실패', 'error');
            }
        } catch (error) {
            console.error('Error updating API key:', error);
            this.showNotification('API 키 업데이트 중 오류가 발생했습니다', 'error');
        }
    }

    async testModel() {
        const testBtn = document.getElementById('test-model');
        testBtn.disabled = true;
        testBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 테스트 중...';

        try {
            const response = await fetch(`${this.apiUrl}/models/test`);
            const data = await response.json();

            if (response.ok) {
                this.showNotification('모델 테스트 성공!', 'success');
                
                // Show test response
                const modal = document.createElement('div');
                modal.className = 'test-result-modal';
                modal.innerHTML = `
                    <div class="modal-content">
                        <h4>모델 테스트 결과</h4>
                        <p><strong>모델:</strong> ${data.model_info.provider} - ${data.model_info.model}</p>
                        <p><strong>응답:</strong></p>
                        <div class="test-response">${data.test_response}</div>
                        <button onclick="this.parentElement.parentElement.remove()" class="btn btn-primary">
                            확인
                        </button>
                    </div>
                `;
                document.body.appendChild(modal);
            } else {
                this.showNotification(data.detail || '모델 테스트 실패', 'error');
            }
        } catch (error) {
            console.error('Error testing model:', error);
            this.showNotification('모델 테스트 중 오류가 발생했습니다', 'error');
        } finally {
            testBtn.disabled = false;
            testBtn.innerHTML = '모델 테스트';
        }
    }

    async applyModel() {
        const provider = document.getElementById('provider-select').value;
        const modelName = document.getElementById('model-select').value;
        const temperature = parseFloat(document.getElementById('temperature-slider').value);
        const maxTokens = parseInt(document.getElementById('max-tokens').value);

        try {
            const response = await fetch(`${this.apiUrl}/models/select`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    provider: provider,
                    model_name: modelName,
                    temperature: temperature,
                    max_tokens: maxTokens
                })
            });

            const data = await response.json();

            if (response.ok) {
                this.showNotification(data.message, 'success');
                this.currentModel = data.current_model;
                this.updateCurrentModelDisplay();
                
                // Update main app model status
                if (window.updateModelStatus) {
                    window.updateModelStatus();
                }
                
                // Hide modal after success
                setTimeout(() => this.hide(), 1500);
            } else {
                this.showNotification(data.detail || '모델 선택 실패', 'error');
            }
        } catch (error) {
            console.error('Error applying model:', error);
            this.showNotification('모델 선택 중 오류가 발생했습니다', 'error');
        }
    }

    updateCurrentModelDisplay() {
        const display = document.querySelector('.current-model-display');
        if (display && this.currentModel) {
            display.innerHTML = `
                <span class="provider">${this.currentModel.provider}</span> - 
                <span class="model">${this.currentModel.model}</span>
            `;
        }
    }

    show() {
        this.container.style.display = 'block';
        setTimeout(() => {
            this.container.classList.add('show');
        }, 10);
    }

    hide() {
        this.container.classList.remove('show');
        setTimeout(() => {
            this.container.style.display = 'none';
        }, 300);
    }

    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
            <span>${message}</span>
        `;

        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.classList.add('show');
        }, 10);

        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }

    getProviderName(provider) {
        const names = {
            'openai': 'OpenAI',
            'claude': 'Claude (Anthropic)'
        };
        return names[provider] || provider;
    }
}

// Initialize when DOM is ready
let modelSelector;
document.addEventListener('DOMContentLoaded', () => {
    modelSelector = new ModelSelector();
    
    // Add model selector button to UI
    const settingsBtn = document.createElement('button');
    settingsBtn.className = 'btn-model-settings';
    settingsBtn.innerHTML = '<i class="fas fa-cog"></i> AI 모델 설정';
    settingsBtn.onclick = () => modelSelector.show();
    
    // Add to header or appropriate location
    const header = document.querySelector('.header-actions') || document.querySelector('.chat-header-actions');
    if (header) {
        header.appendChild(settingsBtn);
    }
    
    // Also make model status clickable
    document.addEventListener('click', (e) => {
        if (e.target.closest('.model-status')) {
            modelSelector.show();
        }
    });
});

// Export for global access
window.ModelSelector = ModelSelector;
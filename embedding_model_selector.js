/**
 * Embedding Model and LLM Selection System
 * Handles model selection UI and compatibility matching
 */

class EmbeddingModelSelector {
    constructor() {
        this.selectedEmbeddingModel = null;
        this.selectedLLM = null;
        this.fileInfo = null;
        this.onSelectionComplete = null;
        
        this.embeddingModels = {
            'openai-ada-002': {
                name: 'OpenAI Ada-002',
                description: '고성능 텍스트 임베딩 모델, 다양한 언어 지원',
                dimensions: 1536,
                maxTokens: 8191,
                costPer1K: 0.0001,
                performance: {
                    accuracy: 90,
                    speed: 85,
                    languages: 95
                },
                compatibleLLMs: ['gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-4', 'gpt-4-turbo-preview'],
                specs: ['1536차원', '8K 토큰', '다국어', '고정밀']
            },
            'openai-embedding-3-small': {
                name: 'OpenAI Embedding v3 Small',
                description: '효율적이면서도 정확한 최신 임베딩 모델',
                dimensions: 1536,
                maxTokens: 8191,
                costPer1K: 0.00002,
                performance: {
                    accuracy: 85,
                    speed: 95,
                    languages: 90
                },
                compatibleLLMs: ['gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-4', 'gpt-4-turbo-preview'],
                specs: ['1536차원', '8K 토큰', '고속', '경제적']
            },
            'openai-embedding-3-large': {
                name: 'OpenAI Embedding v3 Large',
                description: '최고 성능의 임베딩 모델, 복잡한 문서 분석에 최적',
                dimensions: 3072,
                maxTokens: 8191,
                costPer1K: 0.00013,
                performance: {
                    accuracy: 95,
                    speed: 75,
                    languages: 95
                },
                compatibleLLMs: ['gpt-4', 'gpt-4-turbo-preview'],
                specs: ['3072차원', '8K 토큰', '최고정밀', '대용량']
            },
            'huggingface-sentence-transformers': {
                name: 'Sentence Transformers',
                description: '오픈소스 임베딩 모델, 무료 사용 가능',
                dimensions: 768,
                maxTokens: 512,
                costPer1K: 0,
                performance: {
                    accuracy: 75,
                    speed: 90,
                    languages: 70
                },
                compatibleLLMs: ['claude-3-haiku-20240307', 'claude-instant-1.2'],
                specs: ['768차원', '512 토큰', '무료', '오픈소스']
            },
            'cohere-embed-multilingual': {
                name: 'Cohere Embed Multilingual',
                description: '다국어 특화 임베딩 모델, 100개 이상 언어 지원',
                dimensions: 1024,
                maxTokens: 2048,
                costPer1K: 0.0001,
                performance: {
                    accuracy: 88,
                    speed: 80,
                    languages: 98
                },
                compatibleLLMs: ['claude-3-sonnet-20240229', 'claude-3-haiku-20240307'],
                specs: ['1024차원', '2K 토큰', '100개 언어', '다국어 특화']
            }
        };
        
        this.llmModels = {
            'gpt-3.5-turbo': {
                name: 'GPT-3.5 Turbo',
                description: '빠르고 효율적인 대화형 AI 모델',
                provider: 'OpenAI',
                contextWindow: 4096,
                costPer1K: 0.002,
                performance: {
                    reasoning: 80,
                    creativity: 75,
                    speed: 95
                }
            },
            'gpt-3.5-turbo-16k': {
                name: 'GPT-3.5 Turbo 16K',
                description: '긴 문서 처리가 가능한 확장 버전',
                provider: 'OpenAI',
                contextWindow: 16384,
                costPer1K: 0.004,
                performance: {
                    reasoning: 80,
                    creativity: 75,
                    speed: 90
                }
            },
            'gpt-4': {
                name: 'GPT-4',
                description: '최고 수준의 추론 능력을 가진 프리미엄 모델',
                provider: 'OpenAI',
                contextWindow: 8192,
                costPer1K: 0.03,
                performance: {
                    reasoning: 95,
                    creativity: 90,
                    speed: 70
                }
            },
            'gpt-4-turbo-preview': {
                name: 'GPT-4 Turbo',
                description: '최신 정보와 긴 컨텍스트를 지원하는 최신 모델',
                provider: 'OpenAI',
                contextWindow: 128000,
                costPer1K: 0.01,
                performance: {
                    reasoning: 95,
                    creativity: 90,
                    speed: 85
                }
            },
            'claude-3-opus-20240229': {
                name: 'Claude 3 Opus',
                description: '가장 강력한 Claude 모델, 복잡한 분석과 추론에 최적',
                provider: 'Anthropic',
                contextWindow: 200000,
                costPer1K: 0.015,
                performance: {
                    reasoning: 98,
                    creativity: 95,
                    speed: 60
                }
            },
            'claude-3-sonnet-20240229': {
                name: 'Claude 3 Sonnet',
                description: '균형잡힌 성능과 속도를 제공하는 다재다능한 모델',
                provider: 'Anthropic',
                contextWindow: 200000,
                costPer1K: 0.003,
                performance: {
                    reasoning: 90,
                    creativity: 85,
                    speed: 80
                }
            },
            'claude-3-haiku-20240307': {
                name: 'Claude 3 Haiku',
                description: '빠른 응답과 효율적인 처리를 위한 경량 모델',
                provider: 'Anthropic',
                contextWindow: 200000,
                costPer1K: 0.00025,
                performance: {
                    reasoning: 80,
                    creativity: 75,
                    speed: 95
                }
            }
        };
        
        this.init();
    }
    
    init() {
        this.createModal();
        this.setupEventListeners();
    }
    
    createModal() {
        const modalHTML = `
            <div class="model-selection-overlay" id="model-selection-overlay">
                <div class="model-selection-modal">
                    <div class="modal-header">
                        <h2 class="modal-title">
                            <i class="fas fa-cog"></i>
                            모델 선택
                        </h2>
                        <p class="modal-subtitle">파일 업로드 전에 임베딩 모델과 LLM을 선택해주세요</p>
                    </div>
                    
                    <div class="modal-content">
                        <div class="file-info-display" id="file-info-display">
                            <div class="file-info-title">
                                <i class="fas fa-file"></i>
                                업로드할 파일
                            </div>
                            <div class="file-info-details" id="file-info-details">
                                <!-- File info will be populated here -->
                            </div>
                        </div>
                        
                        <div class="selection-grid">
                            <div class="selection-section">
                                <h3 class="section-title">
                                    <i class="fas fa-vector-square"></i>
                                    임베딩 모델
                                </h3>
                                <p class="section-description">
                                    문서를 벡터로 변환하여 의미적 검색을 가능하게 하는 모델을 선택하세요
                                </p>
                                <div class="model-options" id="embedding-options">
                                    <!-- Embedding models will be populated here -->
                                </div>
                            </div>
                            
                            <div class="selection-section">
                                <h3 class="section-title">
                                    <i class="fas fa-brain"></i>
                                    LLM 모델
                                </h3>
                                <p class="section-description">
                                    질문에 답변을 생성할 대화형 AI 모델을 선택하세요
                                </p>
                                <div class="model-options" id="llm-options">
                                    <!-- LLM models will be populated here -->
                                </div>
                            </div>
                        </div>
                        
                        <div class="compatibility-info" id="compatibility-info" style="display: none;">
                            <div class="compatibility-title">
                                <i class="fas fa-link"></i>
                                모델 호환성 정보
                            </div>
                            <div class="compatibility-grid" id="compatibility-grid">
                                <!-- Compatibility info will be populated here -->
                            </div>
                        </div>
                    </div>
                    
                    <div class="modal-actions">
                        <div class="selection-info">
                            <span id="selection-status">임베딩 모델과 LLM을 선택해주세요</span>
                        </div>
                        <div class="action-buttons">
                            <button class="btn btn-cancel" onclick="embeddingSelector.cancel()">
                                <i class="fas fa-times"></i>
                                취소
                            </button>
                            <button class="btn btn-confirm" id="confirm-selection" disabled onclick="embeddingSelector.confirm()">
                                <i class="fas fa-check"></i>
                                선택 완료
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        document.body.insertAdjacentHTML('beforeend', modalHTML);
    }
    
    setupEventListeners() {
        // Close modal on overlay click
        document.getElementById('model-selection-overlay').addEventListener('click', (e) => {
            if (e.target.id === 'model-selection-overlay') {
                this.cancel();
            }
        });
        
        // ESC key to close
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.isVisible()) {
                this.cancel();
            }
        });
    }
    
    show(fileInfo, onComplete) {
        this.fileInfo = fileInfo;
        this.onSelectionComplete = onComplete;
        this.selectedEmbeddingModel = null;
        this.selectedLLM = null;
        
        this.populateFileInfo();
        this.populateEmbeddingModels();
        this.populateLLMModels();
        this.updateSelectionStatus();
        
        document.getElementById('model-selection-overlay').classList.add('active');
        document.body.style.overflow = 'hidden';
    }
    
    hide() {
        document.getElementById('model-selection-overlay').classList.remove('active');
        document.body.style.overflow = '';
    }
    
    isVisible() {
        return document.getElementById('model-selection-overlay').classList.contains('active');
    }
    
    populateFileInfo() {
        const container = document.getElementById('file-info-details');
        const fileSize = this.formatFileSize(this.fileInfo.size);
        const fileType = this.fileInfo.type || 'Unknown';
        
        container.innerHTML = `
            <div class="file-info-item">
                <span class="file-info-label">파일명</span>
                <span class="file-info-value">${this.fileInfo.name}</span>
            </div>
            <div class="file-info-item">
                <span class="file-info-label">크기</span>
                <span class="file-info-value">${fileSize}</span>
            </div>
            <div class="file-info-item">
                <span class="file-info-label">타입</span>
                <span class="file-info-value">${fileType}</span>
            </div>
        `;
    }
    
    populateEmbeddingModels() {
        const container = document.getElementById('embedding-options');
        container.innerHTML = '';
        
        Object.entries(this.embeddingModels).forEach(([key, model]) => {
            const optionElement = this.createModelOption(key, model, 'embedding');
            container.appendChild(optionElement);
        });
    }
    
    populateLLMModels() {
        const container = document.getElementById('llm-options');
        container.innerHTML = '';
        
        Object.entries(this.llmModels).forEach(([key, model]) => {
            const optionElement = this.createModelOption(key, model, 'llm');
            container.appendChild(optionElement);
        });
    }
    
    createModelOption(key, model, type) {
        const option = document.createElement('div');
        option.className = 'model-option';
        option.dataset.modelKey = key;
        option.dataset.modelType = type;
        
        const performanceHTML = type === 'embedding' ? `
            <div class="performance-indicator">
                <span class="perf-label">정확도</span>
                <div class="perf-bar">
                    <div class="perf-fill ${this.getPerformanceClass(model.performance.accuracy)}" 
                         style="width: ${model.performance.accuracy}%"></div>
                </div>
            </div>
            <div class="performance-indicator">
                <span class="perf-label">속도</span>
                <div class="perf-bar">
                    <div class="perf-fill ${this.getPerformanceClass(model.performance.speed)}" 
                         style="width: ${model.performance.speed}%"></div>
                </div>
            </div>
        ` : `
            <div class="performance-indicator">
                <span class="perf-label">추론력</span>
                <div class="perf-bar">
                    <div class="perf-fill ${this.getPerformanceClass(model.performance.reasoning)}" 
                         style="width: ${model.performance.reasoning}%"></div>
                </div>
            </div>
            <div class="performance-indicator">
                <span class="perf-label">창의성</span>
                <div class="perf-bar">
                    <div class="perf-fill ${this.getPerformanceClass(model.performance.creativity)}" 
                         style="width: ${model.performance.creativity}%"></div>
                </div>
            </div>
        `;
        
        option.innerHTML = `
            <div class="model-name">${model.name}</div>
            <div class="model-description">${model.description}</div>
            ${performanceHTML}
            <div class="model-specs">
                ${model.specs ? model.specs.map(spec => `<span class="spec-tag">${spec}</span>`).join('') : ''}
                ${type === 'embedding' ? `<span class="spec-tag highlight">$${model.costPer1K}/1K</span>` : `<span class="spec-tag highlight">$${model.costPer1K}/1K</span>`}
            </div>
        `;
        
        option.addEventListener('click', () => this.selectModel(key, type));
        
        return option;
    }
    
    getPerformanceClass(value) {
        if (value >= 90) return 'high';
        if (value >= 70) return 'medium';
        return 'low';
    }
    
    selectModel(key, type) {
        // Remove previous selection
        document.querySelectorAll(`.model-option[data-model-type="${type}"]`).forEach(option => {
            option.classList.remove('selected');
        });
        
        // Add selection to clicked option
        const option = document.querySelector(`.model-option[data-model-key="${key}"]`);
        option.classList.add('selected');
        
        if (type === 'embedding') {
            this.selectedEmbeddingModel = key;
            this.updateCompatibleLLMs();
        } else {
            this.selectedLLM = key;
        }
        
        this.updateSelectionStatus();
        this.updateCompatibilityInfo();
    }
    
    updateCompatibleLLMs() {
        if (!this.selectedEmbeddingModel) return;
        
        const embeddingModel = this.embeddingModels[this.selectedEmbeddingModel];
        const compatibleLLMs = embeddingModel.compatibleLLMs;
        
        document.querySelectorAll('.model-option[data-model-type="llm"]').forEach(option => {
            const key = option.dataset.modelKey;
            if (compatibleLLMs.includes(key)) {
                option.style.opacity = '1';
                option.style.pointerEvents = 'auto';
            } else {
                option.style.opacity = '0.5';
                option.style.pointerEvents = 'none';
                option.classList.remove('selected');
                if (this.selectedLLM === key) {
                    this.selectedLLM = null;
                }
            }
        });
        
        this.updateSelectionStatus();
    }
    
    updateSelectionStatus() {
        const statusElement = document.getElementById('selection-status');
        const confirmButton = document.getElementById('confirm-selection');
        
        if (this.selectedEmbeddingModel && this.selectedLLM) {
            statusElement.textContent = '모델 선택 완료';
            confirmButton.disabled = false;
        } else if (this.selectedEmbeddingModel) {
            statusElement.textContent = 'LLM 모델을 선택해주세요';
            confirmButton.disabled = true;
        } else {
            statusElement.textContent = '임베딩 모델과 LLM을 선택해주세요';
            confirmButton.disabled = true;
        }
    }
    
    updateCompatibilityInfo() {
        const infoContainer = document.getElementById('compatibility-info');
        const gridContainer = document.getElementById('compatibility-grid');
        
        if (this.selectedEmbeddingModel && this.selectedLLM) {
            const embedding = this.embeddingModels[this.selectedEmbeddingModel];
            const llm = this.llmModels[this.selectedLLM];
            
            gridContainer.innerHTML = `
                <div class="compatibility-item">
                    <div class="compatibility-item-title">벡터 차원</div>
                    <div class="compatibility-item-desc">${embedding.dimensions}차원 임베딩</div>
                </div>
                <div class="compatibility-item">
                    <div class="compatibility-item-title">컨텍스트 윈도우</div>
                    <div class="compatibility-item-desc">${llm.contextWindow.toLocaleString()} 토큰</div>
                </div>
                <div class="compatibility-item">
                    <div class="compatibility-item-title">예상 비용</div>
                    <div class="compatibility-item-desc">임베딩: $${embedding.costPer1K}/1K + LLM: $${llm.costPer1K}/1K</div>
                </div>
                <div class="compatibility-item">
                    <div class="compatibility-item-title">최적 용도</div>
                    <div class="compatibility-item-desc">${this.getOptimalUseCase(embedding, llm)}</div>
                </div>
            `;
            
            infoContainer.style.display = 'block';
        } else {
            infoContainer.style.display = 'none';
        }
    }
    
    getOptimalUseCase(embedding, llm) {
        const embeddingPerf = embedding.performance.accuracy;
        const llmPerf = llm.performance.reasoning;
        
        if (embeddingPerf >= 90 && llmPerf >= 90) {
            return '복잡한 문서 분석, 전문적 질의응답';
        } else if (embeddingPerf >= 80 && llmPerf >= 80) {
            return '일반적인 문서 검색, 비즈니스 분석';
        } else {
            return '빠른 검색, 간단한 질의응답';
        }
    }
    
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    confirm() {
        if (this.selectedEmbeddingModel && this.selectedLLM && this.onSelectionComplete) {
            const selection = {
                embedding: {
                    key: this.selectedEmbeddingModel,
                    model: this.embeddingModels[this.selectedEmbeddingModel]
                },
                llm: {
                    key: this.selectedLLM,
                    model: this.llmModels[this.selectedLLM]
                },
                fileInfo: this.fileInfo
            };
            
            this.onSelectionComplete(selection);
            this.hide();
        }
    }
    
    cancel() {
        this.hide();
        this.selectedEmbeddingModel = null;
        this.selectedLLM = null;
        this.fileInfo = null;
        this.onSelectionComplete = null;
    }
    
    // Get recommended models based on file characteristics
    getRecommendedModels(fileInfo) {
        const fileSize = fileInfo.size;
        const fileType = fileInfo.type;
        
        let recommendedEmbedding, recommendedLLM;
        
        // Size-based recommendations
        if (fileSize > 10 * 1024 * 1024) { // > 10MB
            recommendedEmbedding = 'openai-embedding-3-large';
            recommendedLLM = 'gpt-4-turbo-preview';
        } else if (fileSize > 1 * 1024 * 1024) { // > 1MB
            recommendedEmbedding = 'openai-ada-002';
            recommendedLLM = 'gpt-3.5-turbo-16k';
        } else {
            recommendedEmbedding = 'openai-embedding-3-small';
            recommendedLLM = 'gpt-3.5-turbo';
        }
        
        // Type-based adjustments
        if (fileType && fileType.includes('csv')) {
            recommendedLLM = 'gpt-4'; // Better for data analysis
        }
        
        return {
            embedding: recommendedEmbedding,
            llm: recommendedLLM
        };
    }
    
    showRecommendations(fileInfo) {
        const recommendations = this.getRecommendedModels(fileInfo);
        
        // Highlight recommended options
        setTimeout(() => {
            const embeddingOption = document.querySelector(`[data-model-key="${recommendations.embedding}"]`);
            const llmOption = document.querySelector(`[data-model-key="${recommendations.llm}"]`);
            
            if (embeddingOption) {
                embeddingOption.style.border = '2px solid #10b981';
                embeddingOption.insertAdjacentHTML('afterbegin', '<div style="background: #10b981; color: white; padding: 4px 8px; border-radius: 4px; font-size: 0.7rem; position: absolute; top: 8px; right: 8px;">추천</div>');
            }
            
            if (llmOption) {
                llmOption.style.border = '2px solid #10b981';
                llmOption.insertAdjacentHTML('afterbegin', '<div style="background: #10b981; color: white; padding: 4px 8px; border-radius: 4px; font-size: 0.7rem; position: absolute; top: 8px; right: 8px;">추천</div>');
            }
        }, 300);
    }
}

// Global instance
const embeddingSelector = new EmbeddingModelSelector();
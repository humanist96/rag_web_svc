/**
 * Fallback embedding selector for when the main selector fails to load
 */

// Create a simple fallback if the main embedding selector fails
if (typeof embeddingSelector === 'undefined') {
    console.log('Creating fallback embedding selector...');
    
    window.embeddingSelector = {
        show: function(fileInfo, callback) {
            console.log('Using fallback embedding selector');
            
            // Create a simple modal for model selection
            const fallbackModal = this.createFallbackModal(fileInfo, callback);
            document.body.appendChild(fallbackModal);
            fallbackModal.classList.add('active');
        },
        
        createFallbackModal: function(fileInfo, callback) {
            const modal = document.createElement('div');
            modal.className = 'model-selection-overlay active';
            modal.innerHTML = `
                <div class="model-selection-modal" style="max-width: 600px;">
                    <div class="modal-header">
                        <h2 class="modal-title">
                            <i class="fas fa-cog"></i>
                            모델 선택
                        </h2>
                        <p class="modal-subtitle">기본 모델로 진행하거나 선택해주세요</p>
                    </div>
                    
                    <div class="modal-content" style="padding: 20px;">
                        <div class="file-info" style="background: var(--bg-secondary, #2a2a2a); padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                            <strong>파일:</strong> ${fileInfo.name}<br>
                            <strong>크기:</strong> ${this.formatFileSize(fileInfo.size)}
                        </div>
                        
                        <div class="model-options">
                            <div class="option-group" style="margin-bottom: 20px;">
                                <h3>임베딩 모델</h3>
                                <select id="fallback-embedding" style="width: 100%; padding: 10px; border-radius: 5px; border: 1px solid #ccc;">
                                    <option value="openai-ada-002">OpenAI Ada-002 (추천)</option>
                                    <option value="openai-embedding-3-small">OpenAI Embedding v3 Small</option>
                                    <option value="huggingface-sentence-transformers">HuggingFace (무료)</option>
                                </select>
                            </div>
                            
                            <div class="option-group">
                                <h3>LLM 모델</h3>
                                <select id="fallback-llm" style="width: 100%; padding: 10px; border-radius: 5px; border: 1px solid #ccc;">
                                    <option value="gpt-3.5-turbo">GPT-3.5 Turbo (추천)</option>
                                    <option value="gpt-4">GPT-4</option>
                                    <option value="claude-3-haiku-20240307">Claude 3 Haiku</option>
                                </select>
                            </div>
                        </div>
                    </div>
                    
                    <div class="modal-actions" style="padding: 20px; border-top: 1px solid #eee; display: flex; justify-content: space-between;">
                        <button class="btn btn-cancel" onclick="this.closest('.model-selection-overlay').remove()">
                            취소
                        </button>
                        <button class="btn btn-confirm" onclick="embeddingSelector.confirmFallbackSelection(this, '${JSON.stringify(fileInfo).replace(/'/g, "\\'")}', arguments[0])">
                            선택 완료
                        </button>
                    </div>
                </div>
            `;
            
            // Store callback for later use
            modal._callback = callback;
            return modal;
        },
        
        confirmFallbackSelection: function(button, fileInfoStr, event) {
            const modal = button.closest('.model-selection-overlay');
            const embeddingSelect = modal.querySelector('#fallback-embedding');
            const llmSelect = modal.querySelector('#fallback-llm');
            
            const fileInfo = JSON.parse(fileInfoStr);
            
            // Create selection object matching expected format
            const selection = {
                embedding: {
                    key: embeddingSelect.value,
                    model: {
                        name: embeddingSelect.options[embeddingSelect.selectedIndex].text.split(' (')[0],
                        description: 'Fallback selection'
                    }
                },
                llm: {
                    key: llmSelect.value,
                    model: {
                        name: llmSelect.options[llmSelect.selectedIndex].text.split(' (')[0],
                        description: 'Fallback selection'
                    }
                },
                fileInfo: fileInfo
            };
            
            // Call the callback
            if (modal._callback) {
                modal._callback(selection);
            }
            
            // Remove modal
            modal.remove();
        },
        
        formatFileSize: function(bytes) {
            if (bytes === 0) return '0 Bytes';
            const k = 1024;
            const sizes = ['Bytes', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
        },
        
        showRecommendations: function(fileInfo) {
            // Simple console log for fallback
            console.log('File recommendations:', fileInfo);
        }
    };
    
    console.log('Fallback embedding selector created');
}

// Also provide a simple CSS fallback
if (!document.querySelector('style[data-fallback-css]')) {
    const fallbackCSS = document.createElement('style');
    fallbackCSS.setAttribute('data-fallback-css', 'true');
    fallbackCSS.textContent = `
        .model-selection-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            backdrop-filter: blur(10px);
            z-index: 1000;
            display: flex;
            align-items: center;
            justify-content: center;
            opacity: 0;
            visibility: hidden;
            transition: opacity 0.3s ease, visibility 0.3s ease;
        }
        
        .model-selection-overlay.active {
            opacity: 1;
            visibility: visible;
        }
        
        .model-selection-modal {
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            width: 90%;
            max-width: 500px;
            max-height: 90vh;
            overflow: hidden;
            color: #333;
        }
        
        .modal-header {
            padding: 20px;
            border-bottom: 1px solid #eee;
            background: #f8f9fa;
        }
        
        .modal-title {
            margin: 0;
            font-size: 1.2rem;
            color: #333;
        }
        
        .modal-subtitle {
            margin: 5px 0 0;
            color: #666;
            font-size: 0.9rem;
        }
        
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        
        .btn-cancel {
            background: #6c757d;
            color: white;
        }
        
        .btn-cancel:hover {
            background: #5a6268;
        }
        
        .btn-confirm {
            background: #007bff;
            color: white;
        }
        
        .btn-confirm:hover {
            background: #0056b3;
        }
        
        .option-group h3 {
            margin: 0 0 10px;
            font-size: 1rem;
            color: #333;
        }
        
        [data-theme="dark"] .model-selection-modal {
            background: #2a2a2a;
            color: #fff;
        }
        
        [data-theme="dark"] .modal-header {
            background: #333;
            border-color: #555;
        }
        
        [data-theme="dark"] .modal-title,
        [data-theme="dark"] .option-group h3 {
            color: #fff;
        }
        
        [data-theme="dark"] .modal-subtitle {
            color: #ccc;
        }
    `;
    document.head.appendChild(fallbackCSS);
}
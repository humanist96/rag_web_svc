// Enhanced Chat UI with Advanced Features

class EnhancedChatUI {
    constructor() {
        this.messageContainer = document.getElementById('chat-messages');
        this.inputElement = document.getElementById('chat-input');
        this.sendButton = document.getElementById('send-button');
        this.isTyping = false;
        this.streamingMessage = null;
        this.chatHistory = [];
        this.suggestionEngine = new SuggestionEngine();
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupMarkdownRenderer();
        this.setupCodeHighlighting();
        this.loadChatHistory();
    }

    setupEventListeners() {
        // Enhanced input handling
        this.inputElement.addEventListener('input', (e) => {
            this.autoResizeTextarea(e.target);
            this.showSuggestions(e.target.value);
        });

        // Smart enter key handling
        this.inputElement.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            } else if (e.key === 'Tab' && this.suggestionEngine.hasActiveSuggestion()) {
                e.preventDefault();
                this.applySuggestion();
            }
        });

        // Voice input
        const voiceButton = document.querySelector('.voice-input-btn');
        if (voiceButton) {
            voiceButton.addEventListener('click', () => this.startVoiceInput());
        }
    }

    setupMarkdownRenderer() {
        // Configure marked.js for better markdown rendering
        if (typeof marked !== 'undefined') {
            marked.setOptions({
                highlight: function(code, lang) {
                    if (lang && hljs.getLanguage(lang)) {
                        return hljs.highlight(code, { language: lang }).value;
                    }
                    return hljs.highlightAuto(code).value;
                },
                breaks: true,
                gfm: true,
                tables: true
            });
        }
    }

    setupCodeHighlighting() {
        // Load highlight.js for code syntax highlighting
        if (typeof hljs !== 'undefined') {
            hljs.configure({
                languages: ['python', 'javascript', 'sql', 'json', 'bash']
            });
        }
    }

    sendMessage() {
        const message = this.inputElement.value.trim();
        if (!message) return;

        // Add user message to chat
        this.addMessage('user', message);
        
        // Clear input
        this.inputElement.value = '';
        this.autoResizeTextarea(this.inputElement);
        
        // Show typing indicator
        this.showTypingIndicator();
        
        // Send to backend
        this.sendToBackend(message);
    }

    async sendToBackend(message) {
        try {
            const sessionId = window.currentSessionId || 'default';
            
            // Enhance question before sending
            const enhancedMessage = this.enhanceQuestion(message);
            
            const response = await fetch(`${API_URL}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    session_id: sessionId,
                    message: enhancedMessage,
                    options: {
                        streaming: true,
                        include_sources: true,
                        format: 'markdown'
                    }
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            // Handle streaming response
            if (response.headers.get('content-type')?.includes('text/event-stream')) {
                await this.handleStreamingResponse(response);
            } else {
                const data = await response.json();
                this.handleNormalResponse(data);
            }

        } catch (error) {
            console.error('Chat error:', error);
            this.hideTypingIndicator();
            this.addMessage('error', '죄송합니다. 오류가 발생했습니다. 다시 시도해주세요.');
        }
    }

    async handleStreamingResponse(response) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        
        // Create streaming message container
        this.streamingMessage = this.createStreamingMessage();
        
        try {
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';
                
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = line.slice(6);
                        if (data === '[DONE]') {
                            this.finalizeStreamingMessage();
                            return;
                        }
                        
                        try {
                            const parsed = JSON.parse(data);
                            if (parsed.type === 'error') {
                                console.error('Streaming error:', parsed.error);
                                this.finalizeStreamingMessage();
                                this.addMessage('error', parsed.error);
                                return;
                            } else if (parsed.content) {
                                this.updateStreamingMessage(parsed.content);
                            }
                            
                            // Store metadata if available
                            if (parsed.metadata) {
                                this.currentStreamMetadata = parsed.metadata;
                            }
                        } catch (e) {
                            console.error('Parse error:', e);
                        }
                    }
                }
            }
        } finally {
            this.hideTypingIndicator();
        }
    }

    handleNormalResponse(data) {
        this.hideTypingIndicator();
        
        // Check if it's an off-topic response
        if (data.is_off_topic) {
            // Add warning-style message
            this.addMessage('assistant', data.answer, null, 'off-topic');
            
            // Show suggestions if available
            if (data.suggestions && data.suggestions.length > 0) {
                this.displaySuggestions(data.suggestions);
            }
            
            // Add relevance score info
            if (data.relevance_score !== undefined) {
                console.log(`Relevance score: ${data.relevance_score.toFixed(3)}`);
            }
        } else {
            // Add AI response with enhanced formatting
            const formattedAnswer = this.formatResponse(data.answer);
            this.addMessage('assistant', formattedAnswer, data.sources);
            
            // Generate follow-up suggestions
            this.generateFollowUpSuggestions(data.answer);
        }
        
        // Save to chat history
        this.saveChatHistory();
    }

    createStreamingMessage() {
        this.hideTypingIndicator();
        
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant-message streaming';
        messageDiv.innerHTML = `
            <div class="message-avatar">
                <i class="fas fa-robot"></i>
            </div>
            <div class="message-content">
                <div class="message-text"></div>
                <div class="streaming-cursor">▊</div>
            </div>
        `;
        
        this.messageContainer.appendChild(messageDiv);
        this.scrollToBottom();
        
        return messageDiv;
    }

    updateStreamingMessage(content) {
        if (!this.streamingMessage) return;
        
        const textElement = this.streamingMessage.querySelector('.message-text');
        textElement.innerHTML = this.formatResponse(content);
        
        // Update model indicator if metadata available
        if (this.currentStreamMetadata) {
            const modelIndicator = this.streamingMessage.querySelector('.model-indicator');
            if (!modelIndicator) {
                const indicator = document.createElement('div');
                indicator.className = 'model-indicator';
                indicator.innerHTML = `
                    <i class="fas fa-robot"></i>
                    <span>${this.currentStreamMetadata.provider || ''} ${this.currentStreamMetadata.model || ''}</span>
                `;
                this.streamingMessage.querySelector('.message-content').appendChild(indicator);
            }
        }
        
        this.scrollToBottom();
    }

    finalizeStreamingMessage() {
        if (!this.streamingMessage) return;
        
        this.streamingMessage.classList.remove('streaming');
        const cursor = this.streamingMessage.querySelector('.streaming-cursor');
        if (cursor) cursor.remove();
        
        this.streamingMessage = null;
        this.generateFollowUpSuggestions();
    }

    addMessage(type, content, sources = null, extraClass = '') {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}-message fade-in ${extraClass}`;
        
        const timestamp = new Date().toLocaleTimeString('ko-KR', { 
            hour: '2-digit', 
            minute: '2-digit' 
        });
        
        let messageHTML = '';
        
        if (type === 'user') {
            messageHTML = `
                <div class="message-content">
                    <div class="message-text">${this.escapeHtml(content)}</div>
                    <div class="message-time">${timestamp}</div>
                </div>
                <div class="message-avatar">
                    <i class="fas fa-user"></i>
                </div>
            `;
        } else if (type === 'assistant') {
            messageHTML = `
                <div class="message-avatar">
                    <i class="fas fa-robot"></i>
                </div>
                <div class="message-content">
                    <div class="message-text">${content}</div>
                    ${sources ? this.renderSources(sources) : ''}
                    <div class="message-actions">
                        <button onclick="enhancedChat.copyMessage(this)" title="복사">
                            <i class="fas fa-copy"></i>
                        </button>
                        <button onclick="enhancedChat.regenerateMessage(this)" title="다시 생성">
                            <i class="fas fa-redo"></i>
                        </button>
                        <button onclick="enhancedChat.rateMessage(this, 'up')" title="도움됨">
                            <i class="fas fa-thumbs-up"></i>
                        </button>
                        <button onclick="enhancedChat.rateMessage(this, 'down')" title="도움안됨">
                            <i class="fas fa-thumbs-down"></i>
                        </button>
                    </div>
                    <div class="message-time">${timestamp}</div>
                </div>
            `;
        } else if (type === 'error') {
            messageHTML = `
                <div class="message-content error">
                    <i class="fas fa-exclamation-circle"></i>
                    <div class="message-text">${content}</div>
                </div>
            `;
        }
        
        messageDiv.innerHTML = messageHTML;
        this.messageContainer.appendChild(messageDiv);
        this.scrollToBottom();
        
        // Add to chat history
        this.chatHistory.push({ type, content, timestamp: new Date() });
    }

    renderSources(sources) {
        if (!sources || sources.length === 0) return '';
        
        // 소스가 실제로 의미 있는 내용을 포함하는지 확인
        const hasValidSources = sources.some(source => 
            source.content && source.content.trim().length > 10
        );
        
        if (!hasValidSources) return '';
        
        // 고유 ID 생성
        const sourceId = `sources-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
        
        return `
            <div class="message-sources collapsed" id="${sourceId}">
                <div class="sources-header" onclick="enhancedChat.toggleSources('${sourceId}')">
                    <i class="fas fa-book-open"></i>
                    <span class="sources-title">참조 문서 ${sources.length}개</span>
                    <i class="fas fa-chevron-down toggle-icon"></i>
                </div>
                <div class="sources-content">
                    <div class="sources-compact">
                        ${sources.map((source, index) => `
                            <span class="source-badge" onclick="enhancedChat.showSourceDetail('${sourceId}', ${index})">
                                <i class="fas fa-file-alt"></i>
                                ${source.metadata.page ? `페이지 ${source.metadata.page}` : `참조 ${index + 1}`}
                            </span>
                        `).join('')}
                    </div>
                    <div class="sources-detail" style="display: none;">
                        ${sources.map((source, index) => `
                            <div class="source-detail-item" id="${sourceId}-detail-${index}" style="display: none;">
                                <div class="source-detail-header">
                                    <span class="source-detail-number">${index + 1}</span>
                                    ${source.metadata.page ? `<span class="source-detail-page">페이지 ${source.metadata.page}</span>` : ''}
                                    <div class="source-detail-actions">
                                        <button class="copy-source" onclick="enhancedChat.copySource('${sourceId}', ${index})" title="복사">
                                            <i class="fas fa-copy"></i>
                                        </button>
                                        <button class="close-detail" onclick="enhancedChat.hideSourceDetail('${sourceId}')" title="닫기">
                                            <i class="fas fa-times"></i>
                                        </button>
                                    </div>
                                </div>
                                <div class="source-detail-content" id="${sourceId}-content-${index}">
                                    ${source.content}
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
        `;
    }

    formatResponse(text) {
        // Convert markdown to HTML
        if (typeof marked !== 'undefined') {
            let formatted = marked.parse(text);
            
            // Add copy buttons to code blocks
            formatted = formatted.replace(/<pre><code/g, (match) => {
                return '<div class="code-block"><button class="copy-code-btn" onclick="enhancedChat.copyCode(this)"><i class="fas fa-copy"></i></button><pre><code';
            });
            formatted = formatted.replace(/<\/code><\/pre>/g, '</code></pre></div>');
            
            // 부연설명 섹션 스타일링
            formatted = formatted.replace(
                /<p><strong>부연설명:<\/strong>/g, 
                '<div class="supplementary-section"><p class="supplementary-header"><strong>부연설명:</strong></p><div class="supplementary-content">'
            );
            
            // 부연설명 섹션 닫기 (다음 섹션이나 끝까지)
            if (formatted.includes('supplementary-section')) {
                // 마지막에 div 닫기 태그 추가
                formatted = formatted.replace(/(<\/p>)$/, '$1</div></div>');
            }
            
            return formatted;
        }
        
        // Fallback formatting
        return text
            .replace(/\n\n/g, '</p><p>')
            .replace(/\n/g, '<br>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>');
    }

    enhanceQuestion(question) {
        // Add context hints based on question type
        const questionLower = question.toLowerCase();
        let enhanced = question;
        
        // Add implicit context
        if (questionLower.includes('요약')) {
            enhanced += ' (핵심 포인트와 주요 인사이트 포함)';
        } else if (questionLower.includes('분석')) {
            enhanced += ' (정량적/정성적 분석 포함)';
        } else if (questionLower.includes('비교')) {
            enhanced += ' (표 형식으로 정리)';
        }
        
        return enhanced;
    }

    showTypingIndicator() {
        const indicator = document.createElement('div');
        indicator.className = 'typing-indicator';
        indicator.id = 'typing-indicator';
        indicator.innerHTML = `
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        `;
        this.messageContainer.appendChild(indicator);
        this.scrollToBottom();
    }

    hideTypingIndicator() {
        const indicator = document.getElementById('typing-indicator');
        if (indicator) {
            indicator.remove();
        }
    }

    generateFollowUpSuggestions(answer = '') {
        const suggestions = this.suggestionEngine.generateSuggestions(answer, this.chatHistory);
        this.displaySuggestions(suggestions);
    }

    displaySuggestions(suggestions) {
        const container = document.createElement('div');
        container.className = 'suggestions-container';
        container.innerHTML = `
            <div class="suggestions-header">추천 질문:</div>
            <div class="suggestions-list">
                ${suggestions.map(s => `
                    <button class="suggestion-chip" onclick="enhancedChat.useSuggestion('${this.escapeHtml(s)}')">
                        ${this.escapeHtml(s)}
                    </button>
                `).join('')}
            </div>
        `;
        
        // Remove existing suggestions
        const existing = this.messageContainer.querySelector('.suggestions-container');
        if (existing) existing.remove();
        
        this.messageContainer.appendChild(container);
        this.scrollToBottom();
    }

    useSuggestion(suggestion) {
        this.inputElement.value = suggestion;
        this.autoResizeTextarea(this.inputElement);
        this.inputElement.focus();
    }

    copyMessage(button) {
        const messageText = button.closest('.message-content').querySelector('.message-text').innerText;
        navigator.clipboard.writeText(messageText).then(() => {
            this.showToast('메시지가 복사되었습니다');
        });
    }

    copyCode(button) {
        const codeBlock = button.nextElementSibling.querySelector('code').innerText;
        navigator.clipboard.writeText(codeBlock).then(() => {
            this.showToast('코드가 복사되었습니다');
        });
    }

    regenerateMessage(button) {
        const messageDiv = button.closest('.message');
        const previousUserMessage = this.getPreviousUserMessage(messageDiv);
        if (previousUserMessage) {
            this.sendMessage(previousUserMessage);
        }
    }

    rateMessage(button, rating) {
        // Send rating to backend
        console.log('Rating:', rating);
        button.classList.add('rated');
        this.showToast(rating === 'up' ? '피드백 감사합니다!' : '개선하겠습니다!');
    }

    autoResizeTextarea(textarea) {
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
    }

    scrollToBottom() {
        this.messageContainer.scrollTop = this.messageContainer.scrollHeight;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    truncateText(text, maxLength) {
        if (text.length <= maxLength) return text;
        return text.substring(0, maxLength) + '...';
    }

    showToast(message) {
        const toast = document.createElement('div');
        toast.className = 'toast-notification';
        toast.textContent = message;
        document.body.appendChild(toast);
        
        setTimeout(() => {
            toast.classList.add('show');
        }, 100);
        
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }

    getPreviousUserMessage(fromMessage) {
        const messages = Array.from(this.messageContainer.querySelectorAll('.message'));
        const currentIndex = messages.indexOf(fromMessage);
        
        for (let i = currentIndex - 1; i >= 0; i--) {
            if (messages[i].classList.contains('user-message')) {
                return messages[i].querySelector('.message-text').innerText;
            }
        }
        return null;
    }

    saveChatHistory() {
        localStorage.setItem(`chat_history_${window.currentSessionId}`, JSON.stringify(this.chatHistory));
    }

    loadChatHistory() {
        const saved = localStorage.getItem(`chat_history_${window.currentSessionId}`);
        if (saved) {
            this.chatHistory = JSON.parse(saved);
        }
    }

    startVoiceInput() {
        if (!('webkitSpeechRecognition' in window)) {
            this.showToast('음성 인식이 지원되지 않는 브라우저입니다');
            return;
        }

        const recognition = new webkitSpeechRecognition();
        recognition.lang = 'ko-KR';
        recognition.continuous = false;
        recognition.interimResults = false;

        recognition.onstart = () => {
            document.querySelector('.voice-input-btn').classList.add('recording');
        };

        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            this.inputElement.value = transcript;
            this.autoResizeTextarea(this.inputElement);
        };

        recognition.onerror = (event) => {
            console.error('Voice recognition error:', event.error);
            this.showToast('음성 인식 오류가 발생했습니다');
        };

        recognition.onend = () => {
            document.querySelector('.voice-input-btn').classList.remove('recording');
        };

        recognition.start();
    }

    toggleSources(sourceId) {
        const sourceElement = document.getElementById(sourceId);
        if (!sourceElement) return;
        
        sourceElement.classList.toggle('collapsed');
        const icon = sourceElement.querySelector('.toggle-icon');
        if (icon) {
            icon.classList.toggle('fa-chevron-down');
            icon.classList.toggle('fa-chevron-up');
        }
    }

    showSourceDetail(sourceId, index) {
        const sourceElement = document.getElementById(sourceId);
        if (!sourceElement) return;
        
        // Hide compact view and show detail view
        const compactView = sourceElement.querySelector('.sources-compact');
        const detailView = sourceElement.querySelector('.sources-detail');
        
        if (compactView) compactView.style.display = 'none';
        if (detailView) detailView.style.display = 'block';
        
        // Hide all detail items first
        const allDetails = sourceElement.querySelectorAll('.source-detail-item');
        allDetails.forEach(detail => detail.style.display = 'none');
        
        // Show selected detail
        const selectedDetail = document.getElementById(`${sourceId}-detail-${index}`);
        if (selectedDetail) {
            selectedDetail.style.display = 'block';
        }
    }

    hideSourceDetail(sourceId) {
        const sourceElement = document.getElementById(sourceId);
        if (!sourceElement) return;
        
        // Show compact view and hide detail view
        const compactView = sourceElement.querySelector('.sources-compact');
        const detailView = sourceElement.querySelector('.sources-detail');
        
        if (compactView) compactView.style.display = 'flex';
        if (detailView) detailView.style.display = 'none';
    }

    copySource(sourceId, index) {
        const contentElement = document.getElementById(`${sourceId}-content-${index}`);
        if (!contentElement) return;
        
        const text = contentElement.innerText;
        navigator.clipboard.writeText(text).then(() => {
            this.showToast('참조 내용이 복사되었습니다');
            
            // Change icon temporarily
            const copyBtn = contentElement.parentElement.querySelector('.copy-source i');
            if (copyBtn) {
                copyBtn.className = 'fas fa-check';
                setTimeout(() => {
                    copyBtn.className = 'fas fa-copy';
                }, 2000);
            }
        }).catch(() => {
            this.showToast('복사하는 중 오류가 발생했습니다');
        });
    }
}

// Suggestion Engine
class SuggestionEngine {
    constructor() {
        this.templates = {
            summary: ["이 문서의 핵심 내용을 요약해주세요", "주요 포인트 3가지를 알려주세요"],
            analysis: ["이 데이터의 트렌드를 분석해주세요", "가장 중요한 인사이트는 무엇인가요?"],
            comparison: ["이전 데이터와 비교해주세요", "다른 항목들과의 차이점을 설명해주세요"],
            detail: ["더 자세히 설명해주세요", "구체적인 예시를 들어주세요"],
            application: ["실제로 어떻게 활용할 수 있나요?", "다음 단계는 무엇인가요?"]
        };
    }

    generateSuggestions(lastAnswer, chatHistory) {
        const suggestions = [];
        
        // Context-based suggestions
        if (lastAnswer.includes('요약')) {
            suggestions.push(...this.templates.detail);
        } else if (lastAnswer.includes('분석')) {
            suggestions.push(...this.templates.application);
        } else {
            suggestions.push(...this.templates.summary);
        }
        
        // Add dynamic suggestions based on content
        if (lastAnswer.includes('데이터')) {
            suggestions.push("이 데이터의 이상치를 찾아주세요");
        }
        
        if (lastAnswer.includes('표')) {
            suggestions.push("표를 시각화해서 보여주세요");
        }
        
        return suggestions.slice(0, 3);
    }

    hasActiveSuggestion() {
        return document.querySelector('.suggestion-active') !== null;
    }
}

// Initialize enhanced chat when DOM is ready
let enhancedChat;
document.addEventListener('DOMContentLoaded', () => {
    if (document.getElementById('chat-messages')) {
        enhancedChat = new EnhancedChatUI();
    }
});

// Export for global access
window.EnhancedChatUI = EnhancedChatUI;
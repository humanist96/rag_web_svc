/**
 * Debug Mode for AI Nexus
 * Provides debugging tools and mock backend for testing
 */

// Debug configuration
const DEBUG_CONFIG = {
    enabled: localStorage.getItem('debug_mode') === 'true' || window.location.search.includes('debug=true'),
    mock_backend: localStorage.getItem('mock_backend') === 'true' || window.location.search.includes('mock=true'),
    verbose_logging: localStorage.getItem('verbose_logging') === 'true' || window.location.search.includes('verbose=true')
};

// Debug logger
const debugLog = {
    info: (message, data) => {
        if (DEBUG_CONFIG.enabled || DEBUG_CONFIG.verbose_logging) {
            console.log(`[DEBUG] ${message}`, data || '');
        }
    },
    error: (message, error) => {
        if (DEBUG_CONFIG.enabled || DEBUG_CONFIG.verbose_logging) {
            console.error(`[DEBUG ERROR] ${message}`, error || '');
        }
    },
    warn: (message, data) => {
        if (DEBUG_CONFIG.enabled || DEBUG_CONFIG.verbose_logging) {
            console.warn(`[DEBUG WARN] ${message}`, data || '');
        }
    }
};

// Mock backend for testing
const mockBackend = {
    upload: async (formData) => {
        debugLog.info('Mock backend: Simulating file upload');
        
        // Simulate upload delay
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        const file = formData.get('file');
        const embeddingModel = formData.get('embedding_model') || 'default';
        const llmModel = formData.get('llm_model') || 'default';
        
        // Simulate different responses based on file type
        const isCSV = file.name.toLowerCase().endsWith('.csv');
        const isPDF = file.name.toLowerCase().endsWith('.pdf');
        
        const mockResponse = {
            session_id: 'mock_session_' + Date.now(),
            filename: file.name,
            file_type: isCSV ? 'csv' : 'pdf',
            chunks: Math.floor(file.size / 1000) + 10,
            embedding_model: embeddingModel,
            llm_model: llmModel
        };
        
        if (isCSV) {
            mockResponse.rows = Math.floor(Math.random() * 1000) + 100;
            mockResponse.columns = Math.floor(Math.random() * 20) + 5;
        } else {
            mockResponse.pages = Math.floor(file.size / 50000) + 1;
            mockResponse.metadata = {
                title: `Mock Document: ${file.name.split('.')[0]}`
            };
            mockResponse.loader_used = 'PyPDF2';
        }
        
        debugLog.info('Mock backend: Upload successful', mockResponse);
        return mockResponse;
    },
    
    chat: async (message) => {
        debugLog.info('Mock backend: Simulating chat response for:', message);
        
        // Simulate chat delay
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        const responses = [
            "이것은 모크 응답입니다. 실제 백엔드가 연결되지 않은 상태입니다.",
            "디버그 모드에서 실행 중입니다. 파일 업로드와 기본적인 UI 기능을 테스트할 수 있습니다.",
            "백엔드 서비스가 준비되면 실제 AI 응답을 받을 수 있습니다.",
            `질문: "${message}"에 대한 실제 응답을 받으려면 백엔드 서버를 설정해주세요.`
        ];
        
        const randomResponse = responses[Math.floor(Math.random() * responses.length)];
        
        return {
            answer: randomResponse,
            is_off_topic: false,
            sources: []
        };
    }
};

// Override fetch for mock backend
if (DEBUG_CONFIG.mock_backend) {
    const originalFetch = window.fetch;
    window.fetch = async (url, options) => {
        if (url.includes('/upload')) {
            debugLog.info('Intercepting upload request for mock backend');
            try {
                const result = await mockBackend.upload(options.body);
                return {
                    ok: true,
                    status: 200,
                    json: async () => result
                };
            } catch (error) {
                debugLog.error('Mock upload failed:', error);
                return {
                    ok: false,
                    status: 500,
                    text: async () => 'Mock upload failed'
                };
            }
        } else if (url.includes('/chat')) {
            debugLog.info('Intercepting chat request for mock backend');
            try {
                const body = JSON.parse(options.body);
                const result = await mockBackend.chat(body.message);
                return {
                    ok: true,
                    status: 200,
                    json: async () => result
                };
            } catch (error) {
                debugLog.error('Mock chat failed:', error);
                return {
                    ok: false,
                    status: 500,
                    text: async () => 'Mock chat failed'
                };
            }
        }
        
        // For non-mocked requests, use original fetch
        return originalFetch(url, options);
    };
    
    debugLog.info('Mock backend enabled');
}

// Debug UI
if (DEBUG_CONFIG.enabled) {
    // Create debug panel
    const debugPanel = document.createElement('div');
    debugPanel.innerHTML = `
        <div style="
            position: fixed;
            top: 10px;
            right: 10px;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 15px;
            border-radius: 8px;
            font-family: monospace;
            font-size: 12px;
            z-index: 10000;
            max-width: 300px;
            backdrop-filter: blur(10px);
        ">
            <h3 style="margin: 0 0 10px; color: #ff6b35;">🐛 Debug Mode</h3>
            <div style="margin-bottom: 10px;">
                <label style="display: flex; align-items: center;">
                    <input type="checkbox" ${DEBUG_CONFIG.mock_backend ? 'checked' : ''} 
                           onchange="toggleMockBackend(this.checked)" style="margin-right: 8px;">
                    Mock Backend
                </label>
            </div>
            <div style="margin-bottom: 10px;">
                <label style="display: flex; align-items: center;">
                    <input type="checkbox" ${DEBUG_CONFIG.verbose_logging ? 'checked' : ''} 
                           onchange="toggleVerboseLogging(this.checked)" style="margin-right: 8px;">
                    Verbose Logging
                </label>
            </div>
            <div style="margin-bottom: 10px;">
                <button onclick="clearDebugData()" style="
                    background: #ff6b35;
                    color: white;
                    border: none;
                    padding: 5px 10px;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 11px;
                ">Clear Debug Data</button>
            </div>
            <div style="font-size: 10px; opacity: 0.7;">
                API: ${window.APP_CONFIG?.API_URL || 'Not configured'}<br>
                Mode: ${DEBUG_CONFIG.mock_backend ? 'Mock' : 'Live'}<br>
                Scripts: ${document.querySelectorAll('script').length} loaded
            </div>
        </div>
    `;
    
    document.body.appendChild(debugPanel);
    
    // Debug functions
    window.toggleMockBackend = (enabled) => {
        localStorage.setItem('mock_backend', enabled.toString());
        debugLog.info('Mock backend toggled:', enabled);
        location.reload();
    };
    
    window.toggleVerboseLogging = (enabled) => {
        localStorage.setItem('verbose_logging', enabled.toString());
        debugLog.info('Verbose logging toggled:', enabled);
    };
    
    window.clearDebugData = () => {
        localStorage.removeItem('debug_mode');
        localStorage.removeItem('mock_backend');
        localStorage.removeItem('verbose_logging');
        debugLog.info('Debug data cleared');
        location.reload();
    };
    
    debugLog.info('Debug panel initialized');
}

// Export debug utilities
window.DEBUG = {
    config: DEBUG_CONFIG,
    log: debugLog,
    mockBackend: mockBackend,
    enable: () => {
        localStorage.setItem('debug_mode', 'true');
        location.reload();
    },
    disable: () => {
        localStorage.removeItem('debug_mode');
        localStorage.removeItem('mock_backend');
        localStorage.removeItem('verbose_logging');
        location.reload();
    }
};

// Log initialization
debugLog.info('Debug mode initialized', {
    enabled: DEBUG_CONFIG.enabled,
    mock_backend: DEBUG_CONFIG.mock_backend,
    verbose_logging: DEBUG_CONFIG.verbose_logging,
    url: window.location.href
});

// Console helper
if (DEBUG_CONFIG.enabled) {
    console.log(`
%c🐛 AI Nexus Debug Mode
%cTo disable debug mode, run: DEBUG.disable()
To enable mock backend: localStorage.setItem('mock_backend', 'true'); location.reload()
To test upload: Drop a file or click upload area
    `, 
    'color: #ff6b35; font-size: 16px; font-weight: bold;',
    'color: #888; font-size: 12px;'
    );
}
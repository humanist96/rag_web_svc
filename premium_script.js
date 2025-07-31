// Premium Document Chatbot Script with 3D Background and Enhanced UX

// API Configuration - Load from config
const API_URL = window.APP_CONFIG ? window.APP_CONFIG.API_URL : 'http://localhost:8001';
let currentSessionId = null;
let currentPdfName = null;
let messageCount = 0;

console.log('API URL:', API_URL);

// Three.js 3D Background
let scene, camera, renderer, particles, particleSystem;

// Initialize 3D Background
function init3DBackground() {
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.z = 5;
    
    renderer = new THREE.WebGLRenderer({ 
        canvas: document.getElementById('three-canvas'),
        alpha: true,
        antialias: true
    });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    
    // Create particle geometry
    const particleCount = 2000;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);
    
    for (let i = 0; i < particleCount * 3; i += 3) {
        positions[i] = (Math.random() - 0.5) * 20;
        positions[i + 1] = (Math.random() - 0.5) * 20;
        positions[i + 2] = (Math.random() - 0.5) * 20;
        
        // Random colors in purple/blue spectrum
        colors[i] = 0.5 + Math.random() * 0.5; // R
        colors[i + 1] = 0.3 + Math.random() * 0.3; // G
        colors[i + 2] = 0.8 + Math.random() * 0.2; // B
    }
    
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    
    // Create particle material
    const material = new THREE.PointsMaterial({
        size: 0.05,
        vertexColors: true,
        blending: THREE.AdditiveBlending,
        transparent: true,
        opacity: 0.6
    });
    
    particleSystem = new THREE.Points(geometry, material);
    scene.add(particleSystem);
    
    // Create connecting lines
    const lineGeometry = new THREE.BufferGeometry();
    const linePositions = new Float32Array(200 * 3);
    
    for (let i = 0; i < 200 * 3; i += 3) {
        linePositions[i] = (Math.random() - 0.5) * 10;
        linePositions[i + 1] = (Math.random() - 0.5) * 10;
        linePositions[i + 2] = (Math.random() - 0.5) * 10;
    }
    
    lineGeometry.setAttribute('position', new THREE.BufferAttribute(linePositions, 3));
    
    const lineMaterial = new THREE.LineBasicMaterial({
        color: 0x6366f1,
        transparent: true,
        opacity: 0.1
    });
    
    const lines = new THREE.LineSegments(lineGeometry, lineMaterial);
    scene.add(lines);
    
    // Animation
    function animate() {
        requestAnimationFrame(animate);
        
        // Rotate particles
        particleSystem.rotation.x += 0.0005;
        particleSystem.rotation.y += 0.0008;
        
        // Float particles
        const positions = particleSystem.geometry.attributes.position.array;
        for (let i = 0; i < positions.length; i += 3) {
            positions[i + 1] += Math.sin(Date.now() * 0.001 + i) * 0.001;
        }
        particleSystem.geometry.attributes.position.needsUpdate = true;
        
        renderer.render(scene, camera);
    }
    
    animate();
    
    // Handle window resize
    window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    });
}

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    // Initialize 3D background
    init3DBackground();
    
    // Setup navigation
    setupNavigation();
    
    // Setup upload handlers
    setupUploadHandlers();
    
    // Setup chat handlers
    setupChatHandlers();
    
    // Initialize stats
    updateStats();
    
    // Load sessions
    refreshSessions();
    
    // Initialize analytics charts
    initializeAnalytics();
});

// Navigation
function setupNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    const sections = document.querySelectorAll('.content-section');
    
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = link.getAttribute('href').substring(1);
            
            // Update active nav
            navLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');
            
            // Show target section
            sections.forEach(section => {
                section.style.display = section.id === targetId ? 'block' : 'none';
            });
            
            // Special handling for chat section
            if (targetId === 'chat' && currentSessionId) {
                document.getElementById('chat').style.display = 'block';
            }
        });
    });
}

// Upload Handlers
function setupUploadHandlers() {
    const uploadZone = document.getElementById('upload-zone');
    const pdfInput = document.getElementById('pdf-input');
    
    // Click to upload
    uploadZone.addEventListener('click', () => {
        if (!uploadZone.classList.contains('uploading')) {
            pdfInput.click();
        }
    });
    
    // File input change
    pdfInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            const isValidFile = validateFile(file);
            if (isValidFile) {
                handleFileUpload(file);
            }
        }
    });
    
    // Drag and drop
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('drag-over');
    });
    
    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('drag-over');
    });
    
    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('drag-over');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            const file = files[0];
            const isValidFile = validateFile(file);
            if (isValidFile) {
                handleFileUpload(file);
            }
        }
    });
}

// File validation
function validateFile(file) {
    // Check if file exists
    if (!file) {
        showToast('파일이 선택되지 않았습니다.', 'error');
        return false;
    }
    
    // Check file extension
    const fileName = file.name.toLowerCase();
    const fileExtension = fileName.substring(fileName.lastIndexOf('.'));
    const allowedExtensions = ['.pdf', '.csv'];
    
    if (!allowedExtensions.includes(fileExtension)) {
        showToast(`지원하지 않는 파일 형식입니다. PDF 또는 CSV 파일만 업로드 가능합니다. (현재: ${fileExtension})`, 'error');
        return false;
    }
    
    // Check file size (50MB limit)
    const maxSize = 50 * 1024 * 1024; // 50MB
    if (file.size > maxSize) {
        const fileSizeMB = (file.size / 1024 / 1024).toFixed(2);
        showToast(`파일 크기가 너무 큽니다. 최대 50MB까지 업로드 가능합니다. (현재: ${fileSizeMB}MB)`, 'error');
        return false;
    }
    
    // Check if file is empty
    if (file.size === 0) {
        showToast('파일이 비어있습니다. 내용이 있는 파일을 업로드해주세요.', 'error');
        return false;
    }
    
    // Validate MIME type
    const validMimeTypes = {
        '.pdf': ['application/pdf', 'application/x-pdf'],
        '.csv': ['text/csv', 'application/csv', 'text/plain', 'application/vnd.ms-excel']
    };
    
    const expectedMimeTypes = validMimeTypes[fileExtension];
    if (expectedMimeTypes && file.type && !expectedMimeTypes.includes(file.type)) {
        console.warn(`MIME type mismatch: expected ${expectedMimeTypes.join(' or ')}, got ${file.type}`);
    }
    
    return true;
}

// File Upload Handler
async function handleFileUpload(file) {
    const uploadContent = document.getElementById('upload-content');
    const uploadProgress = document.getElementById('upload-progress');
    const uploadSuccess = document.getElementById('upload-success');
    const progressFill = document.getElementById('progress-fill');
    const progressPercentage = document.getElementById('progress-percentage');
    const progressFilename = document.getElementById('progress-filename');
    const progressStatus = document.getElementById('progress-status');
    
    // Show progress
    uploadContent.style.display = 'none';
    uploadProgress.style.display = 'block';
    progressFilename.textContent = file.name;
    progressStatus.textContent = 'Uploading...';
    
    // Create form data
    const formData = new FormData();
    formData.append('file', file);
    if (currentSessionId) {
        formData.append('session_id', currentSessionId);
    }
    
    // Simulate progress (since we can't track real upload progress with fetch)
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress += Math.random() * 15;
        if (progress > 90) {
            progress = 90;
            clearInterval(progressInterval);
        }
        progressFill.style.width = progress + '%';
        progressPercentage.textContent = Math.round(progress) + '%';
    }, 200);
    
    try {
        progressStatus.textContent = 'Processing file...';
        
        console.log(`Uploading to: ${API_URL}/upload`);
        console.log('File:', file.name, 'Size:', file.size);
        console.log('API_URL:', API_URL);
        console.log('window.APP_CONFIG:', window.APP_CONFIG);
        
        // FormData 내용 확인
        for (let [key, value] of formData.entries()) {
            console.log(`FormData - ${key}:`, value);
        }
        
        const response = await fetch(`${API_URL}/upload`, {
            method: 'POST',
            body: formData
        });
        
        console.log('Response status:', response.status);
        console.log('Response headers:', response.headers);
        
        if (!response.ok) {
            const errorText = await response.text();
            console.error('Upload error response:', errorText);
            let errorMessage = 'Upload failed';
            try {
                const errorData = JSON.parse(errorText);
                errorMessage = errorData.detail || errorMessage;
            } catch (e) {
                errorMessage = `Upload failed: ${response.status} ${response.statusText}`;
            }
            throw new Error(errorMessage);
        }
        
        const result = await response.json();
        
        // Complete progress
        clearInterval(progressInterval);
        progressFill.style.width = '100%';
        progressPercentage.textContent = '100%';
        
        // Update session info
        currentSessionId = result.session_id;
        currentPdfName = result.filename;
        
        // Show success
        setTimeout(() => {
            uploadProgress.style.display = 'none';
            uploadSuccess.style.display = 'block';
            // Display file info based on type
            let infoText = `${result.filename} • `;
            if (result.file_type === 'csv') {
                infoText += `${result.rows} rows • ${result.columns} columns • `;
            } else {
                infoText += `${result.pages} pages • `;
                // PDF 메타데이터 표시 (있는 경우)
                if (result.metadata && result.metadata.title) {
                    infoText += `"${result.metadata.title}" • `;
                }
            }
            infoText += `${result.chunks} chunks`;
            
            // 로더 정보 표시 (PDF인 경우)
            if (result.loader_used) {
                infoText += ` • ${result.loader_used}`;
            }
            
            document.getElementById('success-info').textContent = infoText;
            
            // Update stats
            updateStats();
            
            // Enable chat
            enableChat();
            
            showToast('File uploaded successfully!', 'success');
        }, 500);
        
    } catch (error) {
        console.error('Upload error:', error);
        clearInterval(progressInterval);
        
        // Reset upload
        uploadProgress.style.display = 'none';
        uploadContent.style.display = 'block';
        
        // Parse error response
        let errorMessage = '';
        
        // Check if it's our thrown error with message
        if (error.message) {
            errorMessage = error.message;
        } else if (error.name === 'TypeError' && error.message.includes('Failed to fetch')) {
            errorMessage = '서버에 연결할 수 없습니다. 네트워크 연결을 확인해주세요.';
        } else {
            errorMessage = '알 수 없는 오류가 발생했습니다. 다시 시도해주세요.';
        }
        
        // Show detailed error message with retry suggestion
        showToast(errorMessage, 'error');
        
        // Log detailed error for debugging
        console.error('Upload failed:', {
            fileName: file.name,
            fileSize: file.size,
            fileType: file.type,
            error: error
        });
    }
}

// Reset Upload
function resetUpload() {
    document.getElementById('upload-content').style.display = 'block';
    document.getElementById('upload-progress').style.display = 'none';
    document.getElementById('upload-success').style.display = 'none';
    document.getElementById('pdf-input').value = '';
    
    // Reset progress
    document.getElementById('progress-fill').style.width = '0%';
    document.getElementById('progress-percentage').textContent = '0%';
}

// Start Chat
function startChat() {
    // Switch to chat section
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
    });
    document.querySelector('a[href="#chat"]').classList.add('active');
    
    document.querySelectorAll('.content-section').forEach(section => {
        section.style.display = 'none';
    });
    document.getElementById('chat').style.display = 'block';
    
    // Clear previous messages
    clearChat();
    
    // Add welcome message
    addMessage(`Great! I've loaded "${currentPdfName}". What would you like to know about this document?`, false);
}

// Chat Handlers
function setupChatHandlers() {
    const chatInput = document.getElementById('chat-input');
    const sendButton = document.getElementById('send-button');
    
    // Auto-resize textarea
    chatInput.addEventListener('input', () => {
        chatInput.style.height = 'auto';
        chatInput.style.height = chatInput.scrollHeight + 'px';
    });
    
    // Send on Enter
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // Send button click
    sendButton.addEventListener('click', sendMessage);
}

// Enable/Disable Chat
function enableChat() {
    const chatInput = document.getElementById('chat-input');
    const sendButton = document.getElementById('send-button');
    const chatFilename = document.getElementById('chat-filename');
    
    chatInput.disabled = false;
    sendButton.disabled = false;
    chatInput.placeholder = 'Type your question...';
    chatFilename.textContent = currentPdfName || 'No document loaded';
}

function disableChat() {
    const chatInput = document.getElementById('chat-input');
    const sendButton = document.getElementById('send-button');
    
    chatInput.disabled = true;
    sendButton.disabled = true;
    chatInput.placeholder = 'Please upload a file first...';
}

// Send Message
async function sendMessage() {
    const chatInput = document.getElementById('chat-input');
    const message = chatInput.value.trim();
    
    if (!message || !currentSessionId) return;
    
    // Add user message
    addMessage(message, true);
    
    // Clear input
    chatInput.value = '';
    chatInput.style.height = 'auto';
    
    // Show typing indicator
    showTypingIndicator();
    
    try {
        const response = await fetch(`${API_URL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                session_id: currentSessionId,
                message: message
            })
        });
        
        if (!response.ok) {
            throw new Error('Failed to get response');
        }
        
        const result = await response.json();
        
        // Remove typing indicator
        removeTypingIndicator();
        
        // Add bot response
        addMessage(result.answer, false, result.sources);
        
        // Update stats
        messageCount++;
        updateStats();
        
        // Refresh analytics if analytics tab is active
        const analyticsSection = document.getElementById('analytics');
        if (analyticsSection && analyticsSection.style.display !== 'none') {
            setTimeout(async () => {
                const analyticsData = await fetchAnalyticsData();
                updateUsageChart(analyticsData);
                updateQueryChart(analyticsData);
                updateResponseChart(analyticsData);
                createKeywordsCloud(analyticsData);
            }, 1000);
        }
        
    } catch (error) {
        console.error('Chat error:', error);
        removeTypingIndicator();
        addMessage('Sorry, I encountered an error. Please try again.', false);
        showToast('Failed to get response', 'error');
    }
}

// Add Message to Chat
function addMessage(content, isUser = false, sources = []) {
    const chatMessages = document.getElementById('chat-messages');
    
    // Remove welcome message if exists
    const welcomeMessage = chatMessages.querySelector('.chat-welcome');
    if (welcomeMessage) {
        welcomeMessage.remove();
    }
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    messageContent.textContent = content;
    messageDiv.appendChild(messageContent);
    
    // Add timestamp
    const messageTime = document.createElement('div');
    messageTime.className = 'message-time';
    messageTime.textContent = new Date().toLocaleTimeString('ko-KR', { 
        hour: '2-digit', 
        minute: '2-digit' 
    });
    messageDiv.appendChild(messageTime);
    
    // Add sources if available
    if (!isUser && sources && sources.length > 0) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'message-sources';
        sourcesDiv.style.marginTop = '8px';
        sourcesDiv.style.paddingTop = '8px';
        sourcesDiv.style.borderTop = '1px solid rgba(255,255,255,0.1)';
        
        const sourcesTitle = document.createElement('div');
        sourcesTitle.style.fontSize = '0.75rem';
        sourcesTitle.style.opacity = '0.7';
        sourcesTitle.innerHTML = '<i class="fas fa-book"></i> References:';
        sourcesDiv.appendChild(sourcesTitle);
        
        sources.forEach((source, index) => {
            const sourceItem = document.createElement('div');
            sourceItem.style.fontSize = '0.75rem';
            sourceItem.style.opacity = '0.6';
            sourceItem.style.marginTop = '4px';
            
            const pageInfo = source.metadata?.page ? `Page ${source.metadata.page}` : '';
            sourceItem.textContent = `${index + 1}. ${pageInfo} - ${source.content.substring(0, 100)}...`;
            sourcesDiv.appendChild(sourceItem);
        });
        
        messageDiv.appendChild(sourcesDiv);
    }
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Typing Indicator
function showTypingIndicator() {
    const chatMessages = document.getElementById('chat-messages');
    
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message bot typing-indicator-message';
    typingDiv.innerHTML = `
        <div class="typing-indicator">
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
        </div>
    `;
    
    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function removeTypingIndicator() {
    const typingIndicator = document.querySelector('.typing-indicator-message');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

// Clear Chat
function clearChat() {
    const chatMessages = document.getElementById('chat-messages');
    chatMessages.innerHTML = `
        <div class="chat-welcome">
            <i class="fas fa-robot chat-welcome-icon"></i>
            <h3>Welcome to AI Nexus</h3>
            <p>Upload a PDF or CSV file to start asking questions</p>
        </div>
    `;
}

// Export Chat
function exportChat() {
    // TODO: Implement chat export functionality
    showToast('Export feature coming soon!', 'warning');
}

// Sessions Management
async function refreshSessions() {
    try {
        const response = await fetch(`${API_URL}/sessions`);
        const data = await response.json();
        
        const sessionsGrid = document.getElementById('sessions-grid');
        
        if (data.sessions.length === 0) {
            sessionsGrid.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-folder-open" style="font-size: 3rem; color: var(--text-tertiary); margin-bottom: 1rem;"></i>
                    <p style="color: var(--text-secondary);">No active sessions</p>
                </div>
            `;
        } else {
            sessionsGrid.innerHTML = data.sessions.map(session => `
                <div class="session-card glass" onclick="loadSession('${session.session_id}')">
                    <div class="session-header">
                        <div>
                            <h3 class="session-title">${session.pdf_name || 'Untitled Session'}</h3>
                            <p class="session-date">${new Date(session.created_at).toLocaleDateString()}</p>
                        </div>
                        <span class="session-status ${session.pdf_name ? 'active' : ''}">
                            ${session.pdf_name ? 'Active' : 'Empty'}
                        </span>
                    </div>
                    <div class="session-stats">
                        <div class="session-stat">
                            <i class="fas fa-comments"></i>
                            <span>${session.messages_count} messages</span>
                        </div>
                        <div class="session-stat">
                            <i class="fas fa-clock"></i>
                            <span>${new Date(session.created_at).toLocaleTimeString()}</span>
                        </div>
                    </div>
                </div>
            `).join('');
        }
    } catch (error) {
        console.error('Failed to load sessions:', error);
        showToast('Failed to load sessions', 'error');
    }
}

// Load Session
async function loadSession(sessionId) {
    try {
        const response = await fetch(`${API_URL}/sessions/${sessionId}`);
        if (!response.ok) {
            throw new Error('Session not found');
        }
        
        const session = await response.json();
        
        // Update current session
        currentSessionId = session.session_id;
        currentPdfName = session.pdf_name;
        
        if (currentPdfName) {
            enableChat();
            showToast(`Loaded session with "${currentPdfName}"`, 'success');
            
            // Switch to chat
            startChat();
        } else {
            disableChat();
            showToast('This session has no file loaded', 'warning');
        }
        
    } catch (error) {
        console.error('Failed to load session:', error);
        showToast('Failed to load session', 'error');
    }
}

// Update Stats
async function updateStats() {
    try {
        const analyticsData = await fetchAnalyticsData();
        
        document.getElementById('total-pdfs').textContent = analyticsData.total_uploads || 0;
        document.getElementById('total-queries').textContent = analyticsData.total_queries || 0;
        
        // Update average response time
        if (analyticsData.average_response_time) {
            document.getElementById('response-time').textContent = 
                analyticsData.average_response_time.toFixed(1) + 's';
        }
    } catch (error) {
        console.error('Failed to update stats:', error);
    }
}

// Toast Notifications
function showToast(message, type = 'info') {
    const toastContainer = document.getElementById('toast-container');
    
    const toast = document.createElement('div');
    toast.className = `toast glass ${type}`;
    
    const icon = {
        success: 'fas fa-check-circle',
        error: 'fas fa-exclamation-circle',
        warning: 'fas fa-exclamation-triangle',
        info: 'fas fa-info-circle'
    }[type];
    
    toast.innerHTML = `
        <i class="${icon}"></i>
        <span>${message}</span>
    `;
    
    toastContainer.appendChild(toast);
    
    // Auto remove after duration (longer for errors)
    const duration = type === 'error' ? 6000 : 3000;
    setTimeout(() => {
        toast.style.animation = 'toastSlide 0.3s ease reverse';
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

// Quick Actions
function toggleQuickActions() {
    const quickActions = document.getElementById('quick-actions');
    quickActions.classList.toggle('show');
}

function quickUpload() {
    toggleQuickActions();
    document.getElementById('pdf-input').click();
}

function newSession() {
    toggleQuickActions();
    currentSessionId = null;
    currentPdfName = null;
    resetUpload();
    clearChat();
    disableChat();
    showToast('New session created', 'success');
}

function showHelp() {
    toggleQuickActions();
    showToast('Help documentation coming soon!', 'info');
}

// Voice Input (placeholder)
function toggleVoiceInput() {
    showToast('Voice input coming soon!', 'info');
}

// Attach File (placeholder)
function attachFile() {
    showToast('File attachment coming soon!', 'info');
}

// Close quick actions when clicking outside
document.addEventListener('click', (e) => {
    const fab = document.querySelector('.fab');
    const quickActions = document.getElementById('quick-actions');
    
    if (!fab.contains(e.target) && !quickActions.contains(e.target)) {
        quickActions.classList.remove('show');
    }
});

// Analytics Charts
let usageChart, queryChart, responseChart;

function initializeAnalytics() {
    // Initialize charts when analytics section is first shown
    const analyticsLink = document.querySelector('a[href="#analytics"]');
    analyticsLink.addEventListener('click', () => {
        setTimeout(async () => {
            // Fetch real analytics data
            const analyticsData = await fetchAnalyticsData();
            
            if (!usageChart) {
                createUsageChart(analyticsData);
            } else {
                updateUsageChart(analyticsData);
            }
            
            if (!queryChart) {
                createQueryChart(analyticsData);
            } else {
                updateQueryChart(analyticsData);
            }
            
            if (!responseChart) {
                createResponseChart(analyticsData);
            } else {
                updateResponseChart(analyticsData);
            }
            
            createKeywordsCloud(analyticsData);
        }, 100);
    });
}

async function fetchAnalyticsData() {
    try {
        const response = await fetch(`${API_URL}/analytics`);
        if (!response.ok) {
            throw new Error('Failed to fetch analytics');
        }
        const data = await response.json();
        
        // Also save to localStorage for persistence
        localStorage.setItem('analyticsData', JSON.stringify(data));
        localStorage.setItem('analyticsTimestamp', new Date().toISOString());
        
        return data;
    } catch (error) {
        console.error('Analytics fetch error:', error);
        // Fallback to localStorage data if available
        const savedData = localStorage.getItem('analyticsData');
        if (savedData) {
            return JSON.parse(savedData);
        }
        // Return default empty data
        return {
            total_uploads: 0,
            total_queries: 0,
            query_types: {},
            response_times: [],
            keywords: {},
            daily_usage: {},
            average_response_time: 0
        };
    }
}

function createUsageChart(analyticsData) {
    const ctx = document.getElementById('usage-chart');
    if (!ctx) return;
    
    // Process daily usage data
    const sortedDates = Object.keys(analyticsData.daily_usage).sort();
    const labels = [];
    const data = [];
    
    // Get last 7 days
    for (let i = 6; i >= 0; i--) {
        const date = new Date();
        date.setDate(date.getDate() - i);
        const dateStr = date.toISOString().split('T')[0];
        const dateLabel = date.toLocaleDateString('ko-KR', { month: 'short', day: 'numeric' });
        
        labels.push(dateLabel);
        data.push(analyticsData.daily_usage[dateStr] || 0);
    }
    
    usageChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Queries per Day',
                data: data,
                borderColor: '#6366f1',
                backgroundColor: 'rgba(99, 102, 241, 0.1)',
                tension: 0.4,
                fill: true,
                pointBackgroundColor: '#6366f1',
                pointBorderColor: '#fff',
                pointBorderWidth: 2,
                pointRadius: 5,
                pointHoverRadius: 7
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#fff',
                    bodyColor: '#fff',
                    borderColor: '#6366f1',
                    borderWidth: 1,
                    cornerRadius: 8,
                    padding: 12
                }
            },
            scales: {
                x: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#a3a3a3'
                    }
                },
                y: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#a3a3a3',
                        stepSize: 10
                    },
                    beginAtZero: true
                }
            }
        }
    });
}

function createQueryChart(analyticsData) {
    const ctx = document.getElementById('query-chart');
    if (!ctx) return;
    
    // Process query types data
    const queryTypes = analyticsData.query_types || {};
    const labels = Object.keys(queryTypes);
    const data = Object.values(queryTypes);
    
    // If no data, show empty state
    if (labels.length === 0) {
        labels.push('No queries yet');
        data.push(1);
    }
    
    queryChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: [
                    '#6366f1',
                    '#a855f7',
                    '#ec4899',
                    '#f59e0b',
                    '#10b981'
                ],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: '#a3a3a3',
                        padding: 20,
                        font: {
                            size: 12
                        }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#fff',
                    bodyColor: '#fff',
                    borderColor: '#6366f1',
                    borderWidth: 1,
                    cornerRadius: 8,
                    padding: 12,
                    callbacks: {
                        label: function(context) {
                            return context.label + ': ' + context.parsed + '%';
                        }
                    }
                }
            }
        }
    });
}

function createResponseChart(analyticsData) {
    const ctx = document.getElementById('response-chart');
    if (!ctx) return;
    
    // Process response time data
    const responseTimes = analyticsData.response_times || [];
    const buckets = {
        '< 1s': 0,
        '1-2s': 0,
        '2-3s': 0,
        '3-5s': 0,
        '> 5s': 0
    };
    
    // Categorize response times
    responseTimes.forEach(time => {
        if (time < 1) buckets['< 1s']++;
        else if (time < 2) buckets['1-2s']++;
        else if (time < 3) buckets['2-3s']++;
        else if (time < 5) buckets['3-5s']++;
        else buckets['> 5s']++;
    });
    
    const total = responseTimes.length || 1;
    const labels = Object.keys(buckets);
    const data = Object.values(buckets).map(count => (count / total) * 100);
    
    responseChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Response Distribution (%)',
                data: data,
                backgroundColor: 'rgba(99, 102, 241, 0.8)',
                borderColor: '#6366f1',
                borderWidth: 1,
                borderRadius: 8,
                barThickness: 40
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#fff',
                    bodyColor: '#fff',
                    borderColor: '#6366f1',
                    borderWidth: 1,
                    cornerRadius: 8,
                    padding: 12,
                    callbacks: {
                        label: function(context) {
                            return context.parsed.y + '% of queries';
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        color: '#a3a3a3'
                    }
                },
                y: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#a3a3a3',
                        stepSize: 10,
                        callback: function(value) {
                            return value + '%';
                        }
                    },
                    beginAtZero: true,
                    max: 50
                }
            }
        }
    });
}

function createKeywordsCloud(analyticsData) {
    const container = document.getElementById('keywords-cloud');
    if (!container) return;
    
    // Clear existing content
    container.innerHTML = '';
    
    // Process keywords data
    const keywordsData = analyticsData.keywords || {};
    const keywords = Object.entries(keywordsData)
        .map(([text, count]) => ({ text, weight: count }))
        .sort((a, b) => b.weight - a.weight)
        .slice(0, 20); // Top 20 keywords
    
    // If no keywords, show message
    if (keywords.length === 0) {
        container.innerHTML = '<p style="color: var(--text-secondary); text-align: center;">No keywords analyzed yet</p>';
        return;
    }
    
    // Normalize weights
    const maxWeight = Math.max(...keywords.map(k => k.weight));
    keywords.forEach(keyword => {
        keyword.weight = (keyword.weight / maxWeight) * 10; // Scale to 0-10
    });
    
    // Create keyword elements
    keywords.forEach(keyword => {
        const span = document.createElement('span');
        span.className = 'keyword-tag';
        span.textContent = keyword.text;
        
        // Style based on weight
        const size = 0.8 + (keyword.weight / 10) * 0.8;
        span.style.fontSize = `${size}rem`;
        span.style.opacity = 0.5 + (keyword.weight / 10) * 0.5;
        span.style.padding = `${size * 0.3}rem ${size * 0.6}rem`;
        span.style.margin = '0.25rem';
        span.style.display = 'inline-block';
        span.style.background = 'var(--surface)';
        span.style.border = '1px solid rgba(255, 255, 255, 0.1)';
        span.style.borderRadius = 'var(--radius-full)';
        span.style.color = keyword.weight > 7 ? 'var(--primary-light)' : 'var(--text-secondary)';
        span.style.transition = 'all var(--transition-base)';
        span.style.cursor = 'pointer';
        
        // Hover effect
        span.addEventListener('mouseenter', () => {
            span.style.transform = 'translateY(-2px)';
            span.style.background = 'var(--surface-hover)';
            span.style.color = 'var(--primary)';
        });
        
        span.addEventListener('mouseleave', () => {
            span.style.transform = 'translateY(0)';
            span.style.background = 'var(--surface)';
            span.style.color = keyword.weight > 7 ? 'var(--primary-light)' : 'var(--text-secondary)';
        });
        
        container.appendChild(span);
    });
}

// Update chart functions for real-time updates
function updateUsageChart(analyticsData) {
    if (!usageChart) return;
    
    const labels = [];
    const data = [];
    
    // Get last 7 days
    for (let i = 6; i >= 0; i--) {
        const date = new Date();
        date.setDate(date.getDate() - i);
        const dateStr = date.toISOString().split('T')[0];
        const dateLabel = date.toLocaleDateString('ko-KR', { month: 'short', day: 'numeric' });
        
        labels.push(dateLabel);
        data.push(analyticsData.daily_usage[dateStr] || 0);
    }
    
    usageChart.data.labels = labels;
    usageChart.data.datasets[0].data = data;
    usageChart.update();
}

function updateQueryChart(analyticsData) {
    if (!queryChart) return;
    
    const queryTypes = analyticsData.query_types || {};
    const labels = Object.keys(queryTypes);
    const data = Object.values(queryTypes);
    
    if (labels.length === 0) {
        labels.push('No queries yet');
        data.push(1);
    }
    
    queryChart.data.labels = labels;
    queryChart.data.datasets[0].data = data;
    queryChart.update();
}

function updateResponseChart(analyticsData) {
    if (!responseChart) return;
    
    const responseTimes = analyticsData.response_times || [];
    const buckets = {
        '< 1s': 0,
        '1-2s': 0,
        '2-3s': 0,
        '3-5s': 0,
        '> 5s': 0
    };
    
    responseTimes.forEach(time => {
        if (time < 1) buckets['< 1s']++;
        else if (time < 2) buckets['1-2s']++;
        else if (time < 3) buckets['2-3s']++;
        else if (time < 5) buckets['3-5s']++;
        else buckets['> 5s']++;
    });
    
    const total = responseTimes.length || 1;
    const data = Object.values(buckets).map(count => (count / total) * 100);
    
    responseChart.data.datasets[0].data = data;
    responseChart.update();
}

// Make functions global
window.startChat = startChat;
window.resetUpload = resetUpload;
window.clearChat = clearChat;
window.exportChat = exportChat;
window.sendMessage = sendMessage;
window.refreshSessions = refreshSessions;
window.loadSession = loadSession;
window.toggleQuickActions = toggleQuickActions;
window.quickUpload = quickUpload;
window.newSession = newSession;
window.showHelp = showHelp;
window.toggleVoiceInput = toggleVoiceInput;
window.attachFile = attachFile;
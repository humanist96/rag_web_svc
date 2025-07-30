// Configuration for different environments
const config = {
    development: {
        API_URL: 'http://localhost:8001'
    },
    production: {
        API_URL: 'https://rag-web-svc.onrender.com' // Backend still on Render
    }
};

// Detect environment - Vercel sets NODE_ENV
const isProduction = window.location.hostname !== 'localhost' && 
                    window.location.hostname !== '127.0.0.1';

// Export the appropriate configuration
window.APP_CONFIG = isProduction ? config.production : config.development;
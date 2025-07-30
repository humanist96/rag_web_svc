// Configuration for different environments
const config = {
    development: {
        API_URL: 'http://localhost:8001'
    },
    production: {
        API_URL: 'https://ai-nexus-backend.onrender.com' // Will be updated after deployment
    }
};

// Detect environment
const isProduction = window.location.hostname !== 'localhost' && 
                    window.location.hostname !== '127.0.0.1';

// Export the appropriate configuration
window.APP_CONFIG = isProduction ? config.production : config.development;
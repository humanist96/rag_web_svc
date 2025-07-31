/**
 * Simple Upload Fix
 * This overrides the model checking and allows direct upload
 */

// Override the handleFileUpload to bypass model selection for now
window.handleFileUploadOriginal = window.handleFileUpload;

window.handleFileUpload = async function(file) {
    console.log('Using simplified upload handler');
    
    // Create a default model selection
    const defaultModelSelection = {
        embedding: {
            key: 'default',
            model: {
                name: 'Default Embedding',
                description: 'Standard embedding model'
            }
        },
        llm: {
            key: 'claude-3-sonnet-20240229',
            model: {
                name: 'Claude 3 Sonnet',
                description: 'Anthropic Claude 3 Sonnet'
            }
        },
        fileInfo: {
            name: file.name,
            size: file.size,
            type: file.type
        }
    };
    
    // Proceed directly with upload using default models
    await proceedWithUpload(file, defaultModelSelection);
};

// Also disable the model selector from appearing automatically
if (window.embeddingSelector) {
    const originalShow = window.embeddingSelector.show;
    window.embeddingSelector.show = function(fileInfo, callback) {
        console.log('Model selector bypassed - using defaults');
        // Immediately call the callback with default selection
        const defaultSelection = {
            embedding: {
                key: 'default',
                model: {
                    name: 'Default Embedding',
                    description: 'Standard embedding model'
                }
            },
            llm: {
                key: 'claude-3-sonnet-20240229',
                model: {
                    name: 'Claude 3 Sonnet',
                    description: 'Anthropic Claude 3 Sonnet'
                }
            },
            fileInfo: fileInfo
        };
        
        if (callback) {
            callback(defaultSelection);
        }
    };
}

// Make sure the API URL is set correctly
if (!window.API_URL && window.APP_CONFIG) {
    window.API_URL = window.APP_CONFIG.API_URL;
}

console.log('Simple upload fix applied - uploads will use Claude 3 Sonnet by default');
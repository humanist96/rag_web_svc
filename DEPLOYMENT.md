# AI Nexus Deployment Guide

This guide covers the deployment of the AI Nexus document analysis platform with enhanced embedding model selection features.

## Features Implemented

### 1. Embedding Model Selection UI
- **Modal Interface**: Professional modal with model selection before file upload
- **Model Information**: Detailed specs, performance indicators, and cost estimates
- **Compatibility Matrix**: Automatic filtering of compatible LLM models
- **Recommendations**: Smart model recommendations based on file characteristics

### 2. Enhanced Model Integration
- **Multiple Providers**: OpenAI, Anthropic, HuggingFace, Cohere support
- **Dynamic Configuration**: Runtime model selection and configuration
- **Cost Estimation**: Real-time cost calculation for selected models
- **Performance Metrics**: Visual performance indicators for each model

### 3. Improved Upload Flow
- **Pre-Upload Validation**: Model selection required before file processing
- **Enhanced Progress**: Detailed progress with model information
- **Session Management**: Model configuration stored per session
- **Error Handling**: Comprehensive error handling and user feedback

## File Structure

```
├── index.html                          # Main HTML file
├── embedding_model_selector.css        # Model selection UI styles
├── embedding_model_selector.js         # Model selection logic
├── enhanced_upload_handler.py          # Backend upload handler
├── embedding_models.py                 # Embedding model providers
├── llm_provider.py                     # LLM provider management
├── model_specific_prompts.py           # Existing prompt optimization
├── streaming_handler.py                # Existing streaming handler
├── premium_script.js                   # Updated main script
├── enhanced_chat_ui.js                 # Existing chat UI
├── enhanced_chat_styles.css            # Existing chat styles
└── vercel.json                         # Updated deployment config
```

## Deployment Steps

### 1. Frontend Deployment (Vercel)

The frontend is configured for static deployment on Vercel:

```bash
# Install Vercel CLI (if not already installed)
npm install -g vercel

# Deploy to Vercel
vercel --prod
```

**Vercel Configuration** (vercel.json):
- Static hosting for all frontend assets
- Security headers included
- Optimized for fast global delivery

### 2. Backend Requirements

For full functionality, you'll need a backend server with:

#### Required Python Packages:
```bash
pip install fastapi uvicorn python-multipart
pip install openai anthropic cohere sentence-transformers
pip install langchain pypdf2 pandas
```

#### Environment Variables:
```bash
# API Keys (configure in your hosting platform)
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
COHERE_API_KEY=your_cohere_key

# Optional: Custom API endpoint
API_URL=https://your-backend-domain.com
```

### 3. Backend Deployment Options

#### Option A: Vercel Serverless Functions
```bash
# Create api/ directory and add serverless functions
mkdir api
# Move Python files to api/ directory
# Configure vercel.json for serverless functions
```

#### Option B: Separate Backend Service
- Deploy Python backend to services like Railway, Render, or AWS
- Update `config.js` with backend URL
- Ensure CORS is properly configured

### 4. Production Checklist

#### Frontend:
- [x] Model selection UI implemented
- [x] Responsive design for mobile devices
- [x] Error handling and user feedback
- [x] Performance optimizations
- [x] Security headers configured

#### Backend:
- [ ] API endpoints for model management
- [ ] File upload with model selection
- [ ] Session management
- [ ] Error handling and logging
- [ ] Rate limiting and security

#### Configuration:
- [ ] Environment variables set
- [ ] API keys configured
- [ ] CORS policies configured
- [ ] Database/storage configured (if needed)

## Model Support

### Embedding Models:
1. **OpenAI Ada-002** - High performance, 1536 dimensions
2. **OpenAI Embedding v3 Small** - Efficient, cost-effective
3. **OpenAI Embedding v3 Large** - Maximum accuracy, 3072 dimensions
4. **HuggingFace Sentence Transformers** - Free, open-source
5. **Cohere Multilingual** - 100+ languages support

### LLM Models:
1. **GPT-3.5 Turbo** - Fast, efficient
2. **GPT-3.5 Turbo 16K** - Extended context
3. **GPT-4** - Premium reasoning
4. **GPT-4 Turbo** - Latest with 128K context
5. **Claude 3 Opus** - Powerful analysis
6. **Claude 3 Sonnet** - Balanced performance
7. **Claude 3 Haiku** - Fast responses

## Usage Flow

1. **File Selection**: User selects PDF or CSV file
2. **Model Selection**: Modal appears with embedding and LLM options
3. **Recommendations**: System suggests optimal models based on file
4. **Compatibility Check**: Only compatible LLM models are available
5. **Cost Estimation**: Real-time cost calculation displayed
6. **Upload**: File processed with selected models
7. **Chat Interface**: Enhanced with model information

## Performance Considerations

### Frontend:
- Lazy loading of model selection modal
- Optimized CSS and JavaScript
- Responsive images and icons
- Minimal bundle size

### Backend:
- Async processing for file uploads
- Batch embedding creation
- Model instance caching
- Error recovery mechanisms

## Security Features

### Frontend:
- Input validation and sanitization
- XSS protection headers
- Content Security Policy
- Secure HTTPS communication

### Backend:
- API key encryption
- File type validation
- Size limits enforcement
- Session isolation

## Monitoring and Analytics

### Frontend Metrics:
- Model selection preferences
- Upload success rates
- User interaction patterns
- Performance timing

### Backend Metrics:
- API response times
- Model performance
- Error rates
- Cost tracking

## Support and Maintenance

### Regular Updates:
- Model availability checks
- Performance optimization
- Security patches
- Feature enhancements

### Monitoring:
- Uptime monitoring
- Error logging
- Performance metrics
- User feedback collection

## Troubleshooting

### Common Issues:
1. **Model Not Available**: Check API keys and model availability
2. **Upload Fails**: Verify file format and size limits
3. **Slow Performance**: Check network and server response times
4. **UI Issues**: Clear browser cache and check console errors

### Debug Mode:
Enable detailed logging by setting `DEBUG=true` in environment variables.

## Cost Optimization

### Model Selection Tips:
- Use smaller models for simple documents
- Choose multilingual models only when needed
- Monitor usage and adjust model selection
- Implement caching for repeated queries

### Deployment Costs:
- Frontend: Free tier available on Vercel
- Backend: Variable based on usage and provider
- Storage: Minimal for session data
- Models: Pay-per-use based on tokens processed

---

For technical support or feature requests, please refer to the project documentation or contact the development team.
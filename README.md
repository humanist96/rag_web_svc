# AI Nexus - Premium PDF Analysis Platform

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.116+-green.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/LangChain-0.3.18+-orange.svg" alt="LangChain">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</div>

## 🚀 Overview

AI Nexus is a premium PDF analysis platform that transforms your documents into interactive knowledge bases using advanced AI technology. Built with FastAPI and LangChain, it provides real-time chat capabilities, comprehensive analytics, and a modern, responsive UI.

### ✨ Key Features

- **🤖 AI-Powered Analysis**: Leverages OpenAI GPT models for intelligent document understanding
- **💬 Real-time Chat**: Interactive Q&A with your PDFs using advanced RAG (Retrieval-Augmented Generation)
- **📊 Analytics Dashboard**: Track usage metrics, query patterns, and response times
- **🎨 Modern UI**: Beautiful, responsive design with 3D animations and glass morphism effects
- **📈 Real-time Statistics**: Monitor uploads, queries, and performance metrics
- **🔒 Secure Processing**: Safe document handling with privacy-focused design
- **🌐 Multi-language Support**: Handles documents in multiple languages including Korean and English

## 🛠️ Tech Stack

### Backend
- **FastAPI**: High-performance Python web framework
- **LangChain**: Advanced LLM application framework
- **OpenAI API**: GPT models for AI capabilities
- **FAISS**: Efficient vector similarity search
- **PyPDF**: PDF document processing

### Frontend
- **Vanilla JavaScript**: No framework dependencies
- **Three.js**: 3D background animations
- **Chart.js**: Interactive analytics charts
- **CSS3**: Modern styling with animations

## 📦 Installation

### Prerequisites
- Python 3.11 or higher
- Node.js (for local development)
- OpenAI API key

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/humanist96/ai-nexus.git
   cd ai-nexus
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements_deploy.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   ```

5. **Run the application**
   ```bash
   uvicorn enhanced_rag_chatbot:app --reload --port 8001
   ```

6. **Open the frontend**
   - Open `premium_index.html` in your browser
   - Or use a local server: `python -m http.server 8000`

## 🚀 Deployment

### Frontend (GitHub Pages)
The frontend is automatically deployed to GitHub Pages when you push to the main branch.

### Backend (Render.com)
1. Fork this repository
2. Create a new Web Service on Render.com
3. Connect your GitHub repository
4. Set environment variables in Render dashboard:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `ALLOWED_ORIGINS`: Your frontend URL
5. Deploy!

## 📊 API Documentation

Once the backend is running, visit:
- API Documentation: `http://localhost:8001/docs`
- Alternative docs: `http://localhost:8001/redoc`

### Key Endpoints

- `POST /upload`: Upload PDF documents
- `POST /query`: Query the AI about uploaded documents
- `GET /sessions`: List all chat sessions
- `GET /analytics`: Get usage analytics
- `DELETE /clear_session/{session_id}`: Clear a specific session

## 🎯 Usage

1. **Upload a PDF**: Drag and drop or click to browse
2. **Ask Questions**: Use the chat interface to query your document
3. **View Analytics**: Check the Analytics tab for usage insights
4. **Manage Sessions**: Access previous conversations in the Sessions tab

## 📈 Analytics Features

- **Usage Overview**: Track uploads and queries over time
- **Query Types**: Analyze the types of questions being asked
- **Response Times**: Monitor AI response performance
- **Top Keywords**: See trending topics from queries

## 🔧 Configuration

### Environment Variables
- `OPENAI_API_KEY`: Required for AI functionality
- `ALLOWED_ORIGINS`: CORS configuration for deployment
- `PORT`: Server port (default: 8001)

### Frontend Configuration
Edit `config.js` to update API endpoints:
```javascript
const config = {
    development: {
        API_URL: 'http://localhost:8001'
    },
    production: {
        API_URL: 'https://your-backend-url.onrender.com'
    }
};
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for providing the GPT models
- LangChain community for the excellent framework
- Three.js for beautiful 3D visualizations
- Chart.js for analytics visualization

## 📞 Support

For support, email humanist96@gmail.com or open an issue on GitHub.

---

<div align="center">
  Made with ❤️ by humanist96
</div>
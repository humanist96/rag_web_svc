# AI Nexus - Premium PDF Analysis Platform

A sophisticated AI-powered PDF analysis platform with 3D animated backgrounds, glassmorphism design, and advanced RAG (Retrieval-Augmented Generation) capabilities.

![AI Nexus](https://img.shields.io/badge/AI-Nexus-6366f1?style=for-the-badge&logo=artificial-intelligence&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Three.js](https://img.shields.io/badge/Three.js-r128-black?style=for-the-badge&logo=three.js&logoColor=white)

## 🌟 Features

### 🎨 Premium Design
- **3D Particle Animation**: Dynamic Three.js background with floating particles
- **Glassmorphism UI**: Modern frosted glass effects
- **Gradient Animations**: Smooth color transitions
- **Micro-interactions**: Responsive hover and click effects

### 🤖 AI Capabilities
- **PDF Analysis**: Upload and analyze any PDF document
- **RAG-based Q&A**: Ask questions about your documents
- **Session Management**: Multiple concurrent PDF sessions
- **Source Citations**: References with page numbers

### 💎 User Experience
- **Drag & Drop Upload**: Intuitive file upload
- **Real-time Progress**: Visual upload progress tracking
- **Toast Notifications**: Elegant feedback system
- **Floating Action Button**: Quick access to actions
- **Analytics Dashboard**: Usage statistics (coming soon)

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-nexus.git
cd ai-nexus
```

2. Install dependencies:
```bash
pip install -r requirements_simple.txt
```

3. Set up OpenAI API key:
Create a `.env` file with:
```
OPENAI_API_KEY=your-api-key-here
```

### Running the Application

1. Start the backend server:
```bash
python enhanced_rag_chatbot.py
```
Or use the batch file:
```bash
start_premium_server.bat
```

2. Open `premium_index.html` in your browser

3. Upload a PDF and start chatting!

## 📁 Project Structure

```
ai-nexus/
├── premium_index.html      # Main frontend interface
├── premium_styles.css      # Modern CSS with animations
├── premium_script.js       # Frontend logic & 3D effects
├── enhanced_rag_chatbot.py # FastAPI backend server
├── requirements_simple.txt # Python dependencies
├── start_premium_server.bat # Server startup script
└── uploads/               # Temporary PDF storage
```

## 🛠️ Technology Stack

### Backend
- **FastAPI**: Modern web framework
- **LangChain**: RAG implementation
- **OpenAI**: GPT-3.5 for chat
- **FAISS**: Vector database
- **PyPDF**: PDF processing

### Frontend
- **Three.js**: 3D animations
- **Vanilla JS**: No framework overhead
- **CSS3**: Modern animations
- **Font Awesome**: Icons

## 🎯 Key Features Explained

### Session Management
Each uploaded PDF creates a unique session, allowing:
- Multiple PDFs to be analyzed concurrently
- Conversation history per document
- Session persistence and recovery

### Vector Embeddings
- Documents are chunked and embedded using OpenAI embeddings
- FAISS enables fast similarity search
- Relevant chunks are retrieved for each query

### 3D Background
- 2000 animated particles
- Dynamic color gradients
- Responsive to window resizing
- Optimized for performance

## 🐛 Troubleshooting

### Upload Fails
1. Check server is running on http://localhost:8001
2. Verify `.env` file exists with valid API key
3. Ensure `uploads/` directory exists
4. Check browser console for errors

### Missing Dependencies
```bash
pip install pypdf faiss-cpu
```

### API Key Issues
Ensure your OpenAI API key is valid and has sufficient credits.

## 📝 Testing

Run the debug script to verify setup:
```bash
python test_upload_debug.py
```

## 🔮 Future Enhancements
- Voice input support
- File attachments in chat
- Analytics dashboard
- Export conversations
- Multi-language support

## 📄 License
MIT License - feel free to use and modify!

## 🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first.

---
Built with ❤️ using AI-powered development
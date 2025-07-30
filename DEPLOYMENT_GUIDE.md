# Deployment Guide for AI Nexus

## Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Login with your account (humanist96@gmail.com)
3. Create a new repository with these settings:
   - Repository name: `rag_web_svc`
   - Description: "Premium PDF Analysis Platform with AI-powered chat and analytics"
   - Public repository
   - DO NOT initialize with README, .gitignore, or license (we already have them)
4. Click "Create repository"

## Step 2: Push Code to GitHub

After creating the repository, run these commands in your terminal:

```bash
# Add the remote repository (if not already added)
git remote add origin https://github.com/humanist96/rag_web_svc.git

# Push the code
git push -u origin master
```

If you get an authentication error, you may need to:
1. Create a Personal Access Token at https://github.com/settings/tokens
2. Use the token as your password when prompted

## Step 3: Enable GitHub Pages for Frontend

1. Go to your repository settings: https://github.com/humanist96/rag_web_svc/settings
2. Scroll down to "Pages" section
3. Under "Source", select "Deploy from a branch"
4. Choose "master" branch and "/ (root)" folder
5. Click "Save"
6. Your frontend will be available at: https://humanist96.github.io/rag_web_svc/premium_index.html

## Step 4: Deploy Backend to Render.com

1. Go to https://render.com and sign up/login
2. Click "New +" → "Web Service"
3. Connect your GitHub account if not already connected
4. Select the "rag_web_svc" repository
5. Configure the service:
   - Name: `rag-web-svc-backend`
   - Region: Oregon (US West)
   - Branch: master
   - Runtime: Python 3
   - Build Command: `pip install -r requirements_deploy.txt`
   - Start Command: `uvicorn enhanced_rag_chatbot:app --host 0.0.0.0 --port $PORT`
6. Add environment variables:
   - Click "Advanced" → "Add Environment Variable"
   - Add `OPENAI_API_KEY` with your OpenAI API key value
   - Add `ALLOWED_ORIGINS` with value `https://humanist96.github.io`
7. Click "Create Web Service"

## Step 5: Update Frontend Configuration

After your backend is deployed on Render:

1. Get your Render backend URL (will be something like `https://rag-web-svc-backend.onrender.com`)
2. Update `config.js` in your repository:
   ```javascript
   production: {
       API_URL: 'https://rag-web-svc-backend.onrender.com'  // Your actual Render URL
   }
   ```
3. Commit and push the change:
   ```bash
   git add config.js
   git commit -m "Update production API URL"
   git push
   ```

## Step 6: Test Your Deployment

1. Frontend: Visit https://humanist96.github.io/rag_web_svc/premium_index.html
2. Backend API docs: Visit https://rag-web-svc-backend.onrender.com/docs
3. Test the full flow:
   - Upload a PDF
   - Ask questions in the chat
   - Check analytics

## Continuous Deployment

Both GitHub Pages and Render.com support automatic deployment:
- **Frontend**: Automatically updates when you push to master branch
- **Backend**: Automatically redeploys when you push to master branch

## Troubleshooting

### CORS Issues
If you get CORS errors:
1. Check that `ALLOWED_ORIGINS` in Render matches your GitHub Pages URL
2. Make sure the URL doesn't have a trailing slash

### Backend Not Starting
1. Check Render logs for errors
2. Verify all environment variables are set correctly
3. Make sure OpenAI API key is valid

### Frontend Can't Connect to Backend
1. Verify the backend URL in `config.js` is correct
2. Check that the backend is running (visit the /docs endpoint)
3. Look for errors in browser console

## Local Development

To run locally while deployed:
1. Keep `.env` file with your API keys
2. Run backend: `uvicorn enhanced_rag_chatbot:app --reload --port 8001`
3. The frontend will automatically use localhost when opened locally
"""
Clean up old files not related to premium version
"""
import os
import shutil
import sys

# UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')

# Files to keep (premium version and essential files)
KEEP_FILES = {
    # Premium version files
    'premium_index.html',
    'premium_styles.css', 
    'premium_script.js',
    
    # Enhanced backend
    'enhanced_rag_chatbot.py',
    
    # Essential files
    '.env',
    'samilpwc_ai-agent-in-action.pdf',
    'requirements_simple.txt',
    'start_premium_server.bat',
    'test_upload_debug.py',
    'cleanup_old_files.py',
    
    # Directories
    'uploads',
    'faiss_index',
    '.git',
    'venv',
    '__pycache__'
}

# Files to definitely remove
REMOVE_FILES = {
    # Old HTML/CSS/JS files
    'index.html',
    'styles.css',
    'script.js',
    'enhanced_index.html',
    'enhanced_styles.css',
    'enhanced_script.js',
    
    # Old Python files
    'embeddings.py',
    'rag_chatbot.py',
    'simple_embeddings.py',
    'simple_rag_chatbot.py',
    'run_all.py',
    'setup.py',
    'setup_api_key.py',
    'test_setup.py',
    'test_api.py',
    
    # Old scripts
    'run_chatbot.bat',
    'run_chatbot.sh',
    'requirements.txt',
    
    # Test files
    'test_premium.html',
    'test_upload.html',
    
    # README (will create new one)
    'README.md'
}

def cleanup():
    """Remove old files"""
    removed_count = 0
    kept_count = 0
    
    print("🧹 Cleaning up old files...\n")
    
    # Get all files in directory
    for item in os.listdir('.'):
        if item in KEEP_FILES:
            print(f"✅ Keeping: {item}")
            kept_count += 1
        elif item in REMOVE_FILES:
            try:
                if os.path.isfile(item):
                    os.remove(item)
                    print(f"❌ Removed: {item}")
                elif os.path.isdir(item):
                    shutil.rmtree(item)
                    print(f"❌ Removed directory: {item}")
                removed_count += 1
            except Exception as e:
                print(f"⚠️  Failed to remove {item}: {e}")
        else:
            print(f"❓ Skipped (not in list): {item}")
    
    print(f"\n📊 Summary:")
    print(f"  - Files kept: {kept_count}")
    print(f"  - Files removed: {removed_count}")
    
    return removed_count

if __name__ == "__main__":
    removed = cleanup()
    
    if removed > 0:
        print("\n✅ Cleanup completed successfully!")
    else:
        print("\n✅ No files to remove.")
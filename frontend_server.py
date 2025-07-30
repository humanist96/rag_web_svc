#!/usr/bin/env python3
"""
Simple static file server for frontend
"""
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
import socketserver
import mimetypes

class CustomHTTPRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory='.', **kwargs)
    
    def do_GET(self):
        # Redirect root to premium_index.html
        if self.path == '/':
            self.path = '/premium_index.html'
        
        # Serve static files
        return SimpleHTTPRequestHandler.do_GET(self)
    
    def end_headers(self):
        # Add security headers
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header('X-XSS-Protection', '1; mode=block')
        self.send_header('Referrer-Policy', 'strict-origin-when-cross-origin')
        
        # Add caching headers
        if self.path.endswith('.css') or self.path.endswith('.js'):
            self.send_header('Cache-Control', 'public, max-age=86400')
        elif self.path.endswith('.html'):
            self.send_header('Cache-Control', 'public, max-age=3600')
        
        SimpleHTTPRequestHandler.end_headers(self)
    
    def log_message(self, format, *args):
        # Custom logging format
        sys.stdout.write(f"{self.address_string()} - [{self.log_date_time_string()}] {format%args}\n")
        sys.stdout.flush()

def main():
    # Get port from environment
    PORT = int(os.environ.get('PORT', 8080))
    
    # Set up MIME types
    mimetypes.init()
    mimetypes.add_type('application/javascript', '.js')
    mimetypes.add_type('text/css', '.css')
    
    # Create server
    with socketserver.TCPServer(("", PORT), CustomHTTPRequestHandler) as httpd:
        print(f"Frontend server running on port {PORT}")
        print(f"Serving files from: {os.getcwd()}")
        print(f"Access the app at: http://localhost:{PORT}/")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            httpd.shutdown()

if __name__ == "__main__":
    main()
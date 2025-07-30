"""
Vercel Serverless Function Entry Point
"""
import os
os.environ['VERCEL_FUNCTION'] = 'true'

from fastapi import FastAPI
from enhanced_rag_chatbot import app

# Vercel expects a handler
handler = app
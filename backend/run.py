#!/usr/bin/env python3
"""
PersonaCast Backend Runner

Quick start script for running the PersonaCast server
"""

import os
import sys
from pathlib import Path

# Add backend/src to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import and run app
from src.app import create_app

if __name__ == '__main__':
    app = create_app()
    
    print("\n" + "=" * 80)
    print("Starting PersonaCast Backend Server")
    print("=" * 80)
    
    # Check configuration
    if not os.getenv('DEEPGRAM_API_KEY'):
        print("\n⚠️  WARNING: DEEPGRAM_API_KEY not set!")
        print("Please set it in .env file or environment variables")
        print("Get your API key from: https://console.deepgram.com/\n")
    
    print("\nServer starting at: http://localhost:5000")
    print("\nPress CTRL+C to stop\n")
    print("=" * 80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
#!/usr/bin/env python3
"""
Web-based AI Context Manager Demo

A Flask web application that demonstrates the AI Context Manager capabilities
through an interactive web interface.

Features:
- Interactive research assistant
- Real-time context management
- Semantic search demonstration
- Agent statistics and monitoring
- Persistent memory visualization
"""

import os
import sys
import json
import asyncio
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit
import threading
import time

# Add the parent directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ai_context_manager.simple_api import create_agent_context_manager
from demo_apps.research_assistant.app import ResearchAssistant

# Find demo config file in common locations
def find_demo_config():
    """Find demo config file in common locations."""
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "demo-config.toml"),  # Same directory
        os.path.join(os.path.dirname(__file__), "..", "demo-config.toml"),  # Parent directory
        os.path.join(os.path.dirname(__file__), "..", "..", "demo_apps", "demo-config.toml"),  # Project demo dir
        "demo_apps/demo-config.toml",  # Relative to current working directory
        "demo-config.toml",  # Current directory
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # Return default if not found
    return "demo_config.toml"

DEMO_CONFIG_PATH = find_demo_config()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ai-context-manager-demo-2025'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global research assistant instance
assistant = None

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint for Docker."""
    try:
        # Basic health check
        if assistant is None:
            return jsonify({"status": "unhealthy", "reason": "assistant not initialized"}), 503
        
        # Test basic functionality
        status = assistant.get_agent_status()
        return jsonify({
            "status": "healthy",
            "assistant": "initialized",
            "agent_id": status.get('agent_id', 'unknown'),
            "goals_count": len(status.get('goals', [])),
            "timestamp": datetime.now().isoformat()
        }), 200
    except Exception as e:
        return jsonify({
            "status": "unhealthy", 
            "reason": str(e),
            "timestamp": datetime.now().isoformat()
        }), 503

@app.route('/api/status')
def get_status():
    """Get agent status."""
    if assistant:
        status = assistant.get_agent_status()
        return jsonify(status)
    return jsonify({"error": "Assistant not initialized"})

@app.route('/api/research', methods=['POST'])
def research_topic():
    """Research a topic."""
    data = request.json
    topic = data.get('topic', '')
    depth = data.get('depth', 'medium')
    
    if not topic:
        return jsonify({"error": "Topic is required"})
    
    try:
        # Run research in a separate thread
        def run_research():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(assistant.research_topic(topic, depth))
            loop.close()
            
            # Emit results via WebSocket
            socketio.emit('research_complete', {
                'topic': topic,
                'result': result,
                'timestamp': datetime.now().isoformat()
            })
        
        thread = threading.Thread(target=run_research)
        thread.start()
        
        return jsonify({"status": "research_started", "topic": topic})
        
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/ask', methods=['POST'])
def ask_question():
    """Ask a question to the assistant."""
    data = request.json
    question = data.get('question', '')
    
    if not question:
        return jsonify({"error": "Question is required"})
    
    try:
        answer = assistant.ask_question(question)
        return jsonify({
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/search', methods=['POST'])
def search_knowledge():
    """Search through accumulated knowledge."""
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({"error": "Query is required"})
    
    try:
        results = assistant.agent.search_similar(query, limit=10)
        return jsonify({
            "query": query,
            "results": results,
            "count": len(results),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/summary/<topic>')
def get_summary(topic):
    """Get research summary for a topic."""
    try:
        summary = assistant.get_research_summary(topic)
        return jsonify(summary)
    except Exception as e:
        return jsonify({"error": str(e)})

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    emit('status', {'message': 'Connected to AI Context Manager Demo'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print('Client disconnected')

def initialize_assistant():
    """Initialize the research assistant."""
    global assistant
    try:
        # Use demo configuration
        print(f"🔧 Using config: {DEMO_CONFIG_PATH}")
        assistant = ResearchAssistant("web-demo-assistant", config_path=DEMO_CONFIG_PATH)
        print("✅ Research Assistant initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Research Assistant: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        
        # Fallback: try without config
        try:
            print("🔄 Trying fallback initialization...")
            assistant = ResearchAssistant("web-demo-assistant")
            print("✅ Research Assistant initialized with fallback config")
        except Exception as e2:
            print(f"❌ Failed to initialize Research Assistant with fallback: {e2}")
            print(f"   Error type: {type(e2).__name__}")
            traceback.print_exc()
            assistant = None

if __name__ == '__main__':
    print("🚀 Starting AI Context Manager Web Demo...")
    initialize_assistant()
    
    # Check if running in production (Docker)
    import os
    if os.getenv('FLASK_ENV') == 'production':
        # Use production server with unsafe Werkzeug for Docker
        socketio.run(app, debug=False, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
    else:
        # Use development server
        socketio.run(app, debug=True, host='0.0.0.0', port=5000)

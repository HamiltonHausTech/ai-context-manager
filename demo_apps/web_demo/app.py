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

# Get the directory containing this file
DEMO_DIR = os.path.dirname(__file__)
DEMO_CONFIG_PATH = os.path.join(DEMO_DIR, "demo-config.toml")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ai-context-manager-demo-2025'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global research assistant instance
assistant = None

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

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
        assistant = ResearchAssistant("web-demo-assistant", config_path=DEMO_CONFIG_PATH)
        print("✅ Research Assistant initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Research Assistant: {e}")
        # Fallback: try without config
        try:
            assistant = ResearchAssistant("web-demo-assistant")
            print("✅ Research Assistant initialized with fallback config")
        except Exception as e2:
            print(f"❌ Failed to initialize Research Assistant with fallback: {e2}")
            assistant = None

if __name__ == '__main__':
    print("🚀 Starting AI Context Manager Web Demo...")
    initialize_assistant()
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)

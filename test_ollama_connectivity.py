#!/usr/bin/env python3
"""
Simple script to test Ollama connectivity and demonstrate the auto-fallback behavior.
This script will help you verify when Ollama is available on your network.
"""

import os
import requests
import time

def test_ollama_connectivity(host="http://localhost:11434", timeout=5):
    """Test if Ollama is accessible at the given host."""
    try:
        print(f"Testing Ollama connectivity to {host}...")
        response = requests.get(f"{host}/api/tags", timeout=timeout)
        response.raise_for_status()
        
        models = response.json().get("models", [])
        print(f"[OK] Ollama is available at {host}")
        print(f"   Found {len(models)} models:")
        for model in models:
            print(f"   - {model.get('name', 'Unknown')}")
        return True
        
    except requests.exceptions.ConnectionError:
        print(f"[FAIL] Ollama not available at {host} (connection refused)")
        return False
    except requests.exceptions.Timeout:
        print(f"[FAIL] Ollama timeout at {host} (took longer than {timeout}s)")
        return False
    except Exception as e:
        print(f"[FAIL] Ollama error at {host}: {e}")
        return False

def test_different_hosts():
    """Test connectivity to different common Ollama hosts."""
    hosts_to_test = [
        "http://localhost:11434",
        "http://192.168.0.156:11434",  # Your network Ollama
        "http://127.0.0.1:11434",
    ]
    
    print("=" * 60)
    print("Ollama Connectivity Test")
    print("=" * 60)
    
    available_hosts = []
    
    for host in hosts_to_test:
        if test_ollama_connectivity(host):
            available_hosts.append(host)
        print()
    
    if available_hosts:
        print(f"[OK] Found {len(available_hosts)} available Ollama instance(s)")
        print("You can use auto_fallback summarizer with any of these hosts:")
        for host in available_hosts:
            print(f"   OLLAMA_HOST={host}")
    else:
        print("[INFO] No Ollama instances found")
        print("The auto_fallback summarizer will use the naive fallback")
    
    return available_hosts

if __name__ == "__main__":
    # Check environment variable first
    env_host = os.getenv("OLLAMA_HOST")
    if env_host:
        print(f"Testing environment variable OLLAMA_HOST={env_host}")
        test_ollama_connectivity(env_host)
        print()
    
    # Test common hosts
    available_hosts = test_different_hosts()
    
    print("\n" + "=" * 60)
    print("Configuration Recommendations:")
    print("=" * 60)
    
    if available_hosts:
        print("[OK] Ollama is available! You can:")
        print("1. Set type = 'auto_fallback' in config.toml")
        print("2. Set OLLAMA_HOST environment variable to one of the available hosts")
        print("3. The system will automatically use Ollama when available")
    else:
        print("[INFO] No Ollama found. You can:")
        print("1. Keep using type = 'auto_fallback' (will use naive fallback)")
        print("2. Or set type = 'naive' for explicit naive mode")
        print("3. The system will work perfectly with naive summarization")

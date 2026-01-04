#!/usr/bin/env python3
"""
QIE V3 AI Risk Oracle Frontend Launcher
Starts the React dashboard for hackathon demo
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def check_node_installed():
    """Check if Node.js is installed"""
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Node.js found: {result.stdout.strip()}")
            return True
        else:
            print("❌ Node.js not found")
            return False
    except FileNotFoundError:
        print("❌ Node.js not found")
        return False

def check_npm_installed():
    """Check if npm is installed"""
    try:
        result = subprocess.run(['npm', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ npm found: {result.stdout.strip()}")
            return True
        else:
            print("❌ npm not found")
            return False
    except FileNotFoundError:
        print("❌ npm not found")
        return False

def install_dependencies():
    """Install npm dependencies"""
    print("📦 Installing frontend dependencies...")
    try:
        result = subprocess.run(['npm', 'install'], cwd='frontend', check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def start_frontend():
    """Start the React development server"""
    print("🚀 Starting QIE V3 AI Risk Oracle Dashboard...")
    print("📊 Dashboard will be available at: http://localhost:3000")
    print("🔗 Make sure the backend API is running on port 8001")
    print()
    print("Press Ctrl+C to stop the dashboard")
    print("-" * 50)
    
    try:
        # Start the React development server
        subprocess.run(['npm', 'start'], cwd='frontend', check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start frontend: {e}")
        return False
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
        return True

def main():
    """Main function"""
    print("🏆 QIE V3 AI Risk Oracle - Frontend Launcher")
    print("=" * 50)
    
    # Check if frontend directory exists
    frontend_dir = Path("frontend")
    if not frontend_dir.exists():
        print("❌ Frontend directory not found")
        print("Please run this script from the project root directory")
        return 1
    
    # Check prerequisites
    if not check_node_installed():
        print("\n📋 Please install Node.js:")
        print("   - Download from: https://nodejs.org/")
        print("   - Or use package manager: brew install node (macOS) / apt install nodejs (Ubuntu)")
        return 1
    
    if not check_npm_installed():
        print("\n📋 Please install npm (usually comes with Node.js)")
        return 1
    
    # Check if package.json exists
    package_json = frontend_dir / "package.json"
    if not package_json.exists():
        print("❌ package.json not found in frontend directory")
        return 1
    
    # Check if node_modules exists, install if not
    node_modules = frontend_dir / "node_modules"
    if not node_modules.exists():
        print("📦 Node modules not found, installing dependencies...")
        if not install_dependencies():
            return 1
    else:
        print("✅ Dependencies already installed")
    
    # Start the frontend
    print("\n🚀 Launching Dashboard...")
    start_frontend()
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
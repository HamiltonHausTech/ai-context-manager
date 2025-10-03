#!/usr/bin/env python3
"""
PyPI Package Setup Script

This script helps prepare the AI Context Manager package for PyPI publication.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return None

def clean_build_directories():
    """Clean up build directories."""
    directories_to_clean = ['build', 'dist', '*.egg-info']
    
    for pattern in directories_to_clean:
        if '*' in pattern:
            # Handle wildcards
            import glob
            for path in glob.glob(pattern):
                if os.path.exists(path):
                    shutil.rmtree(path)
                    print(f"🗑️  Removed {path}")
        else:
            if os.path.exists(pattern):
                shutil.rmtree(pattern)
                print(f"🗑️  Removed {pattern}")

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")

def check_required_files():
    """Check if all required files exist."""
    required_files = [
        'pyproject.toml',
        'README.md',
        'LICENSE',
        'ai_context_manager/__init__.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        sys.exit(1)
    
    print("✅ All required files present")

def build_package():
    """Build the package."""
    # Install build dependencies
    run_command("pip install build twine", "Installing build dependencies")
    
    # Build the package
    run_command("python -m build", "Building package")
    
    # Check the build
    run_command("python -m build --check", "Checking package build")

def test_package():
    """Test the built package."""
    # Install the package in test mode
    run_command("pip install dist/*.whl", "Installing package for testing")
    
    # Run basic import test
    try:
        import ai_context_manager
        print("✅ Package import test passed")
    except ImportError as e:
        print(f"❌ Package import test failed: {e}")
        return False
    
    # Run quick functionality test
    try:
        from ai_context_manager.simple_api import create_context_manager
        ctx = create_context_manager()
        print("✅ Basic functionality test passed")
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False
    
    return True

def create_source_distribution():
    """Create source distribution."""
    run_command("python setup.py sdist", "Creating source distribution")

def main():
    """Main setup function."""
    print("🚀 AI Context Manager PyPI Package Setup")
    print("=" * 50)
    
    # Check prerequisites
    check_python_version()
    check_required_files()
    
    # Clean previous builds
    print("\n🧹 Cleaning previous builds...")
    clean_build_directories()
    
    # Build package
    print("\n📦 Building package...")
    build_package()
    
    # Test package
    print("\n🧪 Testing package...")
    if not test_package():
        print("❌ Package tests failed. Aborting.")
        sys.exit(1)
    
    print("\n✅ Package setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Review the built package in the 'dist' directory")
    print("2. Test the package: pip install dist/*.whl")
    print("3. Upload to PyPI: python -m twine upload dist/*")
    print("4. Or upload to Test PyPI first: python -m twine upload --repository testpypi dist/*")
    
    print("\n📚 Useful commands:")
    print("- Test locally: pip install dist/*.whl")
    print("- Upload to Test PyPI: python -m twine upload --repository testpypi dist/*")
    print("- Upload to PyPI: python -m twine upload dist/*")
    print("- Check package: python -m twine check dist/*")

if __name__ == "__main__":
    main()

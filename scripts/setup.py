# =================================================================
# scripts/setup.py (for easy development setup)
#!/usr/bin/env python3
"""Setup script for development environment."""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"✅ {description} completed")
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        sys.exit(1)

def main():
    """Set up the development environment."""
    print("🚀 Setting up Remote Sensing Lab development environment with uv")
    
    # Check if uv is installed
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
        print("✅ uv is already installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ uv is not installed. Please install it first:")
        print("   curl -LsSf https://astral.sh/uv/install.sh | sh")
        sys.exit(1)
    
    # Install Python and dependencies
    run_command("uv python install 3.11", "Installing Python 3.11")
    run_command("uv sync --all-extras", "Installing all dependencies")
    
    # Set up pre-commit hooks
    # run_command("uv run pre-commit install", "Setting up pre-commit hooks")
    
    # Run initial tests
    # run_command("uv run pytest tests/ -v", "Running initial tests")
    
    # Create necessary directories
    Path("results").mkdir(exist_ok=True)
    Path("data/indian_pines").mkdir(parents=True, exist_ok=True)
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Download Indian Pines dataset (optional):")
    print("   http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes")
    print("2. Start Jupyter Lab: make jupyter")
    print("3. Run synthetic test: make run-synthetic")
    print("4. Run benchmarks: make benchmark")

if __name__ == "__main__":
    main()
#!/bin/bash

# rebuild_docs.sh - Script to rebuild UQ-PhysiCell documentation
# Usage: ./rebuild_docs.sh [options]
# Options:
#   --clean    Clean build directory before building
#   --serve    Start a local server after building
#   --help     Show this help message

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOC_DIR="$SCRIPT_DIR"
PROJECT_ROOT="$(dirname "$DOC_DIR")"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show help
show_help() {
    echo "UQ-PhysiCell Documentation Builder"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --clean     Clean build directory before building"
    echo "  --serve     Start a local server after building (port 8000)"
    echo "  --live      Start live reload server for development"
    echo "  --help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Build documentation"
    echo "  $0 --clean           # Clean and build documentation"
    echo "  $0 --clean --serve   # Clean, build, and serve documentation"
    echo "  $0 --live            # Start live reload server"
}

# Parse command line arguments
CLEAN=false
SERVE=false
LIVE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN=true
            shift
            ;;
        --serve)
            SERVE=true
            shift
            ;;
        --live)
            LIVE=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Change to documentation directory
cd "$DOC_DIR"

print_status "Building UQ-PhysiCell documentation..."
print_status "Documentation directory: $DOC_DIR"
print_status "Project root: $PROJECT_ROOT"

# Check if we're in a virtual environment
if [[ -n "$VIRTUAL_ENV" ]]; then
    print_status "Virtual environment detected: $(basename "$VIRTUAL_ENV")"
else
    print_warning "No virtual environment detected. Consider using a virtual environment."
fi

# Check if required packages are installed
print_status "Checking dependencies..."

# Check for sphinx
if ! command -v sphinx-build &> /dev/null; then
    print_error "sphinx-build not found. Installing documentation dependencies..."
    pip install -r requirements.txt
fi

# Check if UQ-PhysiCell is importable (for version detection)
if ! python -c "import uq_physicell" 2>/dev/null; then
    print_warning "UQ-PhysiCell not importable. Installing in development mode..."
    cd "$PROJECT_ROOT"
    pip install -e .
    cd "$DOC_DIR"
fi

# Get current version
VERSION=$(python -c "try:
    from uq_physicell.VERSION import __version__
    print(__version__)
except ImportError:
    print('unknown')
")

print_status "Building documentation for UQ-PhysiCell version: $VERSION"

# Clean if requested
if [[ "$CLEAN" == true ]]; then
    print_status "Cleaning build directory..."
    make clean
fi

# Handle live reload server
if [[ "$LIVE" == true ]]; then
    print_status "Starting live reload server..."
    print_status "Documentation will be available at: http://localhost:8000"
    print_status "Press Ctrl+C to stop the server"
    make livehtml
    exit 0
fi

# Build documentation
print_status "Building HTML documentation..."
if make html; then
    print_success "Documentation built successfully!"
    
    # Check if build directory exists and has content
    if [[ -d "_build/html" ]] && [[ -f "_build/html/index.html" ]]; then
        BUILD_SIZE=$(du -sh _build/html | cut -f1)
        NUM_FILES=$(find _build/html -name "*.html" | wc -l)
        print_success "Generated $NUM_FILES HTML files (total size: $BUILD_SIZE)"
        print_status "Documentation location: $DOC_DIR/_build/html/index.html"
    else
        print_error "Build completed but output files not found!"
        exit 1
    fi
else
    print_error "Documentation build failed!"
    exit 1
fi

# Serve documentation if requested
if [[ "$SERVE" == true ]]; then
    print_status "Starting local documentation server..."
    print_status "Documentation will be available at: http://localhost:8000"
    print_status "Press Ctrl+C to stop the server"
    
    cd _build/html
    
    # Try different Python HTTP servers
    if command -v python3 &> /dev/null; then
        python3 -m http.server 8000
    elif command -v python &> /dev/null; then
        python -m http.server 8000
    else
        print_error "Python not found. Cannot start HTTP server."
        exit 1
    fi
fi

print_success "Documentation build completed!"
print_status "To view the documentation, open: $DOC_DIR/_build/html/index.html"

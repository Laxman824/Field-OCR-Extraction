# First, make sure you're in your project root directory
cd /mount/src/field-ocr-extraction

# Remove any existing problematic files
rm -rf src/__init__.py
rm -rf src/utils/__init__.py
rm -rf src/models/__init__.py
rm -rf src/config/__init__.py

# Create directories if they don't exist
mkdir -p src/utils
mkdir -p src/models
mkdir -p src/config

# Create empty __init__.py files (don't add any content)
touch src/__init__.py
touch src/utils/__init__.py
touch src/models/__init__.py
touch src/config/__init__.py
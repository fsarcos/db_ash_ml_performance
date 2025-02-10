#!/bin/bash

# Create a clean directory for packaging
PACKAGE_DIR="oracle_db_ml_ash_monitoring"
VENV_DIR="venv"
REQUIREMENTS_FILE="requirements.txt"

echo "Creating package directory..."
rm -rf $PACKAGE_DIR
mkdir -p $PACKAGE_DIR

# Create requirements.txt if it doesn't exist
if [ ! -f $REQUIREMENTS_FILE ]; then
    echo "Creating requirements.txt..."
    cat > $REQUIREMENTS_FILE << EOF
cx_Oracle==7.3.0
pandas==1.1.5
numpy==1.19.5
scikit-learn==0.24.2
matplotlib==3.3.4
seaborn==0.11.2
Pillow==8.4.0  # Last version supporting Python 3.6
EOF
fi

# Create and activate virtual environment
echo "Creating virtual environment..."
python3 -m venv $VENV_DIR
source $VENV_DIR/bin/activate

# Download packages without installing, specifically requesting wheels
echo "Downloading packages..."
pip download --only-binary :all: -r $REQUIREMENTS_FILE -d $PACKAGE_DIR/packages

# Copy source files
echo "Copying source files..."
cp -r src/python/* $PACKAGE_DIR/
cp $REQUIREMENTS_FILE $PACKAGE_DIR/

# Create deployment instructions for offline installation
cat > $PACKAGE_DIR/INSTALL.txt << EOF
Oracle ASH Analysis Tool Offline Installation Instructions

1. Prerequisites:
   - Python 3.6
   - Oracle Instant Client (matching your Oracle Database version)
   - Required OS packages: libaio1 (Linux)

2. Installation Steps:
   a. Create a virtual environment:
      python3 -m venv venv

   b. Activate the virtual environment:
      source venv/bin/activate

   c. Install dependencies from local packages:
      pip install --no-index --find-links packages -r requirements.txt

3. Configuration:
   - Update config.py with your Oracle environment settings
   - Ensure ORACLE_HOME and other environment variables are set correctly

4. Running the Tool:
   - Collect historical data: python collect_ash_data.py
   - Train the model: python train_model.py
   - Start monitoring: python monitor_ash.py
EOF

# Create archive
echo "Creating archive..."
tar -czf oracle_db_ml_ash_monitoring.tar.gz $PACKAGE_DIR

# Cleanup
echo "Cleaning up..."
rm -rf $PACKAGE_DIR
deactivate
rm -rf $VENV_DIR

echo "Package created: oracle_db_ml_ash_monitoring.tar.gz"
echo "Transfer this file to your target server and follow the instructions in INSTALL.txt"

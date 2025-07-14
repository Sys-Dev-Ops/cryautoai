#!/bin/bash

# Debugging version
{
    echo "=== STARTING RUN: $(date) ==="
    echo "Current PATH: $PATH"
    echo "Python path: $(which python)"
    
    PROJECT_DIR="/home/ubuntu/cryautoai/"
    cd $PROJECT_DIR || exit 1
    
    # Activate venv
    source venv/bin/activate
    echo "Virtualenv: $VIRTUAL_ENV"
    
    # Verify packages
    pip list | grep ccxt
    
    # Run script
    python gemini_crypto_trader.py
    
    echo "=== FINISHED RUN: $(date) ==="
    echo ""
} >> /home/ubuntu/cryautoai/cron.log 2>&1

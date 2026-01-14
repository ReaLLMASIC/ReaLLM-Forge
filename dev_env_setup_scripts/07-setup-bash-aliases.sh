#!/bin/bash

# Check if source file exists
if [ ! -f bash_aliases ]; then
    echo "Error: 'bash_aliases' file not found in the current directory."
    exit 1
fi

# Backup existing ~/.bash_aliases if it exists
if [ -f "$HOME/.bash_aliases" ]; then
    cp "$HOME/.bash_aliases" "$HOME/.bash_aliases.bak"
    if [ $? -eq 0 ]; then
        echo "Existing ~/.bash_aliases backed up to ~/.bash_aliases.bak"
    else
        echo "Warning: Failed to backup existing ~/.bash_aliases"
    fi
fi

# Copy new aliases file
cp bash_aliases "$HOME/.bash_aliases"
if [ $? -eq 0 ]; then
    echo "Successfully installed new bash aliases to ~/.bash_aliases"
else
    echo "Error: Failed to copy bash_aliases to ~/.bash_aliases"
    exit 1
fi

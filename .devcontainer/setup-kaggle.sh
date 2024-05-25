#!/bin/bash

# Create the .kaggle directory
mkdir -p /home/vscode/.kaggle

# Add the kaggle.json file with your credentials
cat <<EOF > /home/vscode/.kaggle/kaggle.json
{
  "username": "mauricioalvarezq",
  "key": "0ef9c395468b646e37543e131fd03cb4"
}
EOF

# Set permissions
chmod 600 /home/vscode/.kaggle/kaggle.json
mkdir -p /home/vscode/datasets

# Download the dataset
kaggle datasets download -d arjunbhasin2013/ccdata -p /home/vscode/datasets

# Unzip the dataset
unzip /home/vscode/datasets/ccdata.zip -d /home/vscode/datasets

#!/bin/bash

set -e # 에러 발생 시 스크립트 중단

echo "Updating package list..."
apt update -y

echo "Installing essential packages..."
apt install -y tmux neovim aria2 p7zip-full

REQ_FILE="requirements.txt"

if [ -f "$REQ_FILE" ]; then
  echo "Found requirements.txt, installing Python packages..."
  pip install -r "$REQ_FILE"
else
  echo "No requirements.txt found at $REQ_FILE, skipping pip install."
fi

# Git config
echo "Configuring git user..."
git config --global user.name "whxtdxsa"
git config --global user.email "whitdisa03@gmail.com"
git config --global credential.helper store

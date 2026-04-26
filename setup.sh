#!/bin/bash
# ╔══════════════════════════════════════════════════════╗
# ║     AutoValue AI - One-Click Setup Script           ║
# ╚══════════════════════════════════════════════════════╝

set -e
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║        🚗  AUTOVALUE AI  -  SETUP              ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════╝${NC}"
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
    echo -e "${RED}❌ Python 3 not found. Install from https://python.org${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python $(python3 --version) found${NC}"

# Create venv if not exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}📦 Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null

echo -e "${YELLOW}📥 Installing dependencies...${NC}"
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo -e "${YELLOW}🤖 Training ML models...${NC}"
python model_trainer.py

echo ""
echo -e "${GREEN}✅ Setup complete!${NC}"
echo -e "${CYAN}🚀 Launching AutoValue AI...${NC}"
echo ""
streamlit run app.py

set -euo pipefail

# Set up the environment for the project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

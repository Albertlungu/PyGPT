#!/bin/bash

# This script sets up the development environment by installing necessary dependencies and configuring settings.

# Install pyenv if not already
curl https://pyenv.run | bash
pyenv --version


pyenv install 3.10 # This will install python 3.10.19 by default
pyenv local 3.10

# Run this to add the setup code to both ~/.zshrc and ~/.zprofile
cat << 'EOF' >> ~/.zshrc
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - zsh)"
EOF

cat << 'EOF' >> ~/.zprofile
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - zsh)"
EOF

# Use the shell and verify python version
pyenv shell 3.10
python --version # This should return "Python 3.10.19"


python -m venv venv # Creates a virtual environment named 'venv'
source venv/bin/activate
which python # Should return "/Users/[your_user]/[something]/PyGPT/venv/bin/python"

pip install -r requirements.txt

"""
Useful global definitions
"""
import os

# Directories
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")  # type: str # Project root directory
CONFIG_DIR = os.path.join(ROOT_DIR, "config")  # type: str # Config file directory
DATA_DIR = os.path.join(ROOT_DIR, "public")  # type: str # Data directory
CODE_DIR = os.path.join(ROOT_DIR, "code")  # type: str # Data directory

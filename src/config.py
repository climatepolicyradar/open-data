"""Override any of these settings by adding environment variables with the same name."""

from pathlib import Path
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

REPO_NAME = os.getenv("REPO_NAME", "ClimatePolicyRadar/all-document-text-data")
REPO_URL = f"https://huggingface.co/datasets/{REPO_NAME}"
CACHE_DIR = os.getenv("CACHE_DIR", Path(__file__).parent / "../cache")

DATA_REVISION = os.getenv(
    "DATA_REVISION", "main"
)  # Use this to set a commit hash. Recommended!

from typing import Optional
from huggingface_hub import snapshot_download

REPO_NAME = "ClimatePolicyRadar/all-document-text-data-weekly"
REPO_URL = f"https://huggingface.co/datasets/{REPO_NAME}"
CACHE_DIR = "../cache"

# INTERNAL NOTE: use this commit hash until the weekly pipeline run that produces the data
# is stable.
REVISION = "bd0abf24ae34d3150bdd8ac66f36a28e47f3ee93"  # Use this to set a commit hash. Recommended!


def download_data(cache_dir: str, revision: Optional[str] = None) -> None:
    """
    Download the data to the cache directory.

    :param cache_dir: the directory to save the data to
    :param revision: optional commit hash, defaults to None
    """
    snapshot_download(
        repo_id=REPO_NAME,
        repo_type="dataset",
        local_dir=cache_dir,
        revision=revision,
        allow_patterns=["*.parquet"],
    )

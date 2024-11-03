from typing import Optional
from huggingface_hub import snapshot_download

from src.config import REPO_NAME


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

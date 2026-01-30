import os
import zipfile
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
from nanoRecSys.config import settings


def download_and_extract():
    """Download MovieLens 20M dataset if not already present."""
    url = settings.ml_20m_url
    zip_path = settings.data_dir / "ml-20m.zip"

    # Check if data already exists (e.g. ratings.csv)
    ratings_path = settings.raw_data_dir / "ratings.csv"
    if ratings_path.exists():
        print(f"Data already exists at {settings.raw_data_dir}")
        return

    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)  # type: ignore
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)

    with open(zip_path, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(settings.data_dir)

    # Clean up zip file
    os.remove(zip_path)
    print("Download and extraction complete.")


def process_data(min_interactions=None):
    """
    Load raw data, encode IDs, and save processed files.

    Args:
        min_interactions (int, optional): If set, perform k-core filtering.
                                          Recursively remove users/items with fewer than k interactions.
    """
    ratings_path = settings.raw_data_dir / "ratings.csv"
    movies_path = settings.raw_data_dir / "movies.csv"

    if not ratings_path.exists():
        download_and_extract()

    print("Loading ratings.csv...")
    df = pd.read_csv(
        ratings_path,
        dtype={"userId": int, "movieId": int, "rating": float, "timestamp": int},
    )

    print(f"Raw interactions: {len(df)}")

    # Optional: k-core filtering
    if min_interactions is not None:
        print(f"Applying {min_interactions}-core filtering (recursive)...")
        while True:
            start_len = len(df)

            # Filter users
            user_counts = df["userId"].value_counts()
            valid_users = user_counts[user_counts >= min_interactions].index
            df = df[df["userId"].isin(valid_users)]

            # Filter items
            item_counts = df["movieId"].value_counts()
            valid_items = item_counts[item_counts >= min_interactions].index
            df = df[df["movieId"].isin(valid_items)]

            if len(df) == start_len:
                break

            print(f"Reduced to {len(df)} interactions...")

        print(f"Final interactions after {min_interactions}-core filtering: {len(df)}")

    # 1. Encode Users
    unique_users = df["userId"].unique()
    user2id = {u: i for i, u in enumerate(unique_users)}
    df["user_idx"] = df["userId"].map(user2id)

    # 2. Encode Items
    # Load movies to ensure we cover all movies, not just those in ratings (though ML-20m is dense)
    print("Loading movies.csv...")
    movies_df = pd.read_csv(movies_path)
    unique_movies = movies_df["movieId"].unique()
    movie2id = {m: i for i, m in enumerate(unique_movies)}

    # Map movies in ratings
    df = df[df["movieId"].isin(unique_movies)]  # filter valid movies
    df["item_idx"] = df["movieId"].map(movie2id)

    print(f"Num Users: {len(unique_users)}")
    print(f"Num Items: {len(unique_movies)}")
    print(f"Num Interactions: {len(df)}")

    # Save processed
    print("Saving processed data...")
    df.to_parquet(settings.processed_data_dir / "interactions.parquet", index=False)

    np.save(settings.processed_data_dir / "user_map.npy", unique_users)
    np.save(settings.processed_data_dir / "item_map.npy", unique_movies)

    # Create sparse matrix or simple index for popularity baseline later

    print("Data processing complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build/Process MovieLens Dataset")
    parser.add_argument(
        "--k_core",
        type=int,
        default=None,
        help="Apply recursive k-core filtering (remove users/items with < k interactions). Default: None (no filtering).",
    )
    args = parser.parse_args()

    process_data(min_interactions=args.k_core)

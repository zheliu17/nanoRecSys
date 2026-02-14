# Copyright (c) 2026 Zhe Liu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Small CLI utility to show a user's ground-truth items (from test set)
and recent watch history (up to 20 most recent).

Usage:
  python scripts/show_user_truth.py --user-id 3
"""

import argparse
import pickle

import numpy as np
import pandas as pd

from nanoRecSys.config import settings


class ShowUserHelper:
    """Helper that loads data once and provides methods for notebooks or CLI.

    Usage in notebook:
        helper = ShowUserHelper()
        helper.load()
        helper.get_ground_truth(3)
        helper.get_recent_history(3, n=20)
    """

    def __init__(self):
        self.movies_df = None
        self.item_map = None
        self.test_user_groups = None
        self.user_id_to_idx = None
        self.user_history = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return

        movies_path = settings.raw_data_dir / "movies.csv"
        movies_df = pd.read_csv(movies_path)
        self.movies_df = movies_df.drop_duplicates("movieId").set_index("movieId")

        self.item_map = np.load(settings.processed_data_dir / "item_map.npy")

        test_df = pd.read_parquet(settings.processed_data_dir / "test.parquet")
        self.test_user_groups = test_df.groupby("user_idx")["item_idx"].apply(list)

        user_map = np.load(settings.processed_data_dir / "user_map.npy")
        self.user_id_to_idx = {uid: i for i, uid in enumerate(user_map)}

        history_cache_path = settings.artifacts_dir / "user_history_cache.pkl"
        with open(history_cache_path, "rb") as f:
            self.user_history = pickle.load(f)

        self._loaded = True

    def _ensure_loaded(self):
        if not self._loaded:
            self.load()

    def get_user_idx(self, user_id: int):
        self._ensure_loaded()
        if user_id not in self.user_id_to_idx:  # type: ignore
            raise KeyError(f"User id {user_id} not found in user map")
        return self.user_id_to_idx[user_id]  # type: ignore

    def get_ground_truth(self, user_id: int):
        """Return a DataFrame of ground-truth movies for the user (or None)."""
        self._ensure_loaded()
        user_idx = self.user_id_to_idx.get(user_id)  # type: ignore
        if user_idx is None:
            return None
        if user_idx in self.test_user_groups.index:  # type: ignore
            gt_item_idxs = self.test_user_groups.loc[user_idx]  # type: ignore
            gt_movie_ids = self.item_map[gt_item_idxs]  # type: ignore
            return self.movies_df.loc[gt_movie_ids]  # type: ignore
        return None

    def get_recent_history(self, user_id: int, n: int = 5):
        """Return a DataFrame of up to `n` most recent watched movies (most recent first)."""
        self._ensure_loaded()
        user_idx = self.user_id_to_idx.get(user_id)  # type: ignore
        if user_idx is None:
            return None

        if isinstance(self.user_history, dict):
            hist = self.user_history.get(user_idx, [])
        else:
            try:
                hist = self.user_history[user_idx]  # type: ignore
            except Exception:
                hist = []

        if not hist:
            return None

        recent_idxs = hist[::-1][:n]
        recent_movie_ids = self.item_map[recent_idxs]  # type: ignore
        return self.movies_df.loc[recent_movie_ids]  # type: ignore

    def show_user(self, user_id: int, n: int = 5):
        """Prints the ground-truth and recent history to stdout (CLI friendly)."""
        try:
            user_idx = self.get_user_idx(user_id)
        except KeyError as e:
            print(e)
            return

        print(f"User id: {user_id}  ->  idx: {user_idx}\n")

        gt = self.get_ground_truth(user_id)
        if gt is not None and not gt.empty:
            print("Ground-truth (test set) movies:")
            print(gt.to_string())
        else:
            print("No ground-truth items for this user in test set.")

        recent = self.get_recent_history(user_id, n=n)
        if recent is not None and not recent.empty:
            print("\nRecent watched (most recent first, up to {0}):".format(n))
            print(recent.to_string())
        else:
            print("\nNo history available for this user.")


def main():
    parser = argparse.ArgumentParser(
        description="Show user's test ground-truth and recent history"
    )
    parser.add_argument(
        "--user-id",
        "-u",
        required=True,
        type=int,
        help="Raw user id (as in raw dataset)",
    )
    args = parser.parse_args()

    helper = ShowUserHelper()
    helper.load()
    helper.show_user(args.user_id)


if __name__ == "__main__":
    main()

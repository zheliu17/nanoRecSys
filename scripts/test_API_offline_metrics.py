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

import random

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from nanoRecSys.config import settings

PROJECT_ROOT = settings.project_root
API_URL = "http://127.0.0.1:8000/recommend"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
TEST_FILE = DATA_DIR / "test.parquet"
USER_MAP_FILE = DATA_DIR / "user_map.npy"
ITEM_MAP_FILE = DATA_DIR / "item_map.npy"
POSITIVE_THRESHOLD = 0.0
K_LIST = [10, 50, 100]
SAMPLE_SIZE = 1000


def load_data():
    user_map = np.load(USER_MAP_FILE)
    item_map = np.load(ITEM_MAP_FILE)

    df = pd.read_parquet(TEST_FILE)
    df = df[df["rating"] >= POSITIVE_THRESHOLD]
    user_groups = df.groupby("user_idx")["item_idx"].apply(list)
    return user_groups, user_map, item_map


def calculate_metrics(user_groups, user_map, item_map):
    # Sample users
    all_users = user_groups.index.tolist()
    if len(all_users) > SAMPLE_SIZE:
        sampled_users_idx = random.sample(all_users, SAMPLE_SIZE)
    else:
        sampled_users_idx = all_users

    hits = {k: 0 for k in K_LIST}
    total_users = 0

    for u_idx in tqdm(sampled_users_idx, desc="Evaluating"):
        user_id = int(user_map[u_idx])

        true_item_indices = user_groups[u_idx]
        true_item_ids = set(item_map[i] for i in true_item_indices)

        if not true_item_ids:
            continue

        try:
            response = requests.post(
                API_URL,
                json={
                    "user_id": user_id,
                    "k": max(K_LIST),
                    "explain": False,
                    "include_history": False,
                },
                timeout=2,
            )

            if response.status_code != 200:
                continue

            data = response.json()
            recs = data.get("movie_ids", [])

            for k in K_LIST:
                top_k_recs = set(recs[:k])
                if not true_item_ids.isdisjoint(top_k_recs):
                    hits[k] += 1

            total_users += 1

        except Exception:
            continue

    print("\nResults:")
    for k in K_LIST:
        hr = hits[k] / total_users if total_users > 0 else 0
        print(f"HR@{k}: {hr:.4f}")


if __name__ == "__main__":
    user_groups, user_map, item_map = load_data()
    calculate_metrics(user_groups, user_map, item_map)

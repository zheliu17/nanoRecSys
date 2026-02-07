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

import argparse
from pathlib import Path

import faiss
import numpy as np

from nanoRecSys.config import settings


def build_flat_index(embeddings_path=None, output_path=None):
    print("Alignment check: Loading item embeddings for Flat Index...")
    if embeddings_path is None:
        embeddings_path = settings.artifacts_dir / "item_embeddings.npy"
    else:
        embeddings_path = Path(embeddings_path)

    if output_path is None:
        output_path = settings.artifacts_dir / "faiss_flat.index"
    else:
        output_path = Path(output_path)

    if not embeddings_path.exists():
        print(f"Error: Embeddings not found at {embeddings_path}")
        return

    embeddings = np.load(embeddings_path).astype("float32")
    faiss.normalize_L2(embeddings)

    d = embeddings.shape[1]
    print(f"Loaded {len(embeddings)} embeddings of dimension {d}")

    index = faiss.IndexFlatIP(d)
    index.add(embeddings)  # type: ignore

    print(f"Added {index.ntotal} vectors to index.")

    print(f"Saving index to {output_path}")
    faiss.write_index(index, str(output_path))
    print("Flat index built successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a Faiss Flat index from embeddings"
    )
    parser.add_argument(
        "--embeddings", type=str, default=None, help="Path to item_embeddings.npy"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Path to save the built index"
    )
    args = parser.parse_args()

    build_flat_index(embeddings_path=args.embeddings, output_path=args.output)

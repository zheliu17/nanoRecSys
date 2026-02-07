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


def build_ivfpq_index(nlist=128, m=8, nbits=8, embeddings_path=None, output_path=None):
    print("Alignment check: Loading item embeddings for IVF-PQ Index...")
    if embeddings_path is None:
        embeddings_path = settings.artifacts_dir / "item_embeddings.npy"
    else:
        embeddings_path = Path(embeddings_path)

    if output_path is None:
        output_path = settings.artifacts_dir / "faiss_ivfpq.index"
    else:
        output_path = Path(output_path)

    if not embeddings_path.exists():
        print(f"Error: Embeddings not found at {embeddings_path}")
        return

    embeddings = np.load(embeddings_path).astype("float32")
    faiss.normalize_L2(embeddings)

    d = embeddings.shape[1]
    print(f"Loaded {len(embeddings)} embeddings of dimension {d}")

    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits, faiss.METRIC_L2)

    print("Training index (this may take a moment)...")
    index.train(embeddings)  # type: ignore

    print("Adding embeddings...")
    index.add(embeddings)  # type: ignore

    print(f"Saving index to {output_path}")
    faiss.write_index(index, str(output_path))
    print("IVF-PQ index built successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a Faiss IVF-PQ index from embeddings"
    )
    parser.add_argument("--nlist", type=int, default=128, help="Number of IVF lists")
    parser.add_argument("--m", type=int, default=8, help="Number of PQ subquantizers")
    parser.add_argument("--nbits", type=int, default=8, help="Bits per subquantizer")
    parser.add_argument(
        "--embeddings", type=str, default=None, help="Path to item_embeddings.npy"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Path to save the built index"
    )
    args = parser.parse_args()

    build_ivfpq_index(
        nlist=args.nlist,
        m=args.m,
        nbits=args.nbits,
        embeddings_path=args.embeddings,
        output_path=args.output,
    )

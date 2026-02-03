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

import logging
import faiss
from pathlib import Path
import torch

logger = logging.getLogger(__name__)


class FaissStore:
    def __init__(self, artifacts_dir: Path, index_type="ivf"):  # ivf or flat
        self.artifacts_dir = artifacts_dir
        self.index = None

        index_name = "faiss_ivfpq.index" if index_type == "ivf" else "faiss_flat.index"
        path = artifacts_dir / index_name

        if not path.exists():
            logger.warning(f"Index {index_name} not found. Trying flat.")
            path = artifacts_dir / "faiss_flat.index"

            if not path.exists():
                logger.error("No FAISS index found!")
                return

        logger.info(f"Loading FAISS index from {path}...")
        self.index = faiss.read_index(str(path))

    def search(self, user_emb: torch.Tensor, k: int = 100, nprobe: int = 8):
        if self.index is None:
            return [], []

        if isinstance(user_emb, torch.Tensor):
            user_emb = user_emb.detach().cpu().numpy()  # type: ignore

        if hasattr(self.index, "nprobe"):
            self.index.nprobe = nprobe

        distances, indices = self.index.search(user_emb, k)
        return indices[0], distances[0]

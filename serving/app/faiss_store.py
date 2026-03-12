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
from pathlib import Path

import faiss
import torch

logger = logging.getLogger(__name__)


class FaissStore:
    def __init__(self, artifacts_dir: Path, index_type="flat"):  # ivf or flat
        faiss.omp_set_num_threads(1)

        self.artifacts_dir = artifacts_dir
        self.index = None
        self.requested_index_type = index_type
        self.loaded_index_type = None
        self.index_path = None

        index_name = "faiss_flat.index" if index_type == "flat" else "faiss_ivfpq.index"
        path = artifacts_dir / index_name

        if not path.exists():
            logger.warning("Index %s not found. Trying flat fallback.", index_name)
            path = artifacts_dir / "faiss_flat.index"

            if not path.exists():
                logger.error("No FAISS index found in %s", artifacts_dir)
                return

            self.loaded_index_type = "flat"
        else:
            self.loaded_index_type = index_type

        logger.info("Loading FAISS index from %s...", path)
        self.index = faiss.read_index(str(path))
        self.index_path = str(path)

    def is_ready(self) -> bool:
        return self.index is not None

    def status(self) -> dict:
        return {
            "ready": self.is_ready(),
            "requested_index_type": self.requested_index_type,
            "loaded_index_type": self.loaded_index_type,
            "index_path": self.index_path,
        }

    def search(self, user_emb: torch.Tensor, k: int = 100, nprobe: int = 8):
        if self.index is None:
            return [], []

        if isinstance(user_emb, torch.Tensor):
            user_emb = user_emb.detach().cpu().numpy()  # type: ignore[assignment]

        if hasattr(self.index, "nprobe"):
            self.index.nprobe = nprobe

        distances, indices = self.index.search(user_emb, k)
        return indices[0], distances[0]

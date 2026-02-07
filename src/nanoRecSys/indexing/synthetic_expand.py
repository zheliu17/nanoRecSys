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

import numpy as np
from tqdm import tqdm

from nanoRecSys.config import settings


def expand_embeddings(target_count=5_000_000):
    """
    Creates a larger synthetic dataset of embeddings
    Args:
        target_count: Total number of items desired (default 1M)
        noise_std: Standard deviation of Gaussian noise to add (default 0.01)
    """
    print(f"Targeting {target_count} items with synthetic expansion...")

    input_path = settings.artifacts_dir / "item_embeddings.npy"
    if not input_path.exists():
        print(f"Error: Real embeddings not found at {input_path}")
        return

    # Load original data
    original_embeddings = np.load(input_path)
    current_count, d = original_embeddings.shape
    print(f"Loaded original embeddings: {original_embeddings.shape}")

    # Calculate global statistics for distractor generation (Option 1)
    global_mean = np.mean(original_embeddings, axis=0)
    global_std = np.std(original_embeddings, axis=0)

    if current_count >= target_count:
        print(f"Dataset already has {current_count} items. No expansion needed.")
        return input_path

    needed = target_count - current_count

    # 2. Setup Output File (using Memmap to avoid OOM)
    output_filename = f"synthetic_{target_count // 1000000}M_embeddings.npy"
    output_path = settings.artifacts_dir / output_filename
    print(f"Generating {needed} vectors. Target file: {output_path}")

    # Open a new .npy file in write mode, mapped to memory
    print(f"Creating memory-mapped file for {target_count} items...")
    fp = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=original_embeddings.dtype, shape=(target_count, d)
    )

    # 3. Copy Original Data
    print("Copying original embeddings...")
    fp[:current_count] = original_embeddings[:]
    fp.flush()  # Ensure originals are written

    # 4. Generate & Write Batches
    batch_size = 100_000  # Process 100k vectors at a time (approx 25MB for 64d float32)
    start_idx = current_count

    # Calculate number of batches needed
    total_needed = needed
    num_batches = (total_needed + batch_size - 1) // batch_size

    print(f"Generating synthetic data in {num_batches} batches...")

    for _ in tqdm(range(num_batches), desc="Generating Batches"):
        current_batch_size = min(batch_size, needed)

        # 4a. Distractor Method: Generate independent random vectors matching global stats
        # This creates "distractors" that fill the space but are distinct from real items.
        synthetic_batch = np.random.normal(
            loc=global_mean, scale=global_std, size=(current_batch_size, d)
        )

        # 4c. Normalize
        norms = np.linalg.norm(synthetic_batch, axis=1, keepdims=True)
        # Safe divide
        norms[norms == 0] = 1.0
        synthetic_batch = synthetic_batch / norms

        # 4d. Write to Disk
        end_idx = start_idx + current_batch_size
        fp[start_idx:end_idx] = synthetic_batch.astype(original_embeddings.dtype)

        # Update pointers
        start_idx = end_idx
        needed -= current_batch_size

    # Final flush to ensure integrity
    fp.flush()
    print("Done. File closed.")
    return output_path


if __name__ == "__main__":
    expand_embeddings(target_count=5_000_000)

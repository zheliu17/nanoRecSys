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
Mine negatives and generate embeddings using a trained SASRec (Transformer) model.
This script generates per-interaction embeddings and corresponding training data for the Ranker.
Processes both Train and Val splits and merges them into a single set of artifacts.
"""

import argparse
import struct

import numpy as np
import numpy.lib.format as fmt
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from nanoRecSys.config import settings
from nanoRecSys.data.datasets import SequentialDataset
from nanoRecSys.models.towers import ItemTower, TransformerUserTower
from nanoRecSys.utils.logging_config import get_logger
from nanoRecSys.utils.utils import collate_fn_numpy_to_tensor


def update_npy_header(fp, shape, dtype, header_len):
    """
    Overwrites the beginning of the file with a valid NPY header of fixed length.
    fp must be an open binary file handle at the beginning or seekable.
    """
    fp.seek(0)
    magic = b"\x93NUMPY"
    version = b"\x01\x00"

    header_dict = {
        "descr": fmt.dtype_to_descr(np.dtype(dtype)),
        "fortran_order": False,
        "shape": tuple(shape),
    }
    header_str = str(header_dict)

    # Calculate padding
    # Format: Magic(6) + Ver(2) + Len(2) + Str + Padding + \n
    # Total = header_len
    # Len(2) stores the length of (Str + Padding + \n)

    preamble_size = 10  # 6 + 2 + 2
    body_size = header_len - preamble_size

    current_body_len = len(header_str) + 1  # +1 for \n
    padding_needed = body_size - current_body_len

    if padding_needed < 0:
        raise ValueError(
            f"Header reserved size {header_len} is too small for shape {shape}"
        )

    # Construct final body
    # Standard NPY padding is spaces before the newline
    final_body = header_str + (" " * padding_needed) + "\n"

    fp.write(magic)
    fp.write(version)
    fp.write(struct.pack("<H", body_size))
    fp.write(final_body.encode("ascii"))
    fp.flush()


def process_split(
    split: str,
    model: TransformerUserTower,
    item_embeddings: torch.Tensor,  # (N_items, D) on device
    temp_emb_file,  # File handle for binary writing
    global_offset: int,
    batch_size: int,
    top_k: int,
    skip_top: int,
    num_random_negatives: int,
    sampling_ratio: float,
    device: str,
    n_items: int,
    suffix: str,
):
    logger = get_logger()
    interactions_path = settings.processed_data_dir / f"{split}.parquet"
    dataset = SequentialDataset(str(interactions_path))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=settings.pin_memory,
        collate_fn=collate_fn_numpy_to_tensor,
    )

    logger.info(f"Processing {len(dataset)} sequences for split '{split}'...")

    # Accumulate metadata for DataFrames
    # Lists are fine for metadata (ints), but embeddings are written to disk
    result_user_embedding_idxs = []
    # result_original_user_ids = []  # Optional, can be used for analysis but not needed for training
    result_pos_items = []
    result_hard_negs = []
    result_random_negs = []

    local_count = 0
    current_offset = global_offset

    with torch.inference_mode():
        for batch_seqs, batch_uids in tqdm(dataloader, desc=f"Mining {split}"):
            batch_seqs = batch_seqs.to(device)

            # We strip the last item from input, as it is only a target.
            # batch_seqs: (B, L+1) -> input: (B, L)
            current_embs = model(batch_seqs[:, :-1])  # (B, L, D)

            # Build next-item targets matrix: default is all next-items (B, L)
            next_items = batch_seqs[:, 1:]

            # We generate user embeddings for all positions
            # File size can be huge (# of interactions x embedding_dim)
            if split == "train" and sampling_ratio is not None and sampling_ratio < 1.0:
                # Create a random mask over positions and apply only to non-padding tokens
                rand = torch.rand(next_items.shape, device=next_items.device)
                keep = (rand < float(sampling_ratio)) & (next_items != 0)
                tmp = torch.zeros_like(next_items)
                tmp[keep] = next_items[keep]
                next_items = tmp

            # For validation only keep the final interaction as the target
            # by zeroing out other positions so the boolean mask selects only the last position per sequence.
            if split == "val":
                tmp = torch.zeros_like(next_items)
                tmp[:, -1] = batch_seqs[:, -1]
                next_items = tmp

            valid_mask = next_items != 0

            # Filter Valid
            # valid_indices = torch.nonzero(valid_mask, as_tuple=True)
            flat_embs = current_embs[valid_mask]  # (N_valid, D)
            flat_targets = next_items[valid_mask]
            flat_targets_0indices = flat_targets - 1  # Convert to 0-based index
            # flat_uids = batch_uids[valid_indices[0].cpu()]

            if flat_embs.shape[0] == 0:
                continue

            # Write Embeddings to Temp File immediately
            flat_embs_cpu = flat_embs.cpu().numpy().astype(np.float32)
            flat_embs_cpu.tofile(temp_emb_file)

            # Update counts
            n_batch = flat_embs.shape[0]
            local_count += n_batch

            # Assign Indices
            indices = np.arange(current_offset, current_offset + n_batch)
            current_offset += n_batch

            # Mining Hard Negatives
            scores = torch.matmul(flat_embs, item_embeddings.T)
            top_scores, top_indices = torch.topk(scores, k=top_k + 1, dim=1)
            top_indices = top_indices.cpu().numpy()
            flat_targets_cpu = flat_targets_0indices.cpu().numpy()

            batch_hard = []
            for i in range(len(flat_targets_cpu)):
                target = flat_targets_cpu[i]
                cands = top_indices[i]
                cands = cands[cands != target]

                if len(cands) > skip_top:
                    pool = cands[skip_top:]
                else:
                    pool = cands

                if len(pool) > 0:
                    chosen = np.random.choice(pool)
                else:
                    # Fallback
                    while True:
                        chosen = np.random.randint(0, n_items)
                        if chosen != target:
                            break
                batch_hard.append(chosen)

            # Mining Random Negatives
            if num_random_negatives > 0:
                rand_cands = np.random.randint(
                    0, n_items, size=(n_batch, num_random_negatives)
                )
                # Quick collision check/fix
                t_exp = flat_targets_cpu[:, None]
                mask = rand_cands == t_exp
                if mask.any():
                    rows, cols = np.where(mask)
                    for r, c in zip(rows, cols):
                        while True:
                            new_val = np.random.randint(0, n_items)
                            if new_val != flat_targets_cpu[r]:
                                rand_cands[r, c] = new_val
                                break
                result_random_negs.append(rand_cands)

            # Store Metadata
            result_user_embedding_idxs.extend(indices)
            # result_original_user_ids.extend(flat_uids.cpu().numpy())
            result_pos_items.extend(flat_targets_cpu)
            result_hard_negs.extend(batch_hard)

    # Save Partial DataFrames for this split
    base_name = f"{split}_{suffix}"

    # Interactions
    # Transformer is trained on (positive or all) next-item prediction
    # Pair with low ratings is not implemented
    df_interactions = pd.DataFrame(
        {
            "user_idx": result_user_embedding_idxs,
            "item_idx": result_pos_items,
            "rating": 5.0,  # Dummy; This effectively treats all interactions as positive for the Ranker
        }
    )

    if split == "val":
        df_interactions = df_interactions.sample(frac=1, random_state=42).reset_index(
            drop=True
        )

    path_int = settings.processed_data_dir / f"{base_name}.parquet"
    df_interactions.to_parquet(path_int)

    # Hard Neg
    df_hard = pd.DataFrame(
        {"user_idx": result_user_embedding_idxs, "neg_item_idx": result_hard_negs}
    )
    path_hard = settings.processed_data_dir / f"{base_name}_negatives_hard.parquet"
    df_hard.to_parquet(path_hard)

    # Random Neg
    if num_random_negatives > 0 and len(result_random_negs) > 0:
        full_random = np.concatenate(result_random_negs, axis=0)  # (count, num_neg)
        r_cols = {"user_idx": result_user_embedding_idxs}
        for i in range(num_random_negatives):
            r_cols[f"neg_item_idx_{i + 1}"] = full_random[:, i]  # type: ignore
        df_rand = pd.DataFrame(r_cols)
        path_rand = (
            settings.processed_data_dir / f"{base_name}_negatives_random.parquet"
        )
        df_rand.to_parquet(path_rand)

    logger.info(f"Finished {split}: {local_count} interactions.")
    return local_count


def run_pipeline(
    top_k: int = 100,
    skip_top: int = 20,
    batch_size: int = 256,
    num_random_negatives: int = 2,
    suffix: str = "sasrec_combined",
    sampling_ratio: float = 1.0,
):
    logger = get_logger()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Item Embeddings
    # It can be recomputed if needed, but we require it here anyway
    item_emb_path = settings.artifacts_dir / "item_embeddings.npy"
    if not item_emb_path.exists():
        logger.error(f"Item embeddings not found at {item_emb_path}")
        logger.error(
            "Please generate embeddings first using: python src/indexing/build_embeddings.py --mode items"
        )
        return
    logger.info("Loading item embeddings...")
    item_embeddings = np.load(item_emb_path)
    n_items, embedding_dim = item_embeddings.shape
    item_embeddings = torch.from_numpy(item_embeddings).to(device)

    # 2. Load Retrieval Model (User Tower)
    dummy_item_tower = ItemTower(
        vocab_size=n_items,
        embed_dim=settings.embed_dim,
        output_dim=settings.tower_out_dim,
        hidden_dims=settings.towers_hidden_dims,
        use_projection=settings.use_projection,
    )

    model = TransformerUserTower(
        vocab_size=n_items,
        embed_dim=settings.embed_dim,
        output_dim=settings.tower_out_dim,
        max_seq_len=settings.max_seq_len,
        n_heads=settings.transformer_heads,
        n_layers=settings.transformer_layers,
        dropout=settings.transformer_dropout,
        swiglu_hidden_dim=settings.swiglu_hidden_dim,
        shared_embedding=dummy_item_tower,
    ).to(device)

    model_path = settings.artifacts_dir / "user_tower.pth"
    logger.info(f"Loading retrieval model from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()
    model.to(device)

    # 3. Open Final File Directly (Write Mode)
    # We write a placeholder header first, then append data, then update header at the end.
    final_npy_path = (
        settings.sasrec_user_embs_npy_path / f"user_embeddings_{suffix}.npy"
    )
    HEADER_SIZE = 4096  # Reserve 4KB for header (plenty for simple shapes)

    logger.info(f"Opening final file {final_npy_path} for direct writing...")
    try:
        final_fp = open(final_npy_path, "wb")
        # Write placeholder header (spaces)
        final_fp.write(b" " * HEADER_SIZE)
    except OSError as e:
        logger.error(f"Could not open final file {final_npy_path}: {e}")
        return

    global_offset = 0

    # 4. Process Splits
    for split in ["train", "val"]:
        count = process_split(
            split=split,
            model=model,
            item_embeddings=item_embeddings,
            temp_emb_file=final_fp,  # Write directly to final file
            global_offset=global_offset,
            batch_size=batch_size,
            top_k=top_k,
            skip_top=skip_top,
            num_random_negatives=num_random_negatives,
            sampling_ratio=sampling_ratio,
            device=device,
            n_items=n_items,
            suffix=suffix,
        )
        global_offset += count

    # 5. Finalize NPY Header
    logger.info(f"Finalizing NPY file (Shape: {global_offset}x{embedding_dim})...")
    update_npy_header(
        final_fp,
        shape=(global_offset, embedding_dim),
        dtype=np.float32,
        header_len=HEADER_SIZE,
    )
    final_fp.close()

    logger.info("Pipeline Complete. Artifacts ready for Ranker.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--top_k", type=int)
    parser.add_argument("--skip_top", type=int)
    parser.add_argument("--sampling_ratio", type=float, default=1.0)
    args = parser.parse_args()

    run_pipeline(
        batch_size=args.batch_size, top_k=args.top_k, sampling_ratio=args.sampling_ratio
    )

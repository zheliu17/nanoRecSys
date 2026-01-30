import numpy as np
import faiss
from nanoRecSys.config import settings


def build_flat_index():
    print("Alignment check: Loading item embeddings for Flat Index...")
    embeddings_path = settings.artifacts_dir / "item_embeddings.npy"
    output_path = settings.artifacts_dir / "faiss_flat.index"

    if not embeddings_path.exists():
        print(f"Error: Embeddings not found at {embeddings_path}")
        return

    embeddings = np.load(embeddings_path).astype("float32")
    d = embeddings.shape[1]
    print(f"Loaded {len(embeddings)} embeddings of dimension {d}")

    # Build IndexFlatIP (Inner Product) - exact search
    # Assuming embeddings are compatible with Dot Product (user tower output @ item tower output)
    # Using IndexFlatIP for exact Maximum Inner Product Search (MIPS)
    print(f"Building Flat IP index for {len(embeddings)} items...")
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)  # type: ignore

    print(f"Added {index.ntotal} vectors to index.")

    print(f"Saving index to {output_path}")
    faiss.write_index(index, str(output_path))
    print("Flat index built successfully.")


if __name__ == "__main__":
    build_flat_index()

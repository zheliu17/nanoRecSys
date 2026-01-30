import numpy as np
import faiss
from nanoRecSys.config import settings


def build_ivfpq_index(nlist=128, m=8, nbits=8):
    print("Alignment check: Loading item embeddings for IVF-PQ Index...")
    embeddings_path = settings.artifacts_dir / "item_embeddings.npy"
    output_path = settings.artifacts_dir / "faiss_ivfpq.index"

    if not embeddings_path.exists():
        print(f"Error: Embeddings not found at {embeddings_path}")
        return

    embeddings = np.load(embeddings_path).astype("float32")
    d = embeddings.shape[1]
    print(f"Loaded {len(embeddings)} embeddings of dimension {d}")

    # Configuration for IVF-PQ
    # IVF{nlist}: IndexIVFFlat with nlist centroids (inverted file)
    # PQ{m}: Product Quantizer with m sub-vectors
    # nbits is usually 8 by default in index_factory strings

    # We use index_factory to easily create complex indices
    # "IVF{nlist},PQ{m}"
    # METRIC_INNER_PRODUCT is critical for Dot Product search

    factory_string = f"IVF{nlist},PQ{m}"
    print(
        f"Building index with factory string: {factory_string} and metric METRIC_INNER_PRODUCT"
    )

    index = faiss.index_factory(d, factory_string, faiss.METRIC_INNER_PRODUCT)

    print("Training index (this may take a moment)...")
    index.train(embeddings)

    print("Adding embeddings...")
    index.add(embeddings)

    print(f"Saving index to {output_path}")
    faiss.write_index(index, str(output_path))
    print("IVF-PQ index built successfully.")


if __name__ == "__main__":
    # Default values as per plan MVP.
    # nlist=128, m=8 are reasonable start points for ~27k items.
    # We use 128 clusters which is ~210 items per cluster.
    build_ivfpq_index(nlist=128, m=8)

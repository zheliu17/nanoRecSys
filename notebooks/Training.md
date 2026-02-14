# Training Details

## Retriever: Sequential Transformer

The core of the retrieval system is a **Sequential Transformer** optimized for next-item prediction. It is inspired by **SASRec**, with several modern architectural and training choices that are common in recent (2023–2025) deep learning stacks.

### Architecture

This implementation follows a "modern recsys stack" variant:

* RoPE (Rotary Positional Embeddings)
* SwiGLU Activation
* RMSNorm, Pre-LN
* InfoNCE Loss, in-batch negatives

### Dataset & Evaluation

* **Dataset:** MovieLens-20M.
* **Splitting Strategy:** Leave-One-One (LOO). The last interaction for each user is held out for testing.
* **Metrics:** HitRate@10 (HR@10) and NDCG@10. We rank the ground-truth item against **all** other items (full ranking), excluding items the user has already interacted with.

### Results

Comparing our implementation against reported results in recent literature:

| Method | Source | HitRate@10 | NDCG@10 |
| :--- | :--- | :--- | :--- |
| **Ours** | | **0.2857** | **0.1622** |
| SASRec (Vanilla BCE) | Klenitskiy et al. '23[^4] | 0.2001 | 0.1067 |
| SASRec (Cross-Entropy) | Klenitskiy et al. '23[^4] | 0.2983 | 0.1833 |
| SASRec | Zhai et al. '24[^1] | 0.2906 | 0.1621 |
| BERT4Rec | Zhai et al. '24[^1] | 0.2816 | 0.1703 |
| **HSTU (SOTA)** | Zhai et al. '24[^1] | 0.3252 | 0.1878 |
| SASRec | Ye et al. '25 [^2] | 0.2781 | 0.1553 |
| **FuXi-𝛼 (SOTA)** | Ye et al. '25 [^2] | 0.3353 | 0.1954 |
| eSASRec | Tikhonovich et al. '25 [^3] | 0.3130 | 0.1770 |

[^1]: Zhai, Jiaqi, et al. "Actions speak louder than words: Trillion-parameter sequential transducers for generative recommendations." ICML 2024. <https://arxiv.org/abs/2402.17152>
[^2]: Ye, Yufei, et al. "FuXi-α: Scaling Recommendation Model with Feature Interaction Enhanced Transformer." WWW Companion 2025. <https://arxiv.org/abs/2502.03036>
[^3]: Tikhonovich, Daria, et al. "eSASRec: Enhancing Transformer-based Recommendations in a Modular Fashion." RecSys 2025. <https://arxiv.org/abs/2508.06450>
[^4]: Klenitskiy, Anton, and Alexey Vasilev. "Turning dross into gold loss: is bert4rec really better than sasrec?." RecSys 2023. <https://arxiv.org/abs/2309.07602>
[^5]: Zhai, Jiaqi, et al. "Revisiting neural retrieval on accelerators."  ACM SIGKDD 2023. <https://arxiv.org/abs/2306.04039>

### Hyperparameters & Design Choices

Below are the configurations selected based on **our experiments** and their alignment recent literature:

1. **RoPE:** Our experiments suggest RoPE improves performance over learnable absolute embeddings.
2. **Embedding Dimension:** 256. Found to be optimal in our tests (consistent with [^1][^2][^3][^4][^5]).
3. **Model Depth:** 4 Layers, 8 Heads. Our experiments confirm that this larger capacity (vs original SASRec's 2 layers) improves performance (also see [^3][^5]).
4. **Dropout:** 0.1 applied to both attention and feedforward layers. A smaller dropout slightly improves performance in our runs.
5. **SwiGLU:** Experiments show SwiGLU requires a larger expansion factor (4x) to perform well (also see [^3]).
6. **Negatives:** In-batch negatives performs similar to the Sampled Softmax approach[^4] with a large number of negatives (3000). Sampled 256 negatives are not optimal.
7. **Loss Function:** InfoNCE with a fixed, low temperature ($ \tau=0.05 $). Our experiments suggest that a low temperature is crucial for performance when using cosine similarity (consistent with [^1][^5]). Decoupled Contrastive Learning (DCL) doesn't show significant improvements in our tests.
8. **Optimizer:** AdamW with $\beta_2=0.98$ and 0 weight decay. We found lower weight decay slightly improves performance.


## Ranker

*A new ranker is in progress*

The second stage is a re-ranking model trained to refine the candidates retrieved by the Transformer.

**Current Architecture:**

* **Type:** MLP Ranker (Interaction-aware).
* **Inputs:** User/Item ID Embeddings, Element-wise interaction product, Genre/Year embeddings, Normalized popularity, and an `IsUnknown` flag (for cold-start handling).
* **Training:** Trained on hard negatives mined from the Transformer retriever.

We utilize ID Dropout (probability to zero-out ID embeddings) during training. This helps the Ranker generalize to new items where ID embeddings are not yet well-learned.

The ranker provides reasonable performance. Even without item IDs (simulating cold start), it captures signal from metadata (Genre/Year/Popularity).

| Method | HitRate@10 | NDCG@10 |
| :--- | :--- | :--- |
| Ours (Retriever) | 0.2857 | 0.1622 |
| Ranker (ID Masked) | 0.1538 | 0.0824 |
| Popularity Baseline | 0.0513 | 0.0255 |

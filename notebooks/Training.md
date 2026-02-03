## Current Architecture
* **Retriever**
    * ID Embeddings + MLPs for User & Item
    * InfoNCE Loss, in batch negatives, mask collisions (for same user in batch)
    * LogQ correction for popularity bias
* **Ranker**
    * Inputs: User/Item ID Embeddings, Element-wise Interactions (element-wise product), Genre/Year embeddings, Normalized popularity, IsUnknown Flag (for new items)
    * Hard Negative Mining from Retriever stage
    * ID dropout (probability to replace ID embedding with zero vector, and change IsUnknown flag)
    * BCE Loss[^1]

## `production_training.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zheliu17/nanoRecSys/blob/main/notebooks/production_training.ipynb)
Training (Retriever & Ranker) and evaluation can be run in Colab. ~20 minutes on T4 GPU.

We split data temporally: train and val on all but last 5 interactions, test on last 5 interactions per user. We only use interactions with ratings >=3.5 as positive samples.

We use this model for production deployment.

Sample results (MovieLens-20M dataset)[^2][^3]:

### Popularity Baseline
| K | HitRate | Recall | NDCG | MRR |
|---|---|---|---|---|
| 10 | 0.111 | 0.038 | 0.026 | 0.038 |
| 20 | 0.178 | 0.068 | 0.037 | 0.042 |
| 50 | 0.295 | 0.128 | 0.055 | 0.046 |
| 100 | 0.414 | 0.202 | 0.072 | 0.047 |

### Retriever (Two-Tower)
| K | HitRate | Recall | NDCG | MRR |
|---|---|---|---|---|
| 10 | 0.116 | 0.039 | 0.026 | 0.038 |
| 20 | 0.198 | 0.075 | 0.040 | 0.043 |
| 50 | 0.374 | 0.178 | 0.069 | 0.048 |
| 100 | 0.544 | 0.304 | 0.098 | 0.051 |

### Ranker (Cross-Encoder)
| K | HitRate | Recall | NDCG | MRR |
|---|---|---|---|---|
| 10 | 0.136 | 0.047 | 0.032 | 0.045 |
| 20 | 0.226 | 0.085 | 0.046 | 0.051 |
| 50 | 0.400 | 0.182 | 0.073 | 0.056 |
| 100 | 0.544 | 0.304 | 0.102 | 0.059 |

## `academic_training_LOO.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zheliu17/nanoRecSys/blob/main/notebooks/academic_training_LOO.ipynb)
Training and evaluation under academic setting can be run in Colab. ~15 minutes on T4 GPU.

Treat all interactions as positive samples. For each user, hold out the last interaction as test, second last as validation, rest for training. Results can be directly compared to literature[^4].

Note this is sampled evaluation, not full ranking. Results vary slightly due to different random seeds.

|        | HR@10  | NDCG@10 | MRR (Global)   |
|--------|--------|---------|--------|
| Popularity| 0.1426 |  0.0717 | 0.0722 |
| NeuCF    | 0.2922 | 0.1271  | 0.1072 |
| **Ours**   | **0.3788** |  **0.1915** | **0.1587** |


[^1]: We experimented with Focal Loss and BPR Loss as well, but did not see significant improvements with the current architecture and dataset.
[^2]: Retriever are trained with logQ correction, while we still use pure cosine similarity for retrieval scoring during evaluation and hard negative mining.
[^3]: User last 5 interactions may not all be in test set as we filter out based on rating threshold.
[^4]: Numbers are taken from https://arxiv.org/abs/1904.06690; Neural Collaborative Filtering (NCF) https://dl.acm.org/doi/abs/10.1145/3038912.3052569

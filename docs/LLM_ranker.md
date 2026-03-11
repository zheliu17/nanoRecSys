# Multimodal LLM for Candidate Re-ranking

This document details the architecture, training dynamics, engineering optimizations, and empirical ablations of the custom multimodal LLM ranker implemented in this repository. The model is designed to efficiently re-rank the top 100 candidates surfaced by the first-stage retriever.

## Methodology [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zheliu17/nanoRecSys/blob/main/docs/LLM_training.ipynb)

Unlike approaches such as LLaRA[^1], which frames the task as a multiple-choice selection from a small pool, this ranker treats the problem as a pointwise probability scoring task.
[^1]: Liao, Jiayi, et al. "Llara: Large language-recommendation assistant." ACM SIGIR 2024. <https://arxiv.org/abs/2312.02445>

* **Base Model**: `unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit`
* **Input Structure**: System instruction + the user's viewing history (top 10 most recently watched movie titles) + candidate movie.
* **Multimodal Injection**: The text prompt is fused directly with collaborative filtering item embeddings via a trained MLP projection layer.
* **Objective**: The model is asked whether a user will watch the candidate movie. Final candidate ranking scores are calculated using the difference in output probabilities/logits between the "Yes" and "No" tokens.

## Evaluation Benchmarks

The table below compares the fine-tuned local LLM ranker against the first-stage sequential retriever, a zero-shot local baseline, and a state-of-the-art API baseline. Evaluation metrics are calculated on the first 500 users in the evaluation set.

| Model / Stage | HitRate@10 | HitRate@50 | HitRate@100 | NDCG@10 | MRR@10 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| First-Stage Retriever (Transformer) | 0.304 | 0.540 | 0.650 | 0.1732 | 0.1338 |
| **Our LLM Ranker** (Fine-tuned Qwen 1.5B) | **0.256** | **0.492** | 0.650 | **0.1413** | **0.1065** |
| LLM API Ranker (Qwen 3.5 Plus, Zero-Shot) | 0.158 | 0.468 | 0.650 | 0.0816 | 0.0584 |
| Local LLM Ranker (Qwen 1.5B, Zero-Shot) | 0.082 | 0.406 | 0.650 | 0.0387 | 0.0259 |

*\* Rankers are evaluated over the top 100 candidates retrieved by the First-Stage Retriever (HitRate@100 is capped at 0.650).*

*\* The zero-shot and API baselines rely entirely on the history of text titles, as collaborative filtering embeddings cannot be passed to them.*

*\* see [LLM Training Notebook](/docs/LLM_training.ipynb) for training details and uploaded model weights.*

### Analysis & Baseline Commentary

* **The "Blank Slate" of the 1.5B Base Model**: The zero-shot Qwen 1.5B model performs at essentially a random guess level (yielding a HitRate@10 of only 0.082 out of a maximum possible 0.650). **Given this weak prior**, the performance leap achieved by our fine-tuned, multimodal 1.5B model is strong.
* **The Power of World Knowledge**: The API baseline utilizes `qwen3.5-plus-2026-02-15` (evaluated with `temperature=0` and `enable_thinking=False`). Relying on the textual history of movie titles, its embedded world knowledge allows it to effectively double the performance of the local zero-shot baseline.
* **Multimodal Ranking**: By fusing collaborative filtering embeddings with text, our fine-tuned 4-bit 1.5B model drastically outperforms the state-of-the-art Qwen 3.5-Plus model (397B), demonstrating a **+62% relative improvement in HitRate@10** and a **+73% relative improvement in NDCG@10**.

## Empirical Findings & Ablations

The following insights are derived from empirical ablations during the model's development:

* **Embedding Positioning**: Injecting the item embedding *before* the item title improves ranking performance compared to placing it after.
* **Negative Sampling Strategy**: Relying solely on hard negatives from the top 100 degrades training. A ratio of 1 positive, 1 hard negative, and 2 random negatives yielded the good downstream metrics.
* **Training Schedule**: We tested learning rates from `5e-5` to `2e-4` (all performed well); the reported metrics come from a run with `lr=1e-4` and cosine decay. The model was trained for ~50k steps with Unsloth 4-bit QLoRA, the best checkpoint by `NDCG@10` appeared at ≈70% of training, and we used `alpha=32`, `rank=16`.
* **End-to-End vs. Two-Stage**: While two-stage training (projection first, then full model) is beneficial for short runs (~10k steps), direct end-to-end training achieved slightly better performance over the full 50k step budget.

## Training Data

Example of a training instance prior to tokenization. The `<movie_emb>` tokens act as placeholders where the MLP-projected item embeddings are injected into the LLM's continuous input embedding space.

```text
<|im_start|>system
You are a movie recommendation assistant. You will be provided with a user's chronologically ordered movie viewing history, followed by a candidate movie. The inputs include multimodal embedding tokens representing the items. Answer Yes if you think the user will watch the candidate movie next, or No if you think they won't. Respond with Yes or No only.<|im_end|>
<|im_start|>user
User Viewing History:
1. <movie_emb> Time Machine, The (2002)
2. <movie_emb> Van Helsing (2004)
3. <movie_emb> Toys (1992)
4. <movie_emb> Scary Movie 3 (2003)
5. <movie_emb> Highlander: Endgame (Highlander IV) (2000)
6. <movie_emb> Warriors of Virtue (1997)
7. <movie_emb> Heavy Metal 2000 (2000)
8. <movie_emb> Godsend (2004)
9. <movie_emb> Masters of the Universe (1987)
10. <movie_emb> Dungeons & Dragons (2000)

Candidate Movie:
<movie_emb> Star Kid (1997)

Will the user watch this next?<|im_end|>
<|im_start|>assistant
Yes<|im_end|>
```

## Engineering & Inference Optimizations

Our pipeline implements several system-level optimizations:

* **Custom Multimodal Collator (Training):** A custom `DataCollatorForLanguageModeling` handles dynamic padding and tensor alignment for the item sequence embeddings. It ensures that cross-entropy loss is strictly calculated only on the **completion tokens** (`Yes<|im_end|>` or `No<|im_end|>`).
* **Prefix Embedding Reuse (Inference):** The evaluator (`LocalLLMScorer`) processes the system prompt and user history text *once*, caches the prefix embeddings, and then dynamically injects candidate embeddings via suffix concatenation in mini-batches.
* **High-Throughput Asynchronous Evaluation:** For API baselines, the evaluator utilizes concurrent `asyncio` and `aiohttp` request pooling with rate-limit handling and automatic caching to maximize throughput.

## Tested Environments

### Environment A: Google Colab (2026.01 Runtime)

<details>
<summary><b>Click to expand exact Colab pip dependencies</b></summary>

```text
bitsandbytes==0.49.2
cut_cross_entropy==25.1.1
datasets==4.3.0
hf_transfer==0.1.9
jedi==0.19.2
librt==0.8.1
lightning-utilities==0.15.3
msgspec==0.20.0
mypy==1.19.1
mypy_extensions==1.1.0
pathspec==1.0.4
pyarrow==23.0.1
pytorch-lightning==2.6.1
torchao==0.16.0
torchmetrics==1.9.0
transformers==5.2.0
trl==0.24.0
tyro==1.0.8
unsloth==2026.3.4
unsloth_zoo==2026.3.2
xformers==0.0.35
```

</details>

### Environment B: Docker

`pytorch/pytorch:2.10.0-cuda12.6-cudnn9-runtime`

<details>
<summary><b>Click to expand exact Docker pip dependencies</b></summary>

```text
accelerate==1.13.0
aiohappyeyeballs==2.6.1
aiohttp==3.13.3
aiosignal==1.4.0
annotated-types==0.7.0
anyio==4.12.1
attrs==25.4.0
bitsandbytes==0.49.2
comm==0.2.3
cut_cross_entropy==25.1.1
datasets==4.7.0
diffusers==0.37.0
dill==0.4.0
docstring-parser==0.17.0
frozenlist==1.8.0
gitdb==4.0.12
gitpython==3.1.46
h11==0.16.0
hf-xet==1.3.2
hf_transfer==0.1.9
httpcore==1.0.9
httpx==0.28.1
huggingface_hub==0.36.2
iniconfig==2.3.0
ipywidgets==8.1.8
jupyterlab_widgets==3.0.16
librt==0.8.1
lightning-utilities==0.15.3
msgspec==0.20.0
multidict==6.7.1
multiprocess==0.70.18
mypy==1.19.1
mypy_extensions==1.1.0
pandas==3.0.1
pathspec==1.0.4
peft==0.18.1
platformdirs==4.9.4
pluggy==1.6.0
propcache==0.4.1
protobuf==6.33.5
pyarrow==23.0.1
pydantic==2.12.5
pydantic-core==2.41.5
pydantic-settings==2.13.1
pytest==9.0.2
python-dateutil==2.9.0.post0
python-dotenv==1.2.2
pytorch-lightning==2.6.1
regex==2026.2.28
ruff==0.15.5
safetensors==0.7.0
sentencepiece==0.2.1
sentry-sdk==2.54.0
smmap==5.0.3
tokenizers==0.22.2
torchao==0.16.0
torchmetrics==1.9.0
tqdm==4.67.3
transformers==4.57.2
trl==0.23.0
typeguard==4.5.1
typing-inspection==0.4.2
tyro==1.0.8
unsloth==2025.11.1
unsloth_zoo==2025.11.2
wandb==0.25.0
widgetsnbextension==4.0.15
xformers==0.0.35
xxhash==3.6.0
yarl==1.23.0
```

</details>

### ⚠️ Note on Checkpoint Compatibility (Vocabulary Size Mismatch)

If you stick to a single environment for both training and inference, you will not face this issue.

However, if you train in Environment A (Colab) and then load the checkpoints in Environment B (Docker), or vice versa, you likely will encounter a `size mismatch` error for the `embed_tokens` layer.

**Why?** Newer versions of `unsloth` automatically inject a `<|PAD_TOKEN|>` into Qwen 2.5 models at runtime to patch a known bug ([Unsloth Issue #3721](https://github.com/unslothai/unsloth/issues/3721)). Because the provided weights were trained before this auto-patching, the tokenizer sizes will clash.

**The Fix:**
Downgrade your environment to match my training setup before loading the [provided checkpoints](https://huggingface.co/zheliu97/nanoRecSys/tree/main). Run this:

```bash
pip install transformers==4.57.2 unsloth==2025.11.1 unsloth_zoo==2025.11.2
```

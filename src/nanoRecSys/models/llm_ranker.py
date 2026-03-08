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

from unsloth import FastLanguageModel  # isort: skip

import os

import torch
import torch.nn as nn

from nanoRecSys.config import settings
from nanoRecSys.utils.logging_config import get_logger


class LLMRanker(nn.Module):
    def __init__(
        self,
        sasrec_emb_dim: int,
        model_name: str = settings.llm_model_name,
        lora_r: int = settings.llm_lora_r,
        lora_alpha: int = settings.llm_lora_alpha,
        lora_dropout: float = settings.llm_lora_dropout,
        use_lora: bool = True,
    ):
        super().__init__()

        # 1. Load Model & Tokenizer using Unsloth (handles 4-bit config natively)
        max_seq_length = settings.llm_max_seq_length
        self.llm, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,  # Auto-detect format
            load_in_4bit=True,
        )

        # 2. Add Special Token
        self.special_token = settings.llm_special_token
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [self.special_token]}
        )
        self.special_token_id = self.tokenizer.convert_tokens_to_ids(self.special_token)

        # Sanity check to ensure special token is truly just one token
        tokenized_special = self.tokenizer.encode(
            self.special_token, add_special_tokens=False
        )
        assert len(tokenized_special) == 1, (
            f"The special token '{self.special_token}' must encode to exactly 1 token! It encodes to {len(tokenized_special)} tokens."
        )

        self.llm.resize_token_embeddings(len(self.tokenizer))

        # Sync tokenizer special tokens with model config to avoid warnings
        self.llm.config.pad_token_id = self.tokenizer.pad_token_id
        self.llm.config.bos_token_id = self.tokenizer.bos_token_id
        self.llm.config.eos_token_id = self.tokenizer.eos_token_id
        if hasattr(self.llm, "generation_config"):
            self.llm.generation_config.pad_token_id = self.tokenizer.pad_token_id
            self.llm.generation_config.bos_token_id = self.tokenizer.bos_token_id
            self.llm.generation_config.eos_token_id = self.tokenizer.eos_token_id

        self.use_lora = use_lora
        if self.use_lora:
            # 3. Apply Unsloth fast QLoRA
            self.llm = FastLanguageModel.get_peft_model(
                self.llm,
                r=lora_r,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,  # Supports 0 ideally but ok
                bias="none",
                use_gradient_checkpointing="unsloth",
            )
        else:
            # Stage 1: Freeze LLM, no LoRA
            for param in self.llm.parameters():
                param.requires_grad = False

        # 4. Multimodal Projection Layer
        # https://github.com/ljy0ustc/LLaRA/blob/main/model/mlp_projector.py
        hidden_size = self.llm.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(sasrec_emb_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.projection.to(device=self.llm.device)

        # Expose underlying model config for trainer compatibility
        # Some trainer implementations expect `model.config` to exist.
        self.config = self.llm.config

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sasrec_embs: torch.Tensor,
        labels=None,
        **kwargs,  # trainer compatibility
    ):
        """
        Multimodal forward pass.
        - input_ids: (batch_size, seq_len)
        - sasrec_embs: (batch_size, num_movies, sasrec_emb_dim)
        """
        batch_size, seq_len = input_ids.shape

        # 1. Get raw token embeddings from the base language model
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)

        # Clone input embeddings so we are not modifying weights via in-place ops for backprop
        inputs_embeds = inputs_embeds.clone()

        # 2. Project SASRec item embeddings
        # Cast input to match projection layer dtype so it works correctly under autocast
        projected_embs = self.projection(
            sasrec_embs.to(
                dtype=self.projection[0].weight.dtype,  # type: ignore
                device=self.llm.device,
            )
        )
        # Cast back to inputs_embeds dtype (often fp16/bf16 via autocast)
        projected_embs = projected_embs.to(inputs_embeds.dtype)

        # 3. Inject the projected embeddings into `inputs_embeds`
        movie_emb_mask = input_ids == self.special_token_id

        for i in range(batch_size):
            cnt = movie_emb_mask[i].sum().item()
            if cnt > 0:
                num_valid_embs = projected_embs.size(1)
                actual_cnt = min(cnt, num_valid_embs)
                if actual_cnt > 0:
                    indices = movie_emb_mask[i].nonzero(as_tuple=True)[0][:actual_cnt]
                    inputs_embeds[i, indices] = projected_embs[i, :actual_cnt]

        # 4. Pass modified `inputs_embeds` through rest of LLM
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

        return outputs

    def print_trainable_parameters(self):
        """Prints the number of trainable parameters in the model."""
        logger = get_logger()
        # Projection trainable params (always accurate)
        proj_trainable = sum(
            p.numel() for p in self.projection.parameters() if p.requires_grad
        )

        if self.use_lora:
            # Use PEFT's built-in count as baseline, then include projection totals
            peft_trainable_params, peft_all_param = (
                self.llm.get_nb_trainable_parameters()
            )
            proj_all = sum(p.numel() for p in self.projection.parameters())

            total_trainable = peft_trainable_params + proj_trainable
            total_all = peft_all_param + proj_all

            logger.info(
                f"trainable params: {total_trainable:,d} || all params: {total_all:,d} || trainable%: {100 * total_trainable / total_all:.4f}"
            )
        else:
            # TODO: Revisit and implement accurate total parameter counting for
            # 4-bit quantized models if/when needed.
            total_trainable = proj_trainable
            logger.info(f"trainable params: {total_trainable:,d}")

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def get_output_embeddings(self):
        return self.llm.get_output_embeddings()

    @property
    def device(self):
        return self.llm.device

    def gradient_checkpointing_enable(self, **kwargs):
        """Enable gradient checkpointing on the underlying LLM if supported."""
        if hasattr(self.llm, "gradient_checkpointing_enable"):
            return self.llm.gradient_checkpointing_enable(**kwargs)
        base = getattr(self.llm, "base_model", None)
        if base is not None and hasattr(base, "gradient_checkpointing_enable"):
            return base.gradient_checkpointing_enable(**kwargs)
        return None

    def gradient_checkpointing_disable(self, **kwargs):
        """Disable gradient checkpointing on the underlying LLM if supported."""
        if hasattr(self.llm, "gradient_checkpointing_disable"):
            return self.llm.gradient_checkpointing_disable(**kwargs)
        base = getattr(self.llm, "base_model", None)
        if base is not None and hasattr(base, "gradient_checkpointing_disable"):
            return base.gradient_checkpointing_disable(**kwargs)
        return None

    def load_checkpoint(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        strict: bool = False,
    ) -> bool:
        """Load projection and LoRA adapter weights from checkpoint directory.

        Args:
            checkpoint_path: Path to checkpoint directory containing projection.pth
                             and optionally adapter_model.{safetensors,bin}
            device: Device to load projection weights to (before moving to model device).
            strict: If False, allows partial checkpoint loading (e.g., from Stage 1 to Stage 2).

        Returns:
            True if projection loaded successfully, False otherwise.
        """
        logger = get_logger()
        projection_path = os.path.join(checkpoint_path, "projection.pth")

        # Load projection
        projection_loaded = False
        if os.path.exists(projection_path):
            try:
                state_dict = torch.load(projection_path, map_location=device)
                self.projection.load_state_dict(state_dict, strict=strict)
                logger.info(f"Loaded projection from {projection_path}")
                projection_loaded = True
            except Exception as e:
                logger.warning(f"Failed to load projection from {projection_path}: {e}")
        else:
            logger.warning(f"Projection file not found at {projection_path}")

        # Load LoRA adapter if enabled
        if self.use_lora:
            adapter_file = os.path.join(checkpoint_path, "adapter_model.safetensors")
            adapter_bin = os.path.join(checkpoint_path, "adapter_model.bin")

            if os.path.exists(adapter_file) or os.path.exists(adapter_bin):
                try:
                    from peft import load_peft_weights, set_peft_model_state_dict

                    adapters_weights = load_peft_weights(checkpoint_path)
                    set_peft_model_state_dict(self.llm, adapters_weights)
                    logger.info(f"Loaded LoRA adapter from {checkpoint_path}")
                except Exception as e:
                    logger.warning(
                        f"Failed to load LoRA adapter from {checkpoint_path}: {e}"
                    )
            else:
                logger.info(
                    f"No LoRA adapter found in {checkpoint_path} "
                    "(using base model weights or bridging from Stage 1)"
                )

        return projection_loaded

    def save_checkpoint(self, checkpoint_path: str) -> None:
        """Save projection and LoRA adapter weights to checkpoint directory.

        Args:
            checkpoint_path: Path to directory where weights will be saved.
        """
        logger = get_logger()

        os.makedirs(checkpoint_path, exist_ok=True)

        # Save projection
        projection_path = os.path.join(checkpoint_path, "projection.pth")
        torch.save(self.projection.state_dict(), projection_path)
        logger.info(f"Saved projection to {projection_path}")

        # Save LoRA adapter if enabled
        if self.use_lora:
            try:
                self.llm.save_pretrained(checkpoint_path)
                logger.info(f"Saved LoRA adapter to {checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to save LoRA adapter: {e}")

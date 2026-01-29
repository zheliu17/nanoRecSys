import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class InfoNCELoss(nn.Module):
    """
    InfoNCE Loss with in-batch negatives.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        candidate_probs: "Optional[torch.Tensor]" = None,
        user_ids: "Optional[torch.Tensor]" = None,
    ) -> torch.Tensor:
        """
        Args:
            user_embeddings: (batch_size, dim)
            item_embeddings: (batch_size, dim) - positive items corresponding to users
            candidate_probs: (batch_size, ) - probability of each item in the batch
            user_ids: (batch_size, ) - User IDs to mask collisions (same user in batch)
        """
        # Cosine similarity (assuming embeddings are already specialized)
        # logits: (batch, batch)
        logits = torch.matmul(user_embeddings, item_embeddings.T) / self.temperature
        batch_size = user_embeddings.size(0)

        # Apply user collision masking if user_ids are provided
        if user_ids is not None:
            match_mask = user_ids.unsqueeze(1) == user_ids.unsqueeze(0)
            identity_mask = torch.eye(batch_size, device=user_embeddings.device).bool()
            collision_mask = match_mask & (~identity_mask)
            logits = logits.masked_fill(collision_mask, -1e9)

        if candidate_probs is not None:
            epsilon = 1e-10
            log_probs = torch.log(candidate_probs + epsilon)
            logits = logits - log_probs.unsqueeze(0)

        # Targets: 0, 1, 2, ..., batch_size-1
        # Each user i matches item i
        labels = torch.arange(batch_size, device=user_embeddings.device)

        return self.criterion(logits, labels)


class MarginRankingLossWrapper(nn.Module):
    """
    Wrapper around PyTorch's MarginRankingLoss for ranking problems.
    Used for ranker training with positive/negative labels.
    """

    def __init__(self, margin: float = 1.0, reduction: str = "none"):
        super().__init__()
        self.margin = margin
        self.reduction = reduction
        self.criterion = nn.MarginRankingLoss(margin=margin, reduction=reduction)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: (B,) - predicted scores
            labels: (B,) - binary labels (0 or 1)
            weights: (B,) - optional weights for each sample

        Returns:
            Scalar loss value
        """
        if weights is not None and self.reduction == "none":
            # Apply weights to unreduced loss
            unreduced_loss = self.criterion(logits, labels)
            return (unreduced_loss * weights).mean()
        else:
            return self.criterion(logits, labels)


class BPRLossWrapper(nn.Module):
    """
    BPR (Bayesian Personalized Ranking) Loss for ranker training.
    Pairs positive and negative predictions within each batch to compute ranking loss.
    loss = -log(sigmoid(pos_logits - neg_logits))
    """

    def __init__(self, reduction: str = "none"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: (B,) - predicted scores
            labels: (B,) - binary labels (0 or 1, where 1 is positive)
            weights: (B,) - optional weights for each sample

        Returns:
            Scalar loss value or unreduced loss if reduction='none'
        """
        # Separate positive and negative logits
        pos_mask = labels == 1
        neg_mask = labels == 0

        pos_logits = logits[pos_mask]
        neg_logits = logits[neg_mask]

        if pos_logits.numel() == 0 or neg_logits.numel() == 0:
            # If no positives or negatives, return zero loss
            return torch.tensor(0.0, device=logits.device)

        # For each positive, compute BPR loss with all negatives
        # pos_logits: (n_pos,), neg_logits: (n_neg,)
        # We compute pairwise comparison: pos - neg for each pos-neg pair
        pos_neg_diff = pos_logits.unsqueeze(1) - neg_logits.unsqueeze(
            0
        )  # (n_pos, n_neg)

        # BPR loss: -log(sigmoid(pos - neg))
        loss_matrix = -F.logsigmoid(pos_neg_diff)  # (n_pos, n_neg)

        # Average over all pairs
        loss = loss_matrix.mean()

        if weights is not None and self.reduction == "none":
            # Weight positive samples
            pos_weights = weights[pos_mask]
            if pos_weights.numel() > 0:
                weighted_loss = (loss_matrix * pos_weights.unsqueeze(1)).mean()
                return weighted_loss

        return loss


def get_ranker_loss(loss_type: str = "bce", **kwargs) -> nn.Module:
    """
    Factory function to create ranker loss functions.

    Args:
        loss_type: Type of loss function. Options: 'bce', 'margin_ranking', 'bpr'
        **kwargs: Additional arguments for the loss function

    Returns:
        Loss function module
    """
    if loss_type.lower() == "bce":
        reduction = kwargs.get("reduction", "none")
        return nn.BCEWithLogitsLoss(reduction=reduction)
    elif loss_type.lower() == "margin_ranking":
        margin = kwargs.get("margin", 1.0)
        reduction = kwargs.get("reduction", "none")
        return MarginRankingLossWrapper(margin=margin, reduction=reduction)
    elif loss_type.lower() == "bpr":
        reduction = kwargs.get("reduction", "none")
        return BPRLossWrapper(reduction=reduction)
    else:
        raise ValueError(
            f"Unknown loss type: {loss_type}. Supported types: 'bce', 'margin_ranking', 'bpr'"
        )

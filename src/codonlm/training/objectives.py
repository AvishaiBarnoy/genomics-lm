import torch

PAD_ID = 0
DEFAULT_BOUNDARY_IDS = (2, 3)  # <EOS_CDS>, <SEP>

def offset_target_mask(yb: torch.Tensor, offset: int, boundary_ids=DEFAULT_BOUNDARY_IDS) -> torch.Tensor:
    """Return valid positions for predicting seq[t + offset] from logits at t."""
    if offset < 1:
        raise ValueError("offset must be >= 1")
    if offset > yb.shape[1]:
        return torch.zeros((yb.shape[0], 0), dtype=torch.bool, device=yb.device)

    target = yb[:, offset - 1 :]
    valid = target != PAD_ID
    boundary = torch.zeros_like(yb, dtype=torch.bool)
    for boundary_id in boundary_ids:
        boundary |= yb == int(boundary_id)

    # Do not train targets that require looking beyond an EOS/SEP boundary.
    # The target boundary itself is allowed; only earlier boundaries invalidate it.
    for shift in range(offset - 1):
        valid &= ~boundary[:, shift : shift + target.shape[1]]
    return valid


def multi_offset_lm_loss(
    logits: torch.Tensor | dict[int, torch.Tensor],
    yb: torch.Tensor,
    offset_weights: dict[int, float],
    label_smoothing: float = 0.0,
    loss_weights: torch.Tensor | None = None,
    boundary_ids=DEFAULT_BOUNDARY_IDS,
):
    losses = {}
    total = logits.new_tensor(0.0) if hasattr(logits, "new_tensor") else torch.tensor(0.0, device=yb.device)
    for offset, weight in offset_weights.items():
        if weight == 0.0 or offset <= 1 or offset > yb.shape[1]:
            continue
        target = yb[:, offset - 1 :]
        if isinstance(logits, dict):
            if offset not in logits:
                continue
            pred = logits[offset][:, : target.shape[1], :]
        else:
            pred = logits[:, : target.shape[1], :]
        valid = offset_target_mask(yb, offset, boundary_ids=boundary_ids)
        if not bool(valid.any()):
            continue
        pred_flat = pred[valid].float()
        target_flat = target[valid]
        offset_loss = torch.nn.functional.cross_entropy(
            pred_flat,
            target_flat,
            ignore_index=PAD_ID,
            label_smoothing=label_smoothing,
            weight=loss_weights,
        )
        losses[offset] = offset_loss
        total = total + (float(weight) * offset_loss)
    return total, losses


def termination_distance_bucket_labels(
    yb: torch.Tensor,
    stop_ids: tuple[int, ...],
    bucket_edges: tuple[int, ...] = (0, 3, 10, 30),
    ignore_index: int = -100,
) -> torch.Tensor:
    """Bucket distance from each target position to the next stop token."""
    if not stop_ids:
        raise ValueError("stop_ids must not be empty")
    if tuple(bucket_edges) != tuple(sorted(bucket_edges)):
        raise ValueError("bucket_edges must be sorted")

    labels = torch.full_like(yb, fill_value=ignore_index, dtype=torch.long)
    n_classes = len(bucket_edges) + 1
    for row_idx in range(yb.shape[0]):
        row = yb[row_idx]
        valid_positions = row != PAD_ID
        stop_positions = torch.nonzero(
            torch.isin(row, torch.tensor(stop_ids, device=row.device)),
            as_tuple=False,
        ).flatten()
        for pos in torch.nonzero(valid_positions, as_tuple=False).flatten():
            future_stops = stop_positions[stop_positions >= pos]
            if len(future_stops) == 0:
                labels[row_idx, pos] = n_classes - 1
                continue
            distance = int(future_stops[0].item() - pos.item())
            bucket = 0
            while bucket < len(bucket_edges) and distance > int(bucket_edges[bucket]):
                bucket += 1
            labels[row_idx, pos] = bucket
    return labels


def termination_aux_loss(
    termination_logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(
        termination_logits.float().view(-1, termination_logits.size(-1)),
        labels.contiguous().view(-1),
        weight=class_weights,
        ignore_index=ignore_index,
    )

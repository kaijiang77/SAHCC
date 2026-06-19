import torch


DEFAULT_THRESHOLDS = (0.4, 0.5, 0.6)
THRESHOLDS = DEFAULT_THRESHOLDS


def update_count_errors(
    abs_err: dict,
    sq_err: dict,
    prob: torch.Tensor,
    gt_cnt: torch.Tensor,
    thresholds=DEFAULT_THRESHOLDS,
):
    for thr in thresholds:
        pred_cnt = (prob > thr).sum(dim=1).long()
        diff = (pred_cnt - gt_cnt).float()
        abs_err[thr].extend(diff.abs().detach().cpu().tolist())
        sq_err[thr].extend((diff * diff).detach().cpu().tolist())


def summarize_count_metrics(abs_err: dict, sq_err: dict, thresholds=DEFAULT_THRESHOLDS):
    out = {}
    for thr in thresholds:
        mae = float(torch.tensor(abs_err[thr]).mean().item()) if abs_err[thr] else 0.0
        rmse = float(torch.tensor(sq_err[thr]).mean().sqrt().item()) if sq_err[thr] else 0.0
        out[thr] = (mae, rmse)
    return out

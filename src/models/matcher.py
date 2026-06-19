import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


class HungarianMatcherCrowd(nn.Module):
    def __init__(
        self,
        cost_class: float = 1.0,
        cost_point: float = 1.0,
        point_source: str = "anchor_points",
    ):
        super().__init__()
        if point_source not in {"anchor_points", "pred_points"}:
            raise ValueError(f"Unsupported point_source: {point_source}")
        if cost_class == 0 and cost_point == 0:
            raise ValueError("cost_class and cost_point cannot both be 0")

        self.cost_class = float(cost_class)
        self.cost_point = float(cost_point)
        self.point_source = point_source

    @torch.no_grad()
    def forward(self, outputs, targets):
        logits = outputs["pred_logits"]
        if logits.dim() == 3:
            logits = logits.squeeze(-1)
        prob_fg = torch.sigmoid(logits)

        src_points_all = outputs[self.point_source]
        indices = []
        for i in range(prob_fg.shape[0]):
            src_prob = prob_fg[i]
            src_points = src_points_all[i]
            tgt_points = targets[i]["points"]

            q = src_points.shape[0]
            m = tgt_points.shape[0]
            if q == 0 or m == 0:
                indices.append(
                    (
                        torch.empty((0,), dtype=torch.int64, device=src_points.device),
                        torch.empty((0,), dtype=torch.int64, device=src_points.device),
                    )
                )
                continue

            cost_class = -src_prob[:, None].expand(q, m)
            cost_point = torch.cdist(src_points, tgt_points, p=2)
            # Baseline matcher: no local-spacing reweighting; SAH lives in src.models.sah.
            cost = self.cost_class * cost_class + self.cost_point * cost_point

            row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
            indices.append(
                (
                    torch.as_tensor(row_ind, dtype=torch.int64, device=src_points.device),
                    torch.as_tensor(col_ind, dtype=torch.int64, device=src_points.device),
                )
            )
        return indices


HungarianMatcher_Crowd = HungarianMatcherCrowd


def build_matcher_crowd_from_matcher(args):
    return HungarianMatcherCrowd(
        cost_class=args.set_cost_class,
        cost_point=args.set_cost_point,
        point_source=args.matcher_point_source,
    )


def build_matcher_crowd(args):
    matcher_impl = str(args.matcher_impl).lower()
    if matcher_impl in {"matcher", "matcher.py"}:
        return build_matcher_crowd_from_matcher(args)
    if matcher_impl in {"sah", "sah.py"}:
        # Keep a single public builder while letting configs select the matcher cost.
        from src.models.sah import build_matcher_crowd as build_matcher_crowd_from_sah

        return build_matcher_crowd_from_sah(args)
    raise ValueError(f"Unsupported matcher_impl: {matcher_impl}. Expected 'matcher' or 'sah'.")

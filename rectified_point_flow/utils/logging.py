from collections import defaultdict
from itertools import chain
import re
from typing import Dict, List, Any

import lightning as L
import torch
import torch.distributed as dist
from rich.console import Console
from rich.table import Table


def log_metrics_on_step(
    module: L.LightningModule,
    metrics: Dict[str, float],
    prefix: str = "train",
):
    """Log per‐step scalars only on rank 0."""
    for name, value in metrics.items():
        module.log(
            f"{prefix}/{name}",
            value,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            rank_zero_only=True,
        )


def log_metrics_on_epoch(
    module: L.LightningModule,
    metrics: Dict[str, float],
    prefix: str = "val",
):
    """Log per‐epoch scalars, automatically synced & averaged across ranks."""
    for name, value in metrics.items():
        module.log(
            f"{prefix}/{name}",
            value,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            rank_zero_only=True,
        )

def print_eval_table(
    results: list[dict[str, float]],
    dataset_names: list[str],
    digits: int = 5,
) -> None:
    """
    Pretty-print a list of evaluation-result dicts with Rich, split into Avg and BoN sections.

    Args:
        results: List of dicts, each with keys like
                 "avg/rotation_error/dataloader_idx_0".
        dataset_names: Mapping from dataloader_idx (int) to column name (str).
        digits: Number of decimal places for floats.
    """
    # Group dicts by dataloader_idx
    per_idx: Dict[int, Dict[str, Any]] = {}
    idx_pattern = re.compile(r"dataloader_idx_(\d+)")
    for d in results:
        sample_key = next(iter(d))
        idx = int(idx_pattern.search(sample_key).group(1))
        per_idx[idx] = d
    metric_pattern = re.compile(r"^(.+?/.+?)/dataloader_idx_\d+$")
    metrics = set()
    for d in results:
        for k in d:
            m = metric_pattern.match(k)
            if m:
                metrics.add(m.group(1))

    # Split into two sections for Avg and BoN metrics
    avg_metrics = sorted(m for m in metrics if m.lower().startswith("avg/"))
    bon_metrics = sorted(m for m in metrics if m.lower().startswith("bon/"))
    table = Table()
    table.add_column("Metrics", style="bold magenta", justify="left")
    for idx in sorted(per_idx):
        col_name = dataset_names[idx]
        table.add_column(col_name, style="cyan")

    fmt = f"{{:.{digits}f}}"
    for metric in avg_metrics:
        row = [metric]
        for idx in sorted(per_idx):
            key = f"{metric}/dataloader_idx_{idx}"
            val = per_idx[idx].get(key)
            if val is None:
                row.append("-")
            elif isinstance(val, float):
                row.append(fmt.format(val))
            else:
                row.append(str(val))
        table.add_row(*row)

    table.add_section()

    for metric in bon_metrics:
        row = [metric]
        for idx in sorted(per_idx):
            key = f"{metric}/dataloader_idx_{idx}"
            val = per_idx[idx].get(key)
            if val is None:
                row.append("-")
            elif isinstance(val, float):
                row.append(fmt.format(val))
            else:
                row.append(str(val))
        table.add_row(*row)

    Console().print(table)


class MetricsMeter:
    """Helper class for accumulating metrics for each dataset.

    Example:
        >>> metrics_meter = MetricsMeter(module)
        >>> metrics_meter.add_metrics(
                dataset_names=["A", "B", "A"], 
                loss=torch.tensor([0.1, 0.2, 0.3]),
                acc=torch.tensor([0.9, 0.8, 0.7]),
            )
        >>> metrics_meter.add_metrics(
                dataset_names=["A", "B", "C"], 
                loss=torch.tensor([0.4, 0.5, 0.6]),
                acc=torch.tensor([0.6, 0.5, 0.4]),
            )
        >>> results = metrics_meter.log_on_epoch_end()
        >>> print(results)
        {
            "A/loss": 0.2667,
            "A/acc": 0.7333,
            "B/loss": 0.35,
            "B/acc": 0.65,
            "C/loss": 0.6,
            "C/acc": 0.4,
            "overall/loss": 0.35,
            "overall/acc": 0.65,
        }
    """

    def __init__(self, module: L.LightningModule):
        self.module = module
        self.reset()

    def reset(self):
        self._sums = defaultdict(lambda: defaultdict(float))
        self._counts = defaultdict(lambda: defaultdict(int))
        self._metrics_seen = set()

    def add_metrics(self, dataset_names: List[str], **metrics: torch.Tensor):
        """Accumulate a batch of per-sample metrics."""
        if not metrics:
            return
        
        if any(ds == "overall" for ds in dataset_names):
            raise ValueError("'overall' is a reserved dataset name and cannot be used.")

        B = next(iter(metrics.values())).shape[0]
        if len(dataset_names) != B:
            raise ValueError(f"len(dataset_names)={len(dataset_names)} != batch size {B}")
        for k, t in metrics.items():
            if t.shape[0] != B:
                raise ValueError(f"metric '{k}' has shape {t.shape} != ({B},)")
            self._metrics_seen.add(k)

        for i, ds in enumerate(dataset_names):
            for k, t in metrics.items():
                v = t[i].item()
                self._sums[k][ds] += v
                self._counts[k][ds] += 1
                self._sums[k]["_overall"] += v
                self._counts[k]["_overall"] += 1

    def compute_average(self) -> Dict[str, torch.Tensor]:
        """Gather per-dataset sums/counts, and compute global averages."""
        # local dataset list
        local_ds = sorted(
            set(chain.from_iterable(self._counts[k].keys() for k in self._metrics_seen))
            - {"_overall"}
        )

        # gather global dataset list
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if world_size > 1:
            gathered = [None] * world_size
            dist.all_gather_object(gathered, local_ds)
            global_ds = sorted(set(chain.from_iterable(gathered)))
        else:
            global_ds = local_ds

        # flatten sums and counts with fixed order
        metrics = sorted(self._metrics_seen)
        N = len(global_ds) + 1
        flat_sums = []
        flat_counts = []
        for k in metrics:
            for ds in global_ds:
                flat_sums.append(self._sums[k].get(ds, 0.0))
                flat_counts.append(self._counts[k].get(ds, 0))
            flat_sums.append(self._sums[k].get("_overall", 0.0))
            flat_counts.append(self._counts[k].get("_overall", 0))

        device = getattr(self.module, "device", torch.device("cpu")) or torch.device("cpu")
        sums_t = torch.tensor(flat_sums, dtype=torch.float64, device=device)
        counts_t = torch.tensor(flat_counts, dtype=torch.float64, device=device)

        # all-reduce
        if world_size > 1:
            dist.all_reduce(sums_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(counts_t, op=dist.ReduceOp.SUM)

        # compute average metrics
        results: Dict[str, torch.Tensor] = {}
        for idx, k in enumerate(metrics):
            for j, ds in enumerate(global_ds + ["overall"]):
                pos = idx * N + j
                total_sum = sums_t[pos]
                total_count = counts_t[pos]
                avg = (
                    total_sum / total_count if total_count > 0 
                    else torch.tensor(float("nan"), device=device)
                )
                results[f"{ds}/{k}"] = avg

        # reset for next epoch
        self.reset()
        return results

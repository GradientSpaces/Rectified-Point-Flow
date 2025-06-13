from typing import List, Dict

import torch
import lightning as L

# Metrics Handler

class MetricsHandler:
    """Lightweight and flexible handler for metrics aggregation and logging."""
    
    def __init__(self, module: L.LightningModule):
        self.module = module
        self.reset()
    
    def reset(self):
        """Reset all metric storage."""
        self.metrics = {}
        self.dataset_names = []
    
    def add_metrics(self, dataset_names: List[str], **metrics: torch.Tensor):
        """Add metrics dynamically.
        
        Args:
            dataset_names: List of dataset names for this batch
            **metrics: Any number of metric tensors with their names
        """
        self.dataset_names.append(dataset_names)
        for metric_name, metric_tensor in metrics.items():
            if metric_name not in self.metrics:
                self.metrics[metric_name] = []
            self.metrics[metric_name].append(metric_tensor)
    
    def log_on_epoch_end(self, prefix: str = "eval") -> Dict[str, torch.Tensor]:
        """Aggregate and log epoch-end metrics."""
        if not self.metrics or not self.dataset_names:
            return {}
        
        all_dataset_names = sum(self.dataset_names, [])
        aggregated_metrics = {}
        for metric_name, metric_list in self.metrics.items():
            aggregated_metrics[metric_name] = torch.cat(metric_list)

        # Distributed gathering
        gathered_metrics = {}
        for metric_name, metric_tensor in aggregated_metrics.items():
            gathered_metrics[metric_name] = self.module.all_gather(metric_tensor).view(-1)
        gathered_dataset_names = self.module.all_gather(all_dataset_names)
        results = {}
        if self.module.global_rank == 0:
            # Per dataset metrics
            unique_dataset_names = sorted(set(gathered_dataset_names))
            for dataset in unique_dataset_names:
                cat_indices = torch.tensor(
                    [i for i, c in enumerate(gathered_dataset_names) if c == dataset], 
                    device=self.module.device
                )
                for metric_name, metric_values in gathered_metrics.items():
                    cat_metric_value = metric_values[cat_indices].mean()
                    self.module.log(f"{prefix}/{dataset}_{metric_name}", cat_metric_value, sync_dist=False)
            
            # Log overall metrics
            for metric_name, metric_values in gathered_metrics.items():
                overall_value = metric_values.mean()
                results[f"overall_{metric_name}"] = overall_value
                self.module.log(f"{prefix}/overall_{metric_name}", overall_value, sync_dist=False)
        else:
            for metric_name in self.metrics.keys():
                results[f"overall_{metric_name}"] = torch.tensor(0.0, device=self.module.device)

        self.reset()
        return results 
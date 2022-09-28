from typing import Callable
import torch
from torch import nn
from torch.utils.data import DataLoader
from bigdl.nano.pytorch import InferenceOptimizer
from bigdl.nano.pytorch.inference.optimizer import _format_optimize_result

class YoloxInferenceOptimizer(InferenceOptimizer):
    def optimize(self,
                 model: nn.Module,
                 training_data: DataLoader,
                 validation_data: DataLoader = None,
                 metric: Callable = None,
                 direction: str = "max",
                 cpu_num: int = None,
                 logging: bool = False,
                 latency_sample_num: int = 100,
                 **metric_kwargs) -> None:

        super().optimize(model,
                         training_data,
                         validation_data,
                         None,
                         direction,
                         cpu_num,
                         logging,
                         latency_sample_num)

        if metric:
            for method, acce_result in self.optimized_model_dict.items():
                with torch.no_grad():
                    result = metric(acce_result["model"], **metric_kwargs)
                self.optimized_model_dict[method]["accuracy"] = result

        self._optimize_result = _format_optimize_result(self.optimized_model_dict,
                                                                self._calculate_accuracy)

        

        

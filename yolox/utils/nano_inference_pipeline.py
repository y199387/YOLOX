from typing import Callable
import torch
from torch import nn
from torch.utils.data import DataLoader
from bigdl.nano.pytorch import InferenceOptimizer

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

        self._optimize_result = self._format_optimize_result(self.optimized_model_dict,
                                                                self._calculate_accuracy)


    def _format_optimize_result(optimize_result_dict: dict,
                                calculate_accuracy: bool) -> str:
        if calculate_accuracy is True:
            horizontal_line = " {0} {1} {2} {3}\n" \
                .format("-" * 32, "-" * 22, "-" * 14, "-" * 22)
            repr_str = horizontal_line
            repr_str += "| {0:^30} | {1:^20} | {2:^12} | {3:^20} |\n" \
                .format("method", "status", "latency(ms)", "accuracy")
            repr_str += horizontal_line
            for method, result in optimize_result_dict.items():
                status = result["status"]
                latency = result.get("latency", "None")
                if latency != "None":
                    latency = round(latency, 3)
                accuracy = result.get("accuracy", "None")
                if accuracy != "None" and isinstance(accuracy, float):
                    accuracy = round(accuracy, 3)
                method_str = f"| {method:^30} | {status:^20} | " \
                            f"{latency:^12} | {accuracy:^20} |\n"
                repr_str += method_str
            repr_str += horizontal_line
        else:
            horizontal_line = " {0} {1} {2}\n" \
                .format("-" * 32, "-" * 22, "-" * 14)
            repr_str = horizontal_line
            repr_str += "| {0:^30} | {1:^20} | {2:^12} |\n" \
                .format("method", "status", "latency(ms)")
            repr_str += horizontal_line
            for method, result in optimize_result_dict.items():
                status = result["status"]
                latency = result.get("latency", "None")
                if latency != "None":
                    latency = round(latency, 3)
                method_str = f"| {method:^30} | {status:^20} | {latency:^12} |\n"
                repr_str += method_str
            repr_str += horizontal_line
        return repr_str


        

        

import numpy as np
import os.path as osp
from collections import OrderedDict, defaultdict
import torch
from sklearn.metrics import f1_score, confusion_matrix

from ..build import EVALUATOR_REGISTRY, EvaluatorBase


def CCC(x, y):
    """Concordance Correlation Coefficient metric."""
    # Compute means
    x_mean = torch.mean(x, dim=0)
    y_mean = torch.mean(y, dim=0)
    # Compute variances
    x_var = torch.var(x, dim=0)
    y_var = torch.var(y, dim=0)
    # Compute covariance matrix
    cov_matrix = torch.matmul(
        (x - x_mean).permute(*torch.arange(x.dim() - 1, -1, -1)), y - y_mean
    ) / (x.size(0) - 1)
    # Compute CCC
    numerator = 2 * cov_matrix
    denominator = x_var + y_var + torch.pow((x_mean - y_mean), 2)
    ccc = torch.mean(numerator / denominator)
    return ccc


@EVALUATOR_REGISTRY.register()
class Regression(EvaluatorBase):
    """Evaluator for regression."""

    def __init__(self, cfg, lab2cname=None, **kwargs):
        super().__init__(cfg)
        self._lab2cname = lab2cname
        self._total = 0
        self._y_true = []
        self._y_pred = []
        self._domain_labels = []

    def reset(self):
        self._total = 0
        self._y_true = []
        self._y_pred = []
        self._domain_labels = []

    def process(self, mo, gt, domain_label):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.Tensor): ground truth [batch, num_classes]
        self._total += gt.shape[0]
        
        self._y_true.extend(gt.data.cpu().numpy().tolist())
        self._y_pred.extend(mo.data.cpu().numpy().tolist())
        self._domain_labels.extend(domain_label)
    
    def _evaluate(self, y_true: np.ndarray, y_pred: np.ndarray):
        results = OrderedDict()
        
        # Calculate MAE, MSE and RMSE for all dimensions
        mae = np.mean(np.abs(y_true - y_pred))
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        
        # Calculate correlation coefficient (PCC)
        corr = np.mean([np.corrcoef(y_true[:, i], y_pred[:, i])[0,1] 
                       for i in range(y_true.shape[1])])
        
        # Calculate CCC
        y_true_tensor = torch.tensor(y_true)
        y_pred_tensor = torch.tensor(y_pred)
        ccc = np.mean([CCC(y_true_tensor[:, i], y_pred_tensor[:, i]).item() 
                       for i in range(y_true.shape[1])])

        # The first value will be returned by trainer.test()
        results["mae"] = mae
        results["rmse"] = rmse
        results["mse"] = mse
        results["correlation"] = corr
        results["ccc"] = ccc

        if self.cfg.TEST.PER_CLASS_RESULT:
            mae_metrics = []
            mse_metrics = []
            rmse_metrics = []
            corr_metrics = []
            ccc_metrics = []

            for dim in range(y_true.shape[1]):
                errors = y_pred[:, dim] - y_true[:, dim]
                mae_dim = np.mean(np.abs(errors))
                mse_dim = np.mean(errors ** 2)
                rmse_dim = np.sqrt(mse_dim)
                corr_dim = np.corrcoef(y_true[:, dim], y_pred[:, dim])[0,1]
                ccc_dim = CCC(torch.tensor(y_true[:, dim]), torch.tensor(y_pred[:, dim])).item()
                
                mae_metrics.append(mae_dim)
                mse_metrics.append(mse_dim)
                rmse_metrics.append(rmse_dim)
                corr_metrics.append(corr_dim)
                ccc_metrics.append(ccc_dim)
                
                classname = self._lab2cname[dim] if self._lab2cname else f"dim_{dim}"
                
                # Add per-dimension metrics with class name
                results[f"{classname}_mae"] = mae_dim
                results[f"{classname}_rmse"] = rmse_dim
                results[f"{classname}_mse"] = mse_dim
                results[f"{classname}_correlation"] = corr_dim
                results[f"{classname}_ccc"] = ccc_dim
        
        return results

    def print_results(self, results):
        print(
            "=> result\n"
            f"* total: {self._total:,}\n"
            f"* MAE: {results['mae']:.2f}\n"
            f"* RMSE: {results['rmse']:.2f}\n"
            f"* MSE: {results['mse']:.2f}\n"
            f"* PCC: {results['correlation']:.2f}\n"
            f"* CCC: {results['ccc']:.2f}"
        )
        
        if self.cfg.TEST.PER_CLASS_RESULT:
            print("Per-dimension result:")
            
            for dim in range(len(self._lab2cname)):
                classname = self._lab2cname[dim] if self._lab2cname else f"dim_{dim}"
                print(
                    f"* dimension: {dim} ({classname})\t"
                    f"MAE: {results[f'{classname}_mae']:.2f}\t"
                    f"RMSE: {results[f'{classname}_rmse']:.2f}\t"
                    f"MSE: {results[f'{classname}_mse']:.2f}\t"
                    f"PCC: {results[f'{classname}_correlation']:.2f}\t"
                    f"CCC: {results[f'{classname}_ccc']:.2f}"
                )
    
    def evaluate_average_on_domain(self):
        results_domain = OrderedDict()
        for domain in set(self._domain_labels):
            print(f"Evaluating domain: {domain}")
            
            mask = np.array(self._domain_labels) == domain
            y_true = np.array(self._y_true)[mask]
            y_pred = np.array(self._y_pred)[mask]
            
            _results = self._evaluate(y_true, y_pred)
            
            for key, value in _results.items():
                results_domain[f"{key}_domain_{domain}"] = value

        # average on domains
        results = OrderedDict()
        for key in _results.keys():
            results[key] = np.mean([results_domain[f"{key}_domain_{domain}"] for domain in set(self._domain_labels)])
        
        self.print_results(results)
        
        return results
    
    def evaluate_average_on_instance(self):
        y_true = np.array(self._y_true)
        y_pred = np.array(self._y_pred)
        
        results = self._evaluate(y_true, y_pred)

        self.print_results(results)
        
        return results
    
    def evaluate(self):
        
        if self.cfg.TEST.AVG_ON_DOMAIN:
            print("Results are averaged on domains.")
            return self.evaluate_average_on_domain()
        else:
            print("Results are averaged on instances.")
            return self.evaluate_average_on_instance()

import torch
from scipy.stats import spearmanr


def spearman_correlation(predictions, targets):
    """
    Calculate Spearman rank correlation using PyTorch's built-in functions.
    Handles ties properly (matching SciPy's implementation).
    
    Args:
        predictions: tensor of predicted values
        targets: tensor of target values
        
    Returns:
        Spearman rank correlation coefficient
    """
    # Convert to numpy for ranking with proper tie handling
    pred_np = predictions.detach().cpu().numpy()
    target_np = targets.detach().cpu().numpy()
    
    # Use scipy's rankdata for proper tie handling
    from scipy.stats import rankdata
    pred_ranks = torch.tensor(rankdata(pred_np), dtype=predictions.dtype, device=predictions.device)
    target_ranks = torch.tensor(rankdata(target_np), dtype=targets.dtype, device=targets.device)
    
    # Calculate the mean of ranks
    pred_ranks_mean = torch.mean(pred_ranks)
    target_ranks_mean = torch.mean(target_ranks)
    
    # Calculate the numerator (covariance of ranks)
    numerator = torch.sum((pred_ranks - pred_ranks_mean) * (target_ranks - target_ranks_mean))
    
    # Calculate the denominator (product of standard deviations of ranks)
    pred_deviation = torch.sqrt(torch.sum((pred_ranks - pred_ranks_mean) ** 2))
    target_deviation = torch.sqrt(torch.sum((target_ranks - target_ranks_mean) ** 2))
    denominator = pred_deviation * target_deviation
    
    # Calculate Spearman correlation
    corr = numerator / denominator
    
    return corr


def differentiable_rank_correlation(predictions, targets, temperature=0.01):
    """
    Differentiable approximation of Spearman correlation.
    
    Args:
        predictions: tensor of predicted values
        targets: tensor of target values
        temperature: controls the smoothness of the approximation
        
    Returns:
        Approximate Spearman correlation coefficient
    """
    n = predictions.size(0)
    
    # Create pairwise difference matrices
    pred_diff = predictions.unsqueeze(1) - predictions.unsqueeze(0)
    targ_diff = targets.unsqueeze(1) - targets.unsqueeze(0)
    
    # Apply sigmoid to get soft comparisons
    pred_sign = torch.sigmoid(pred_diff / temperature)
    targ_sign = torch.sigmoid(targ_diff / temperature)
    
    # Sum to get approximate ranks (adding 1 for 1-based ranking)
    pred_ranks = torch.sum(pred_sign, dim=1) + 1
    target_ranks = torch.sum(targ_sign, dim=1) + 1
    
    # Calculate the mean of ranks
    pred_ranks_mean = torch.mean(pred_ranks)
    target_ranks_mean = torch.mean(target_ranks)
    
    # Calculate the numerator (covariance of ranks)
    numerator = torch.sum((pred_ranks - pred_ranks_mean) * (target_ranks - target_ranks_mean))
    
    # Calculate the denominator (product of standard deviations of ranks)
    pred_deviation = torch.sqrt(torch.sum((pred_ranks - pred_ranks_mean) ** 2))
    target_deviation = torch.sqrt(torch.sum((target_ranks - target_ranks_mean) ** 2))
    denominator = pred_deviation * target_deviation
    
    # Calculate correlation
    corr = numerator / denominator
    
    return corr


class SpearmanLoss(torch.nn.Module):
    def __init__(self, differentiable=True, temperature=0.01):
        super(SpearmanLoss, self).__init__()
        self.differentiable = differentiable
        self.temperature = temperature
        
    def forward(self, predictions, targets):
        # Calculate Spearman correlation
        if self.differentiable:
            corr = differentiable_rank_correlation(predictions, targets, self.temperature)
        else:
            corr = spearman_correlation(predictions, targets)
        
        # Since optimization typically minimizes a loss,
        # we return negative correlation (we want to maximize correlation)
        return -corr


# Test the implementation
if __name__ == "__main__":
    # Create some test data
    pred = torch.tensor([1.0, 2.0, 9.0, 4.0, 5.0])
    target = torch.tensor([5.0, 6.0, 7.0, 8.0, 7.0])
    pred.requires_grad = True
    target.requires_grad = True
    
    # Calculate Spearman correlation (non-differentiable, correct)
    corr = spearman_correlation(pred.detach(), target.detach())
    print(f"Spearman correlation (corrected): {corr.item()}")
    
    # Test the differentiable version
    diff_corr = differentiable_rank_correlation(pred, target)
    diff_corr.backward(retain_graph=True)
    # print("pred grad:", pred.grad)
    print(f"Differentiable Spearman correlation: {diff_corr.item()}")
    
    # Test the loss function (non-differentiable version)
    loss_fn = SpearmanLoss(differentiable=False)
    loss = loss_fn(pred, target)
    print(f"Spearman loss (non-differentiable): {loss.item()}")
    
    # Test the loss function (differentiable version)
    diff_loss_fn = SpearmanLoss(differentiable=True)
    diff_loss = diff_loss_fn(pred, target)
    print(f"Spearman loss (differentiable): {diff_loss.item()}")
    
    # Compare with scipy implementation for validation
    try:
        scipy_corr, p = spearmanr(pred.detach().numpy(), target.detach().numpy())
        print(f"SciPy Spearman correlation: {scipy_corr}, p-value: {p}")
        print(f"Difference (corrected vs scipy): {abs(corr.item() - scipy_corr)}")
        print(f"Difference (differentiable vs scipy): {abs(diff_corr.item() - scipy_corr)}")
    except ImportError:
        print("SciPy not available for comparison")
import os
import sys

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.logger import logger


def compare_tensors(
    a: torch.Tensor,
    b: torch.Tensor,
    tol: float = 1e-6,
    verbose: bool = False
) -> int:
    """
    Compare two PyTorch tensors and count the number of elements whose absolute difference
    exceeds the given tolerance.

    Args:
        a (torch.Tensor): The first tensor to compare.
        b (torch.Tensor): The second tensor to compare.
        tol (float, optional): The absolute tolerance threshold. Defaults to 1e-6.
        verbose (bool, optional): If True, print the indices and values of differing elements. Defaults to False.

    Returns:
        int: The number of elements where abs(a - b) > tol.

    Raises:
        ValueError: If the shapes of the input tensors do not match.
    """
    # Check if tensor shapes are the same
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    # Create a boolean mask where differences exceed the tolerance
    diff_mask = (a - b).abs() > tol

    # Count the number of differing elements
    diff_count = int(diff_mask.sum().item())

    # If verbose, log details of differing elements
    if verbose and diff_count > 0:
        indices = torch.nonzero(diff_mask, as_tuple=False)
        for idx in indices:
            i, j = idx[0].item(), idx[1].item()
            logger.info(
                f"diff at {i},{j}: "
                f"{a[i, j].item():.6f} - {b[i, j].item():.6f} = "
                f"{(a[i, j] - b[i, j]).item():.6f}"
            )

    return diff_count


if __name__ == "__main__":
    # example:
    # a = torch.load("/path/to/a.pt")
    # b = torch.load("/path/to/b.pt")
    # diff_count = compare_tensors(a, b)
    # print(f"diff count: {diff_count}")
    pass

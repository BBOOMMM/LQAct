from math import sqrt
import torch
from torch import Tensor
from .qr import qr
from .utils import scaled_matmul

def energy_randomized_qb(
    A: Tensor,
    rank: int,
    niter: int = 0,
    test_matrix: str = "subs",
    left: bool | None = None,
    energy_threshold: float | None = 0.95,
) -> tuple[Tensor, Tensor]:
    """
    Randomized QB decomposition with optional Adaptive Rank based on energy threshold.
    
    Args:
        A (Tensor): (*, m, n)
        rank (int): The maximum rank (oversampling limit). If threshold is None, this is the fixed rank.
        niter (int): Number of power iterations.
        test_matrix (str): 'gauss' or 'subs'.
        left (bool|None): Decomposition order.
        threshold (float|None): Energy retention threshold (e.g., 0.9). 
                                If provided, the output rank will be dynamic (<= rank).

    Returns:
        (Q, B): Q (*, m, r), B (*, r, n) where r <= rank.
    """
    assert isinstance(A, Tensor)
    assert isinstance(rank, int)
    assert isinstance(niter, int)
    assert isinstance(test_matrix, str)
    assert isinstance(left, bool | None)

    m, n = A.shape[-2], A.shape[-1]

    if (left is True) or (left is None and m <= n):
        k = min(m, n, rank)

        if test_matrix == "gauss":
            Ohm = torch.randn(*A.shape[:-2], n, k, dtype=A.dtype, device=A.device)
        elif test_matrix == "subs":
            idx = torch.randperm(m)[:k]
            Ohm = A[..., idx, :].mT
        else:
            raise ValueError("Invalid value of `test_matrix`.")
        
        # Sample column space of A with Ohm matrix, and generate the approximate orthonormal basis Q
        Y = (A @ Ohm).div_(sqrt(n))  # (*, m, k)
        for _ in range(niter):
            Q, _ = qr(Y)  # (*, m, k)
            Ohm = scaled_matmul(A.mT, Q)  # (*, n, k)
            Y = (A @ Ohm).div_(sqrt(n))  # (*, m, k)
        Q, _ = qr(Y)  # (*, m, k)

        # Compute projected B
        B = scaled_matmul(Q.mT, A) # (*, k, n)

        # Add energy threshold
        if energy_threshold is not None and 0.0 < energy_threshold < 1.0:
            # 2.1 计算 Gram 矩阵: G = B * B^T
            # 为了数值稳定性，建议在 FP32 下计算 Gram 矩阵
            # B: (*, k, n) -> G: (*, k, k)
            # 这一步非常快，因为 k 很小 (如 32, 64)
            B_float = B.to(dtype=torch.float32)
            G = B_float @ B_float.mT.to(dtype=torch.float32)
            
            # 2.2 特征分解 (Eigendecomposition)
            # eigh 针对对称矩阵优化，比 svd 快得多
            # L: 特征值 (升序), V_rot: 特征向量 (列向量对应特征值)
            L, V_rot = torch.linalg.eigh(G.to(dtype=torch.float32))
            
            # 2.3 翻转顺序 (从升序变为降序)
            # L 是能量 (奇异值的平方)
            energy = L.flip(-1) 
            # 确保非负 (数值误差可能导致微小负数)
            energy = torch.nn.functional.relu(energy)
            
            # V_rot 的列需要随之翻转，使其对应最大的特征值
            V_rot = V_rot.flip(-1)

            # 2.4 计算需要保留的秩
            total_energy = energy.sum(dim=-1, keepdim=True)
            cumulative_energy = torch.cumsum(energy, dim=-1)
            ratios = cumulative_energy / (total_energy + 1e-6)
            
            mask = ratios < energy_threshold
            required_ranks = mask.sum(dim=-1) + 1
            
            # 在 Batch 维度取最大值，保持 Tensor 形状规整
            target_rank = required_ranks.max().item()
            target_rank = min(target_rank, k)

            # 2.5 【核心步骤】旋转 (Rotation) 并 截断 (Truncation)
            # 如果只计算 SVD 但不旋转 Q 和 B，截断是错误的。
            # 我们需要把 Q 和 B 变换到 SVD 的基底上。
            
            if target_rank < k:
                # 截取前 target_rank 个特征向量
                # V_keep: (*, k, target_rank)
                V_keep = V_rot[..., :target_rank].to(dtype=B.dtype)
                
                # 更新 Q: Q_new = Q @ V_keep
                # Q 的列现在变成了左奇异向量，且按重要性排序
                Q = Q @ V_keep
                
                # 更新 B: B_new = V_keep.T @ B
                # B 的行现在变成了右奇异向量 * 奇异值
                B = V_keep.mT @ B
                
                # print(f'Adaptive Rank: {k} -> {target_rank}')

        return Q, B

    elif (left is False) or (left is None and m > n):
        QT, BT = energy_randomized_qb(
            A.mT, 
            rank, 
            niter=niter, 
            test_matrix=test_matrix, 
            left=True, 
            threshold=energy_threshold # 传递阈值
        )
        return BT.mT, QT.mT

    else:
        raise ValueError("Invalid value of `left`.")
    
    
def energy_qb_reconstruct(
    Q: Tensor,
    B: Tensor,
) -> Tensor:
    """
    Reconstruction of QB decomposition.

    Args:
        Q (Tensor):
            &#45; with shape `(*, m, k)`.
        B (Tensor):
            &#45; with shape `(*, k, n)`.

    Returns:
        Tensor:
            &#45; with shape `(*, m, n)`.
    """
    assert isinstance(Q, Tensor)
    assert isinstance(B, Tensor)

    return Q @ B
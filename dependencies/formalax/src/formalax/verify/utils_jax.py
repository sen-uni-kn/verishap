# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
import torch
from auto_LiRPA import BoundedTensor, PerturbationLpNorm

__all__ = ["construct_bounded_tensor"]


def construct_bounded_tensor(in_lb: torch.Tensor, in_ub: torch.Tensor) -> BoundedTensor:
    input_domain = PerturbationLpNorm(x_L=in_lb, x_U=in_ub)
    # midpoint = (in_ub + in_lb) / 2
    return BoundedTensor(in_lb, ptb=input_domain)


class PerturbationLpNorm(Perturbation):
    """Perturbation constrained by the L_p norm."""

    def __init__(self, eps=0, norm=np.inf, x_L=None, x_U=None, eps_min=0):
        self.eps = eps
        self.eps_min = eps_min
        self.norm = norm
        self.dual_norm = (
            1 if (norm == np.inf) else (np.float64(1.0) / (1 - 1.0 / self.norm))
        )
        self.x_L = x_L
        self.x_U = x_U
        self.sparse = False

    def get_input_bounds(self, x, A):
        if self.sparse:
            if self.x_L_sparse.shape[-1] == A.shape[-1]:
                x_L, x_U = self.x_L_sparse, self.x_U_sparse
            else:
                # In backward mode, A is not sparse
                x_L, x_U = self.x_L, self.x_U
        else:
            x_L = x - self.eps if self.x_L is None else self.x_L
            x_U = x + self.eps if self.x_U is None else self.x_U
        return x_L, x_U

    def concretize_matrix(self, x, A, sign):
        # If A is an identity matrix, we will handle specially.
        if not isinstance(A, eyeC):
            # A has (Batch, spec, *input_size). For intermediate neurons, spec is *neuron_size.
            A = A.reshape(A.shape[0], A.shape[1], -1)

        if self.norm == np.inf:
            # For Linfinity distortion, when an upper and lower bound is given, we use them instead of eps.
            x_L, x_U = self.get_input_bounds(x, A)
            x_ub = x_U.reshape(x_U.shape[0], -1, 1)
            x_lb = x_L.reshape(x_L.shape[0], -1, 1)
            # Find the uppwer and lower bound similarly to IBP.
            center = (x_ub + x_lb) / 2.0
            diff = (x_ub - x_lb) / 2.0
            if not isinstance(A, eyeC):
                bound = A.matmul(center) + sign * A.abs().matmul(diff)
            else:
                # A is an identity matrix. No need to do this matmul.
                bound = center + sign * diff
        else:
            x = x.reshape(x.shape[0], -1, 1)
            if not isinstance(A, eyeC):
                # Find the upper and lower bounds via dual norm.
                deviation = A.norm(self.dual_norm, -1) * self.eps
                bound = A.matmul(x) + sign * deviation.unsqueeze(-1)
            else:
                # A is an identity matrix. Its norm is all 1.
                bound = x + sign * self.eps
        bound = bound.squeeze(-1)
        return bound

    def concretize_patches(self, x, A, sign):
        if self.norm == np.inf:
            x_L, x_U = self.get_input_bounds(x, A)

            # Here we should not reshape
            # Find the uppwer and lower bound similarly to IBP.
            center = (x_U + x_L) / 2.0
            diff = (x_U - x_L) / 2.0

            if not A.identity == 1:
                bound = A.matmul(center)
                bound_diff = A.matmul(diff, patch_abs=True)

                if sign == 1:
                    bound += bound_diff
                elif sign == -1:
                    bound -= bound_diff
                else:
                    raise ValueError("Unsupported Sign")
            else:
                # A is an identity matrix. No need to do this matmul.
                bound = center + sign * diff
            return bound
        else:  # Lp norm
            input_shape = x.shape
            if not A.identity:
                # Find the upper and lower bounds via dual norm.
                # matrix has shape
                # (batch_size, out_c * out_h * out_w, input_c, input_h, input_w)
                # or (batch_size, unstable_size, input_c, input_h, input_w)
                matrix = patches_to_matrix(
                    A.patches,
                    input_shape,
                    A.stride,
                    A.padding,
                    A.output_shape,
                    A.unstable_idx,
                )
                # Note that we should avoid reshape the matrix.
                # Due to padding, matrix cannot be reshaped without copying.
                deviation = matrix.norm(p=self.dual_norm, dim=(-3, -2, -1)) * self.eps
                # Bound has shape (batch, out_c * out_h * out_w) or (batch, unstable_size).
                bound = torch.einsum("bschw,bchw->bs", matrix, x) + sign * deviation
                if A.unstable_idx is None:
                    # Reshape to (batch, out_c, out_h, out_w).
                    bound = bound.view(
                        matrix.size(0),
                        A.patches.size(0),
                        A.patches.size(2),
                        A.patches.size(3),
                    )
            else:
                # A is an identity matrix. Its norm is all 1.
                bound = x + sign * self.eps
            return bound

    def concretize(self, x, A, sign=-1, aux=None):
        """Given an variable x and its bound matrix A, compute worst case bound according to Lp norm."""
        if A is None:
            return None
        if isinstance(A, eyeC) or isinstance(A, torch.Tensor):
            return self.concretize_matrix(x, A, sign)
        elif isinstance(A, Patches):
            return self.concretize_patches(x, A, sign)
        else:
            raise NotImplementedError()

    def init_sparse_linf(self, x, x_L, x_U):
        """Sparse Linf perturbation where only a few dimensions are actually perturbed"""
        self.sparse = True
        batch_size = x_L.shape[0]
        perturbed = (x_U > x_L).int()
        logger.debug(f"Perturbed: {perturbed.sum()}")
        lb = ub = x_L * (1 - perturbed)  # x_L=x_U holds when perturbed=0
        perturbed = perturbed.view(batch_size, -1)
        index = torch.cumsum(perturbed, dim=-1)
        dim = max(perturbed.view(batch_size, -1).sum(dim=-1).max(), 1)
        self.x_L_sparse = torch.zeros(batch_size, dim + 1).to(x_L)
        self.x_L_sparse.scatter_(
            dim=-1, index=index, src=(x_L - lb).view(batch_size, -1), reduce="add"
        )
        self.x_U_sparse = torch.zeros(batch_size, dim + 1).to(x_U)
        self.x_U_sparse.scatter_(
            dim=-1, index=index, src=(x_U - ub).view(batch_size, -1), reduce="add"
        )
        self.x_L_sparse, self.x_U_sparse = (
            self.x_L_sparse[:, 1:],
            self.x_U_sparse[:, 1:],
        )
        lw = torch.zeros(batch_size, dim + 1, perturbed.shape[-1], device=x.device)
        perturbed = perturbed.to(torch.get_default_dtype())
        lw.scatter_(dim=1, index=index.unsqueeze(1), src=perturbed.unsqueeze(1))
        lw = uw = lw[:, 1:, :].view(batch_size, dim, *x.shape[1:])
        print(f"Using Linf sparse perturbation. Perturbed dimensions: {dim}.")
        print(f"Avg perturbation: {(self.x_U_sparse - self.x_L_sparse).mean()}")
        return LinearBound(lw, lb, uw, ub, x_L, x_U), x, None

    def init(self, x, aux=None, forward=False):
        self.sparse = False
        if self.norm == np.inf:
            x_L = x - self.eps if self.x_L is None else self.x_L
            x_U = x + self.eps if self.x_U is None else self.x_U
        else:
            if int(os.environ.get("AUTOLIRPA_L2_DEBUG", 0)) == 1:
                # FIXME Experimental code. Need to change the IBP code also.
                x_L = x - self.eps if self.x_L is None else self.x_L
                x_U = x + self.eps if self.x_U is None else self.x_U
            else:
                # FIXME This causes confusing lower bound and upper bound
                # For other norms, we pass in the BoundedTensor objects directly.
                x_L = x_U = x
        if not forward:
            return LinearBound(None, None, None, None, x_L, x_U), x, None
        if (
            self.norm == np.inf
            and x_L.numel() > 1
            and (x_L == x_U).sum() > 0.5 * x_L.numel()
        ):
            return self.init_sparse_linf(x, x_L, x_U)

        batch_size = x.shape[0]
        dim = x.reshape(batch_size, -1).shape[-1]
        lb = ub = torch.zeros_like(x)
        eye = torch.eye(dim).to(x).expand(batch_size, dim, dim)
        lw = uw = eye.reshape(batch_size, dim, *x.shape[1:])
        return LinearBound(lw, lb, uw, ub, x_L, x_U), x, None

    def __repr__(self):
        if self.norm == np.inf:
            if self.x_L is None and self.x_U is None:
                return f"PerturbationLpNorm(norm=inf, eps={self.eps})"
            else:
                return f"PerturbationLpNorm(norm=inf, eps={self.eps}, x_L={self.x_L}, x_U={self.x_U})"
        else:
            return f"PerturbationLpNorm(norm={self.norm}, eps={self.eps})"

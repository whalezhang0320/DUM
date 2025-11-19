# Copyright (c) 2025
#
# Conditional flow-based density model reused from the UCR_NO_OOD_25_10_19
# project, with minor packaging adjustments for EDAC integration.

from __future__ import annotations

import torch
import torch.nn as nn


class ConditionalNet(nn.Module):
    """Generate affine parameters given split input and conditioning context."""

    def __init__(
            self,
            in_features: int,
            out_features: int,
            context_features: int,
            hidden_features: int = 256,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features + context_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x_context = torch.cat([x, context], dim=-1)
        return self.net(x_context)


class AffineCouplingLayer(nn.Module):
    """RealNVP-style affine coupling layer with optional input flip."""

    def __init__(self, dim: int, context_dim: int, flip: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.flip = flip

        dim1 = (dim + 1) // 2
        dim2 = dim // 2

        if self.flip:
            in_dim, out_dim = dim2, dim1
        else:
            in_dim, out_dim = dim1, dim2

        self.s_t_net = ConditionalNet(in_dim, out_dim * 2, context_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor):
        if self.flip:
            x_to_transform, x_for_params = x.chunk(2, dim=-1)
        else:
            x_for_params, x_to_transform = x.chunk(2, dim=-1)

        s_t = self.s_t_net(x_for_params, context)
        s, t = s_t.chunk(2, dim=-1)
        s = torch.tanh(s)
        y_transformed = x_to_transform * torch.exp(s) + t

        if self.flip:
            y = torch.cat([y_transformed, x_for_params], dim=-1)
        else:
            y = torch.cat([x_for_params, y_transformed], dim=-1)

        log_det_jacobian = s.sum(dim=-1)
        return y, log_det_jacobian

    def reverse(self, y: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if self.flip:
            y_transformed, y_for_params = y.chunk(2, dim=-1)
        else:
            y_for_params, y_transformed = y.chunk(2, dim=-1)

        s_t = self.s_t_net(y_for_params, context)
        s, t = s_t.chunk(2, dim=-1)
        s = torch.tanh(s)

        x_transformed = (y_transformed - t) * torch.exp(-s)

        if self.flip:
            x = torch.cat([x_transformed, y_for_params], dim=-1)
        else:
            x = torch.cat([y_for_params, x_transformed], dim=-1)

        return x


class ConditionalFlow(nn.Module):
    """Conditional normalising flow for modelling p(a | s)."""

    def __init__(self, action_dim: int, state_dim: int, num_layers: int = 4) -> None:
        super().__init__()
        if action_dim <= 1:
            raise ValueError("action_dim must be > 1 for AffineCouplingLayer")

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(AffineCouplingLayer(action_dim, state_dim, flip=i % 2 == 1))

        self.register_buffer('base_loc', torch.zeros(action_dim))
        self.register_buffer('base_cov', torch.eye(action_dim))

    @property
    def base_dist(self) -> torch.distributions.MultivariateNormal:
        return torch.distributions.MultivariateNormal(self.base_loc, self.base_cov)

    def log_prob(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        log_det_sum = torch.zeros(x.shape[0], device=x.device)
        z = x
        for layer in self.layers:
            z, log_det = layer(z, context)
            log_det_sum += log_det

        base_log_prob = self.base_dist.log_prob(z)
        return base_log_prob + log_det_sum

    def sample(self, num_samples: int, context: torch.Tensor) -> torch.Tensor:
        z = self.base_dist.sample((num_samples,))

        if context.shape[0] != num_samples:
            if context.shape[0] == 1:
                context = context.repeat(num_samples, 1)
            else:
                assert context.shape[0] == num_samples

        x = z
        for layer in reversed(self.layers):
            x = layer.reverse(x, context)
        return x


class Discriminator(nn.Module):
    """Adversarial discriminator used during flow-GAN training (optional)."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 750) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        state_action = torch.cat([state, action], dim=-1)
        return self.net(state_action)


# 文件名: flow_gan.py
# 描述: 定义条件标准化流模型 (生成器) 和判别器

import torch
import torch.nn as nn
import numpy as np

class ConditionalNet(nn.Module):
    """
    一个简单的条件网络，用于在流模型中生成 s 和 t 参数。
    它接收条件 c (state) 和输入 x 的一部分，输出变换参数。
    """

    def __init__(self, in_features, out_features, context_features, hidden_features=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features + context_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, x, context):
        x_context = torch.cat([x, context], dim=-1)
        return self.net(x_context)

class AffineCouplingLayer(nn.Module):
    """
    仿射耦合层 (Affine Coupling Layer)，RealNVP的核心组件。
    它将输入 x 分为两部分，并对其中一部分进行仿射变换，变换的参数由另一部分和条件 c 决定。
    """

    def __init__(self, dim, context_dim, flip=False):
        super().__init__()
        self.dim = dim
        self.flip = flip

        # --- 建议的修复 ---
        # 明确计算分割后的维度
        dim1 = (dim + 1) // 2  # 向上取整
        dim2 = dim // 2  # 向下取整

        if self.flip:
            # x_for_params 是后半部分 (dim2), x_to_transform 是前半部分 (dim1)
            in_dim, out_dim = dim2, dim1
        else:
            # x_for_params 是前半部分 (dim1), x_to_transform 是后半部分 (dim2)
            in_dim, out_dim = dim1, dim2

        self.s_t_net = ConditionalNet(in_dim, out_dim * 2, context_dim)

    def forward(self, x, context):
        # 根据 flip 决定哪部分被变换，哪部分用于生成参数
        if self.flip:
            x_to_transform, x_for_params = x.chunk(2, dim=-1)
        else:
            x_for_params, x_to_transform = x.chunk(2, dim=-1)

        s_t = self.s_t_net(x_for_params, context)
        # 将输出分割为 scale 和 translation
        s, t = s_t.chunk(2, dim=-1)

        # 为了稳定性，使用 tanh
        s = torch.tanh(s)

        # 应用变换: y_transformed = x_transformed * exp(s) + t
        y_transformed = x_to_transform * torch.exp(s) + t

        if self.flip:
            y = torch.cat([y_transformed, x_for_params], dim=-1)
        else:
            y = torch.cat([x_for_params, y_transformed], dim=-1)

        # 对数雅可比行列式就是 s 的和
        log_det_jacobian = s.sum(dim=-1)

        return y, log_det_jacobian

    def reverse(self, y, context):
        if self.flip:
            y_transformed, y_for_params = y.chunk(2, dim=-1)
        else:
            y_for_params, y_transformed = y.chunk(2, dim=-1)

        s_t = self.s_t_net(y_for_params, context)
        s, t = s_t.chunk(2, dim=-1)
        s = torch.tanh(s)

        # 反向变换: x_transformed = (y_transformed - t) * exp(-s)
        x_transformed = (y_transformed - t) * torch.exp(-s)

        if self.flip:
            x = torch.cat([x_transformed, y_for_params], dim=-1)
        else:
            x = torch.cat([y_for_params, x_transformed], dim=-1)

        return x

class ConditionalFlow(nn.Module):
    """
    条件标准化流模型 (Generator)。
    """

    def __init__(self, action_dim, state_dim, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList()
        if action_dim <= 1:
            raise ValueError("action_dim must be > 1 for AffineCouplingLayer")

        for i in range(num_layers):
            self.layers.append(AffineCouplingLayer(action_dim, state_dim, flip=i % 2 == 1))

        # --- 性能修复 ---
        # 将基础分布的均值和协方差矩阵注册为 buffer
        # 这样调用 .to(device) 时它们会自动移动
        self.register_buffer('base_loc', torch.zeros(action_dim))
        self.register_buffer('base_cov', torch.eye(action_dim))
        # -----------------

    @property
    def base_dist(self):
        # 每次访问时动态创建分布，确保它在正确的设备上
        return torch.distributions.MultivariateNormal(self.base_loc, self.base_cov)

    def log_prob(self, x, context):
        log_det_sum = torch.zeros(x.shape[0], device=x.device)
        z = x
        for layer in self.layers:
            z, log_det = layer(z, context)
            log_det_sum += log_det

        # --- 性能修复 ---
        # 现在 z 不需要移动到 CPU，因为 base_dist 已经在 z.device 上了
        base_log_prob = self.base_dist.log_prob(z)
        # -----------------
        return base_log_prob + log_det_sum

    def sample(self, num_samples, context):
        # --- 性能修复 ---
        # 从在正确设备上的分布采样
        z = self.base_dist.sample((num_samples,))
        # -----------------

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
    """
    判别器 (Discriminator)。
    一个简单的分类器，用于判断 (state, action) 对是真实的还是生成的。
    """

    def __init__(self, state_dim, action_dim, hidden_dim=750):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 输出一个 logit
        )

    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=-1)
        return self.net(state_action)
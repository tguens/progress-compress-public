"""
Progress & Compress models: CNN backbone(s), adaptor(s), and end-to-end model.
"""

from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn

from typing import Literal
from torch.distributions.categorical import Categorical


# -------------------------
#       Backbones
# --------------------------
class _CNNBackbone(nn.Module):
    def _conv_block(
        self, in_channels: int, out_channels: int, use_batchnorm: bool = False
    ) -> nn.Sequential:
        # A little helper
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
        )

    def __init__(self, use_batchnorm: bool):
        super().__init__()
        # Assume the input dim is 1x28x28. The heads live in `CNNModel` (one per task).
        self.use_batchnorm = (
            use_batchnorm  # P&C's ref (Vinyals et al., 2016) uses batchnorm
        )
        self.blocks = nn.ModuleList(
            [self._conv_block(1, 64, use_batchnorm)]
            + [self._conv_block(64, 64, use_batchnorm) for _ in range(3)]
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.feature_dim = 64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Features [B, 64] of this column alone."""
        for block in self.blocks:
            x = block(x)
            x = self.relu(x)
            x = self.pool(x)

        return x.flatten(1)  # Keep [B,...], flattens the rest


# Reference: https://github.com/vwxyzjn/cleanrl/blob/fe8d8a03c41a7ef5b523e2e354bd01c363e786bb/cleanrl/ppo_atari_envpool.py#L123
# AI-generated class
class _NatureCNN(nn.Module):
    """CleanRL's Nature CNN with the layers kept apart,
    so that we can add the adaptor output before each ReLU.

    Reference
    ---------
    https://github.com/vwxyzjn/cleanrl/blob/fe8d8a03c41a7ef5b523e2e354bd01c363e786bb/cleanrl/ppo_atari_envpool.py#L123
    """

    @staticmethod
    def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        # From CleanRL
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def __init__(self):
        super().__init__()
        # fmt:off
        self.conv1 = self.layer_init(nn.Conv2d(4, 32, 8, stride=4))   # [B, 4, 84, 84] -> [B, 32, 20, 20]
        self.conv2 = self.layer_init(nn.Conv2d(32, 64, 4, stride=2))  # [B, 32, 20, 20] -> [B, 64, 9, 9]
        self.conv3 = self.layer_init(nn.Conv2d(64, 64, 3, stride=1))  # [B, 64, 9, 9] -> [B, 64, 7, 7]
        self.fc = self.layer_init(nn.Linear(64 * 7 * 7, 512))         # [B, 3136] -> [B, 512]
        # fmt:on
        self.feature_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Features [B, 512] of this column alone. The heads live in the agent."""
        x = x / 255.0  # uint8 frames -> [0, 1]
        h = torch.relu(self.conv1(x))
        h = torch.relu(self.conv2(h))
        h = torch.relu(self.conv3(h))
        return torch.relu(self.fc(h.flatten(1)))  # [B, 512]


# -----------------------
#       Adaptors
# -----------------------
class _CNNAdaptor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        adaptor_init: Literal["uniform", "gaussian"],
        u_stride: int = 1,
        u_kernel_size: int = 1,
    ):
        super().__init__()
        if adaptor_init == "uniform":
            self.alpha = nn.Parameter(torch.empty(output_dim).uniform_(0, 0.1))
        elif adaptor_init == "gaussian":
            self.alpha = nn.Parameter(torch.randn(output_dim) * 0.1)
        else:
            raise ValueError(f"Invalid adaptor initialization: {adaptor_init}")
        # V is a 1x1 projection that keeps the KB width; U has the kernel and stride
        # of the active layer W_i it feeds, so the sum with W_i h^{active} is well
        # defined. Omniglot: 1x1 stride 1 (its convs keep the spatial size).
        # Atari: 4x4 stride 2 for conv2 (32 -> 64 channels), 3x3 for conv3.
        self.U = nn.Conv2d(
            input_dim,
            output_dim,
            kernel_size=u_kernel_size,
            stride=u_stride,
            padding=0,
            bias=False,
        )  # No bias for the U layer
        self.V = nn.Conv2d(
            input_dim, input_dim, kernel_size=1, stride=1, padding=0, bias=True
        )  # Add the bias -> 'c_i')

    def forward(self, h_kb):
        # \alpha_i \odot (U_i @ \sigma(V_i h^{KB}_{i-1} + c_i))
        # alpha is element-wise (Hadamard), U is a matmul

        # \sigma(V_i h^{KB}_{i-1} + c_i)) -> [B, C_in, H, W], the shape of h_kb
        v = torch.relu(self.V(h_kb))

        # U_i * ... -> the shape of W_i h^{active}_{i-1}: [B, C_out, H', W']
        uv = self.U(v)

        # alpha \circ ... -> [B, 64, H, W]
        # alpha shape: [1, 64] -> [1, 64, 1, 1]
        alpha = self.alpha.view(1, -1, 1, 1)

        return alpha * uv  # [B, 64, H, W]


# AI-generated class
class _DenseAdaptor(nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, adaptor_init: Literal["uniform", "gaussian"]
    ):
        super().__init__()
        if adaptor_init == "uniform":
            self.alpha = nn.Parameter(torch.empty(out_dim).uniform_(0, 0.1))
        elif adaptor_init == "gaussian":
            self.alpha = nn.Parameter(torch.randn(out_dim) * 0.1)
        else:
            raise ValueError(f"Invalid adaptor initialization: {adaptor_init}")
        self.V = nn.Linear(in_dim, out_dim, bias=True)
        self.U = nn.Linear(out_dim, out_dim, bias=False)

    def forward(self, h_kb: torch.Tensor) -> torch.Tensor:
        # \alpha_i \odot (U_i @ \sigma(V_i h^{KB}_{i-1} + c_i))
        return self.alpha * self.U(torch.relu(self.V(h_kb)))  # [B, 512]


# -------------------------
#       Main Models
# --------------------------


# Omniglot model
class CNNModel(nn.Module):
    """Omniglot model: two CNN columns with adaptors and one head per task.

    Implementation note about the adaptor
    The CNN backbone would transform the input as follows
    - [B, 64, 28, 28] (after the first conv block)
    - -> [B, 64, 14, 14] (after MaxPool2d + second conv block)
    - -> [B, 64, 7, 7] (after MaxPool2d + third conv block)
    - -> [B, 64, 3, 3] (after MaxPool2d + fourth conv block)
    - -> [B, 64, 1, 1] (after last MaxPool2d)
    """

    # We make methods for both because we'll need to re-use it when training the model
    def _init_active_layers(self):
        self.active_layers = _CNNBackbone(use_batchnorm=self.use_batchnorm)

    def _init_adaptors(self):
        self.adaptors = nn.ModuleList(
            [
                _CNNAdaptor(
                    input_dim=self.adaptor_dim,
                    output_dim=self.adaptor_dim,
                    adaptor_init=self.adaptor_init,  # type: ignore
                )
                for _ in range(self.n_hidden_layers - 1)
            ]
        )

    def __init__(
        self,
        output_dim: int,
        use_batchnorm: bool,
        adaptor_init: Literal["uniform", "gaussian"],
        multi_head: bool,
    ):
        super().__init__()

        self.adaptor_dim = 64  # TODO: Make this more parametrized and dynamic
        self.output_dim = output_dim
        self.use_batchnorm = use_batchnorm
        self.adaptor_init = adaptor_init
        self.multi_head = multi_head

        self.kb_layers = _CNNBackbone(use_batchnorm=self.use_batchnorm)

        self.n_hidden_layers = len(self.kb_layers.blocks)

        self._init_active_layers()
        self._init_adaptors()

        # Heads, one per task (multi-head) or one shared (`"shared"`), for each column.
        # `prep_for_task` creates them, `switch` and `using` pick the current one.
        self.active_heads = nn.ModuleDict()
        self.kb_heads = nn.ModuleDict()
        self._head_key: str | None = None

    # ---- heads ----
    @property
    def active_head(self) -> nn.Module:
        assert self._head_key is not None
        return self.active_heads[self._head_key]

    @property
    def kb_head(self) -> nn.Module:
        assert self._head_key is not None
        return self.kb_heads[self._head_key]

    def switch(self, task_id) -> None:
        """Make the heads of `task_id` the current ones. The task must have been
        prepared with `prep_for_task` before; a silent new head would hide a bug."""
        key = str(task_id) if self.multi_head else "shared"
        if key not in self.kb_heads:
            raise KeyError(f"No heads for task {task_id}: call prep_for_task first")
        self._head_key = key

    # Context manager to swtich the linear head to a known existing task
    # with model.switch(previous_task): eval(model...)
    @contextmanager
    def using(self, task_id):
        """Switch to the heads of `task_id` for the block, then switch back."""
        previous = self._head_key
        self.switch(task_id)
        try:
            yield self
        finally:
            self._head_key = previous

    def prep_for_task(
        self,
        task_id,
        reinit: bool,
        copy_kb: bool,
        previous_kb: dict | None,
    ) -> None:
        """Set the columns up for a task, then switch to its heads:
        - `reinit`: a fresh active column and fresh adaptors (the paper re-initializes);
          the active head is fresh too.
        - `copy_kb`: the active backbone starts from the KB backbone `previous_kb`
          (a `kb_layers.state_dict()`), when given.
        - the KB head of the task is created if the task has none yet (every task in
          multi-head mode, the first task only in single-head mode)."""
        device = next(self.parameters()).device

        if reinit:
            self._init_active_layers()
            self._init_adaptors()

        if copy_kb and previous_kb is not None:
            self.active_layers.load_state_dict(previous_kb)
        key = str(task_id) if self.multi_head else "shared"

        if reinit or key not in self.active_heads:
            self.active_heads[key] = nn.Linear(
                self.kb_layers.feature_dim, self.output_dim
            )

        if key not in self.kb_heads:
            self.kb_heads[key] = nn.Linear(self.kb_layers.feature_dim, self.output_dim)

        self.to(device)
        self.switch(task_id)

    def parameters_for(self, mode: Literal["progress", "compress"]) -> list:
        """What a phase trains. Progress: the active column, its head and the
        adaptors. Compress: the KB and its current head."""
        if mode == "progress":
            return (
                list(self.active_layers.parameters())
                + list(self.active_head.parameters())
                + list(self.adaptors.parameters())
            )
        elif mode == "compress":
            return list(self.kb_layers.parameters()) + list(self.kb_head.parameters())
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def forward(
        self, x: torch.Tensor, mode: Literal["progress", "compress"]
    ) -> torch.Tensor:
        # We deconstruct the CNN backbone and have a less error prone implementation.
        # All the ReLUs are explicit
        # We opt for simplicity for now.
        assert mode in ("progress", "compress")
        # One column gets gradients, and only if the caller allows them:
        # an outer `torch.no_grad()` builds no graph.
        grad_active = mode == "progress" and torch.is_grad_enabled()
        grad_kb = mode == "compress" and torch.is_grad_enabled()

        # The KB is frozen during progress, its BatchNorm running statistics too
        kb_was_training = self.kb_layers.training
        if mode == "progress":
            self.kb_layers.eval()

        with torch.set_grad_enabled(grad_active):
            h_active = torch.relu(self.active_layers.blocks[0](x))  # type: ignore
            h_active = self.active_layers.pool(h_active)  # type: ignore

        with torch.set_grad_enabled(grad_kb):
            h_kb = torch.relu(self.kb_layers.blocks[0](x))  # type: ignore
            h_kb = self.kb_layers.pool(h_kb)  # type: ignore

        # Example shapes
        # i -> {0, 1, 2}
        # 3 adaptors (0 to 2) and 3 blocks (1 to 3)
        # 0th conv block used before; 4th last linear layer is used later on
        # Keep B and C shapes constant -> [B, 64, ..., ...], only the pooling affects the last two
        for i in range(self.n_hidden_layers - 1):
            with torch.set_grad_enabled(grad_active):
                # W_i * h_{i-1} + bi
                # Conv + optional BN, no relu
                proj_h_active = self.active_layers.blocks[i + 1](h_active)  # type: ignore

                # \alpha_i \odot (U_i @ \sigma(V_i h^{KB}_{i-1} + c_i))
                adapted_h_kb = self.adaptors[i](h_kb)

                # h_i^active = sigma(proj_h_active + adapted_h_kb)
                h_active = torch.relu(proj_h_active + adapted_h_kb)

            with torch.set_grad_enabled(grad_kb):
                # h_{i-1}^kb -> h_i^kb (no grad, KB is frozen during progress)
                h_kb = torch.relu(self.kb_layers.blocks[i + 1](h_kb))  # type: ignore

            # Apply the pooling after the transformations
            # Since the adaptor applies over channels, the last two dimensions do not matter
            # TODO: Could we perform the pooling before ? More efficient ?
            h_active = self.active_layers.pool(h_active)  # type: ignore
            h_kb = self.kb_layers.pool(h_kb)  # type: ignore

        # Flatten hidden state to [B, 64 * 1 * 1] + the head of the current task
        with torch.set_grad_enabled(grad_active):
            h_active = self.active_head(h_active.flatten(1))

        with torch.set_grad_enabled(grad_kb):
            h_kb = self.kb_head(h_kb.flatten(1))

        self.kb_layers.train(kb_was_training)
        return h_active, h_kb  # type: ignore


# Atari model
# AI-generated class
class PCAgent(nn.Module):
    """Two Nature CNN columns with adaptors, for PPO (progress) and distillation
    (compress). Same head API as `CNNModel`, keyed by game: one actor per game
    for each column, and a critic on the active column. A revisit of a game
    reuses its KB actor; the active column, its adaptors, actor and critic are
    fresh for every task (the paper re-initializes)."""

    def _init_active_layers(self):
        self.active_layers = _NatureCNN()

    def _init_adaptors(self):
        # One adaptor per layer after the first: conv2, conv3, fc.
        self.adaptors = nn.ModuleList(
            [
                _CNNAdaptor(
                    32,
                    64,
                    u_kernel_size=4,
                    u_stride=2,
                    adaptor_init=self.adaptor_init,  # type: ignore
                ),
                _CNNAdaptor(
                    64,
                    64,
                    u_kernel_size=3,
                    u_stride=1,
                    adaptor_init=self.adaptor_init,  # type: ignore
                ),
                _DenseAdaptor(64 * 7 * 7, 512, adaptor_init=self.adaptor_init),  # type: ignore
            ]
        )

    def __init__(self, adaptor_init: Literal["uniform", "gaussian"]):
        super().__init__()
        self.adaptor_init = adaptor_init
        self.kb_layers = _NatureCNN()
        self._init_active_layers()
        self._init_adaptors()
        self.active_heads = nn.ModuleDict()  # actor of the active column, per game
        self.kb_heads = nn.ModuleDict()  # actor of the KB, per game
        self.critics = nn.ModuleDict()  # critic of the active column, per game
        self._head_key: str | None = None

    # ---- heads ----
    @property
    def active_head(self) -> nn.Module:
        assert self._head_key is not None
        return self.active_heads[self._head_key]

    @property
    def kb_head(self) -> nn.Module:
        assert self._head_key is not None
        return self.kb_heads[self._head_key]

    @property
    def critic(self) -> nn.Module:
        assert self._head_key is not None
        return self.critics[self._head_key]

    def switch(self, game: str) -> None:
        """Make the heads of `game` the current ones; `prep_for_task` must have run."""
        if game not in self.kb_heads:
            raise KeyError(f"No heads for {game}: call prep_for_task first")
        self._head_key = game

    @contextmanager
    def using(self, game: str):
        """Switch to the heads of `game` for the block, then switch back."""
        previous = self._head_key
        self.switch(game)
        try:
            yield self
        finally:
            self._head_key = previous

    def prep_for_task(self, game: str, n_actions: int, reinit: bool = True) -> None:
        """A fresh active column, adaptors, actor and critic for the task (`reinit`,
        the paper's choice); the KB actor of the game is created on its first visit
        and kept for a revisit. Then switch to the heads of the game."""
        device = next(self.parameters()).device
        if reinit:
            self._init_active_layers()
            self._init_adaptors()
        self.active_heads[game] = _NatureCNN.layer_init(
            nn.Linear(512, n_actions), std=0.01
        )
        self.critics[game] = _NatureCNN.layer_init(nn.Linear(512, 1), std=1)
        if game not in self.kb_heads:
            self.kb_heads[game] = _NatureCNN.layer_init(
                nn.Linear(512, n_actions), std=0.01
            )
        self.to(device)
        self.switch(game)

    def parameters_for(self, mode: Literal["progress", "compress"]) -> list:
        """What a phase trains. Progress (PPO): the active column, its actor and
        critic, the adaptors. Compress: the KB and its current actor."""
        if mode == "progress":
            return (
                list(self.active_layers.parameters())
                + list(self.active_head.parameters())
                + list(self.critic.parameters())
                + list(self.adaptors.parameters())
            )
        elif mode == "compress":
            return list(self.kb_layers.parameters()) + list(self.kb_head.parameters())
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def columns(
        self, x: torch.Tensor, mode: Literal["progress", "compress"]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Hidden features [B, 512] of both columns. Only one column gets gradients,
        and only if the caller allows them (no graph under `torch.no_grad()`)."""
        assert mode in ("progress", "compress")
        grad_active = mode == "progress" and torch.is_grad_enabled()
        grad_kb = mode == "compress" and torch.is_grad_enabled()

        x = x / 255.0  # uint8 frames -> [0, 1]

        with torch.set_grad_enabled(grad_active):
            h_active = torch.relu(self.active_layers.conv1(x))  # [B, 32, 20, 20]
        with torch.set_grad_enabled(grad_kb):
            h_kb = torch.relu(self.kb_layers.conv1(x))  # [B, 32, 20, 20]

        for i, name in enumerate(["conv2", "conv3"]):
            with torch.set_grad_enabled(grad_active):
                # h_i^active = σ(W_i h_{i-1}^active + α_i ⊙ U_i σ(V_i h^{KB}_{i-1} + c_i))
                proj_h_active = getattr(self.active_layers, name)(h_active)
                adapted_kb = self.adaptors[i](h_kb)
                h_active = torch.relu(proj_h_active + adapted_kb)

            with torch.set_grad_enabled(grad_kb):
                h_kb = torch.relu(getattr(self.kb_layers, name)(h_kb))

        # conv2: [B, 64, 9, 9]; conv3: [B, 64, 7, 7]
        with torch.set_grad_enabled(grad_active):
            proj_h_active = self.active_layers.fc(h_active.flatten(1))
            h_active = torch.relu(
                proj_h_active + self.adaptors[2](h_kb.flatten(1))
            )  # [B, 512]
        with torch.set_grad_enabled(grad_kb):
            h_kb = torch.relu(self.kb_layers.fc(h_kb.flatten(1)))  # [B, 512]
        return h_active, h_kb

    def forward(
        self, x: torch.Tensor, mode: Literal["progress", "compress"]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Policy logits of both columns, (active, kb). Same contract as `CNNModel`,
        so `empirical_fisher` in `src/utils.py` works on this agent."""
        h_active, h_kb = self.columns(x, mode)
        with torch.set_grad_enabled(mode == "progress" and torch.is_grad_enabled()):
            y_active = self.active_head(h_active)
        with torch.set_grad_enabled(mode == "compress" and torch.is_grad_enabled()):
            y_kb = self.kb_head(h_kb)
        return y_active, y_kb

    def get_action_and_value(self, x: torch.Tensor, action=None):
        """PPO interface, as in CleanRL. Progress mode: the active column and the
        adaptors get gradients, the knowledge base is frozen."""
        h_active, _ = self.columns(x, mode="progress")
        probs = Categorical(logits=self.active_head(h_active))
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(h_active)


# Baseline
# AI-generated class
class SingleAgent(nn.Module):
    """One Nature CNN trained by PPO on every game in turn: the fine-tuning and
    online EWC baselines of the paper (its Figure 4). The network sits in
    `kb_layers` with one actor per game in `kb_heads` and a critic per task, so
    `empirical_fisher`, `mahalanobis_distance` and the evaluation of the
    knowledge base apply to it unchanged. Same head API as `PCAgent`."""

    def __init__(self):
        super().__init__()
        self.kb_layers = _NatureCNN()
        self.active_layers = self.kb_layers  # the same network; both names exist
        self.kb_heads = nn.ModuleDict()
        self.critics = nn.ModuleDict()
        self._head_key: str | None = None

    @property
    def kb_head(self) -> nn.Module:
        assert self._head_key is not None
        return self.kb_heads[self._head_key]

    @property
    def active_head(self) -> nn.Module:
        return self.kb_head

    @property
    def critic(self) -> nn.Module:
        assert self._head_key is not None
        return self.critics[self._head_key]

    def switch(self, game: str) -> None:
        if game not in self.kb_heads:
            raise KeyError(f"No heads for {game}: call prep_for_task first")
        self._head_key = game

    @contextmanager
    def using(self, game: str):
        previous = self._head_key
        self.switch(game)
        try:
            yield self
        finally:
            self._head_key = previous

    def prep_for_task(self, game: str, n_actions: int, reinit: bool = False) -> None:
        """A new critic for the task; the actor of the game is created on its first
        visit and kept for a revisit. The network itself is never re-initialized."""
        device = next(self.parameters()).device
        self.critics[game] = _NatureCNN.layer_init(nn.Linear(512, 1), std=1)
        if game not in self.kb_heads:
            self.kb_heads[game] = _NatureCNN.layer_init(
                nn.Linear(512, n_actions), std=0.01
            )
        self.to(device)
        self.switch(game)

    def parameters_for(self, mode: Literal["progress", "compress"]) -> list:
        """PPO trains the whole network; there is no compress phase."""
        return (
            list(self.kb_layers.parameters())
            + list(self.kb_head.parameters())
            + list(self.critic.parameters())
        )

    def forward(self, x: torch.Tensor, mode: Literal["progress", "compress"]):
        y = self.kb_head(self.kb_layers(x))
        return y, y

    def get_action_and_value(self, x: torch.Tensor, action=None):
        h = self.kb_layers(x)
        probs = Categorical(logits=self.kb_head(h))
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(h)


# AI-generated function
def load_teacher_state_dict(state_dict: dict, game: str) -> dict:
    """Map a teacher snapshot saved before the heads moved out of the backbones
    (`kb_layers.actor.*`, `active_layers.actor.*`, `active_layers.critic.*`) to the
    current layout. Snapshots in the current layout pass through unchanged."""
    renames = {
        "kb_layers.actor.": f"kb_heads.{game}.",
        "active_layers.actor.": f"active_heads.{game}.",
        "active_layers.critic.": f"critics.{game}.",
    }
    out = {}
    for k, v in state_dict.items():
        for old, new in renames.items():
            if k.startswith(old):
                k = new + k[len(old) :]
                break
        out[k] = v
    return out

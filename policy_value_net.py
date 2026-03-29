"""
An implementation of the policyValueNet in PyTorch.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Tuple


class Net(nn.Module):
    """policy-value network module"""
    def __init__(self, board_width, board_height):
        super(Net, self).__init__()

        self.board_width = board_width
        self.board_height = board_height

        # common layers (Conv-BN-ReLU)
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # action policy layers (policy network)
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.bn_act = nn.BatchNorm2d(4)
        self.act_fc1 = nn.Linear(4*board_width*board_height, board_width*board_height)

        # state value layers (value network)
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.bn_val = nn.BatchNorm2d(2)
        self.val_fc1 = nn.Linear(2*board_width*board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):
        # common layers
        x = F.relu(self.bn1(self.conv1(state_input)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # action policy layers
        x_act = F.relu(self.bn_act(self.act_conv1(x)))
        x_act = x_act.view(-1, 4*self.board_width*self.board_height)
        x_act = F.log_softmax(self.act_fc1(x_act), dim=1)

        # state value layers
        x_val = F.relu(self.bn_val(self.val_conv1(x)))
        x_val = x_val.view(-1, 2*self.board_width*self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = torch.tanh(self.val_fc2(x_val))
        return x_act, x_val


class PolicyValueNet():
    """policy-value network """
    def __init__(self, board_width: int, board_height: int,
                 model_file: Optional[str] = None, use_gpu: bool = False, base_lr: float = 2e-3):
        self.use_gpu = bool(use_gpu)
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4  # coef of l2 penalty (weight decay)
        self.device = torch.device(
            "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"
        )
        self.base_lr = float(base_lr)
        self._lr_multiplier = 1.0

        # the policy value net module
        self.policy_value_net = Net(board_width, board_height).to(self.device)
        self.optimizer = optim.Adam(
            self.policy_value_net.parameters(),
            lr=self.base_lr,
            weight_decay=self.l2_const,
        )

        if model_file:
            net_params = torch.load(model_file, map_location=self.device, weights_only=True)
            self.policy_value_net.load_state_dict(net_params)

    def set_lr_multiplier(self, lr_multiplier: float) -> None:
        self._lr_multiplier = float(lr_multiplier)
        eff_lr = self.base_lr * self._lr_multiplier
        for pg in self.optimizer.param_groups:
            pg["lr"] = eff_lr

    #def set_base_lr(self, base_lr: float) -> None:
    #    """
    #    Update base_lr (useful when the training pipeline changes learn_rate).
    #    """
    #    self.base_lr = float(base_lr)
    #    for pg in self.optimizer.param_groups:
    #        pg["lr"] = self.base_lr
    #    self.set_lr_multiplier(self._lr_multiplier)

    def current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    @torch.no_grad()
    def policy_value(self, state_batch: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        self.policy_value_net.eval()
        state_batch_np = np.ascontiguousarray(np.asarray(state_batch, dtype=np.float32))
        state_batch_t = torch.from_numpy(state_batch_np).to(self.device)
        log_act_probs, value = self.policy_value_net(state_batch_t)
        act_probs = torch.exp(log_act_probs).cpu().numpy()
        return act_probs, value.view(-1, 1).cpu().numpy()

    @torch.no_grad()
    def policy_value_fn(self, board) -> Tuple[List[Tuple[int, float]], float]:
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(
                -1, 4, self.board_width, self.board_height))
        self.policy_value_net.eval()
        state = torch.as_tensor(current_state, dtype=torch.float32, device=self.device)
        log_act_probs, value = self.policy_value_net(state)
        act_probs = torch.exp(log_act_probs).view(-1).cpu().numpy()
        value = float(value.view(-1).cpu().numpy()[0])
        action_priors = [(pos, float(act_probs[pos])) for pos in legal_positions]
        return action_priors, value

    def train_step(self, state_batch, mcts_probs, winner_batch) -> Tuple[float, float, float, float]:
        """perform a training step"""

        self.policy_value_net.train()
        state_batch_np = np.ascontiguousarray(np.asarray(state_batch, dtype=np.float32))
        mcts_probs_np = np.ascontiguousarray(np.asarray(mcts_probs, dtype=np.float32))
        winner_batch_np = np.ascontiguousarray(np.asarray(winner_batch, dtype=np.float32))

        state_batch_t = torch.from_numpy(state_batch_np).to(self.device)
        mcts_probs_t = torch.from_numpy(mcts_probs_np).to(self.device)
        winner_batch_t = torch.from_numpy(winner_batch_np).to(self.device)

        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward
        log_act_probs, value = self.policy_value_net(state_batch_t)
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        value_loss = F.mse_loss(value.view(-1), winner_batch_t)
        policy_loss = -torch.mean(torch.sum(mcts_probs_t * log_act_probs, 1))
        loss = value_loss + policy_loss
        # backward and optimize
        loss.backward()
        self.optimizer.step()
        # calc policy entropy, for monitoring only
        entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
                )
        return loss.item(), entropy.item(), policy_loss.item(), value_loss.item()

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        torch.save(net_params, model_file)

    def _move_optimizer_state_to_device(self) -> None:
        """Move optimizer state tensors to the same device as the model."""
        if self.device.type != "cuda":
            return
        for state in self.optimizer.state.values():
            for k, v in list(state.items()):
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)

    def save_checkpoint(self, checkpoint_file: str, *, extra_state: Optional[Dict[str, Any]] = None) -> None:
        """
        Save a training checkpoint (model + optimizer + extra state).

        This is different from save_model(): it includes optimizer state and
        arbitrary metadata needed to resume training.
        """
        ckpt = {
            "format_version": 1,
            "board_width": self.board_width,
            "board_height": self.board_height,
            "use_gpu": self.use_gpu,
            "device": str(self.device),
            "base_lr": self.base_lr,
            "lr_multiplier": self._lr_multiplier,
            "model_state_dict": self.policy_value_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "extra_state": extra_state or {},
            "torch_version": torch.__version__,
        }
        torch.save(ckpt, checkpoint_file)

    def load_checkpoint(self, checkpoint_file: str) -> Dict[str, Any]:
        """
        Load a training checkpoint saved by save_checkpoint().

        Returns the stored extra_state dict.
        """
        # Load to CPU first to avoid placing RNG state tensors on CUDA.
        map_location = torch.device("cpu")
        ckpt = torch.load(checkpoint_file, map_location=map_location, weights_only=False)
        self.policy_value_net.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.base_lr = float(ckpt.get("base_lr", self.base_lr))
        for pg in self.optimizer.param_groups:
            pg["lr"] = self.base_lr
        self._lr_multiplier = float(ckpt.get("lr_multiplier", self._lr_multiplier))
        self._move_optimizer_state_to_device()
        self.set_lr_multiplier(self._lr_multiplier)
        extra = ckpt.get("extra_state") or {}
        return extra

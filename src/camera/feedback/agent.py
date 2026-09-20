"""Targeted grayscale SAC implementation for structured-light parameter tuning."""

from __future__ import annotations

import argparse
import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from .environment import CameraTuningEnv, CurriculumScheduler, FixedCleanupEvaluator, MockCameraBackend


class StructuredLightEncoder(nn.Module):
    """Structured-light encoder with local contrast and GroupNorm for tiny batches."""

    def __init__(self, feature_dim: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 24, 7, stride=2, padding=3),
            nn.GroupNorm(6, 24),
            nn.SiLU(),
            nn.Conv2d(24, 48, 5, stride=2, padding=2),
            nn.GroupNorm(8, 48),
            nn.SiLU(),
            nn.Conv2d(48, 96, 3, stride=2, padding=1),
            nn.GroupNorm(12, 96),
            nn.SiLU(),
            nn.Conv2d(96, 128, 3, stride=2, padding=1),
            nn.GroupNorm(16, 128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.SiLU(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 3:
            image = image.unsqueeze(0)
        image = image.float().clamp(0.0, 1.0)
        local_contrast = image - torch.nn.functional.avg_pool2d(image, 7, stride=1, padding=3)
        return self.features(torch.cat((image, local_contrast), dim=1))


class GaussianActor(nn.Module):
    def __init__(self, encoder: StructuredLightEncoder, action_dim: int = 2) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, action_dim))
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def distribution(self, image: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        mean = self.head(self.encoder(image))
        log_std = self.log_std.expand_as(mean).clamp(-5.0, 1.0)
        return Normal(mean, log_std.exp()), mean

    def sample(self, image: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        distribution, mean = self.distribution(image)
        raw = mean if deterministic else distribution.rsample()
        action = torch.tanh(raw)
        log_prob = distribution.log_prob(raw).sum(dim=-1) - torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
        return action, log_prob


class TwinQ(nn.Module):
    def __init__(self, action_dim: int = 2) -> None:
        super().__init__()
        self.encoder = StructuredLightEncoder()
        self.q1 = nn.Sequential(nn.Linear(128 + action_dim, 128), nn.SiLU(), nn.Linear(128, 1))
        self.q2 = nn.Sequential(nn.Linear(128 + action_dim, 128), nn.SiLU(), nn.Linear(128, 1))

    def forward(self, image: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(image)
        inputs = torch.cat((features, action), dim=-1)
        return self.q1(inputs).squeeze(-1), self.q2(inputs).squeeze(-1)


class ReplayBuffer:
    def __init__(self, capacity: int = 10000) -> None:
        self.data: deque[tuple[np.ndarray, np.ndarray, float, np.ndarray, float]] = deque(maxlen=capacity)

    def add(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.data.append((state.astype(np.float32), action.astype(np.float32), float(reward), next_state.astype(np.float32), float(done)))

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        if len(self.data) < batch_size:
            raise ValueError("not enough replay samples")
        indices = np.random.randint(0, len(self.data), size=batch_size)
        batch = [self.data[i] for i in indices]
        return tuple(torch.from_numpy(np.stack(values)) for values in zip(*batch))

    def __len__(self) -> int:
        return len(self.data)


@dataclass(frozen=True, slots=True)
class SACConfig:
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.10
    learning_rate: float = 3e-4
    batch_size: int = 32
    warmup_episodes: int = 32
    seed: int = 7
    checkpoint: str = "data/tuning/camera_sac.pt"


class ImageSACAgent:
    def __init__(self, config: SACConfig | None = None, *, device: str = "cpu") -> None:
        self.config = config or SACConfig()
        self.device = torch.device(device)
        torch.manual_seed(self.config.seed)
        self.actor = GaussianActor(StructuredLightEncoder()).to(self.device)
        self.critic = TwinQ().to(self.device)
        self.target = TwinQ().to(self.device)
        self.target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.learning_rate)

    def act(self, image: np.ndarray, *, deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            action, _ = self.actor.sample(torch.from_numpy(image).unsqueeze(0).to(self.device), deterministic=deterministic)
        return action.cpu().numpy()[0]

    def update(self, replay: ReplayBuffer) -> dict[str, float]:
        states, actions, rewards, next_states, dones = [value.to(self.device) for value in replay.sample(self.config.batch_size)]
        with torch.no_grad():
            next_actions, next_log_prob = self.actor.sample(next_states)
            target_q1, target_q2 = self.target(next_states, next_actions)
            target = rewards + (1.0 - dones) * self.config.gamma * (torch.minimum(target_q1, target_q2) - self.config.alpha * next_log_prob)
        q1, q2 = self.critic(states, actions)
        critic_loss = nn.functional.mse_loss(q1, target) + nn.functional.mse_loss(q2, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 5.0)
        self.critic_optimizer.step()
        new_actions, log_prob = self.actor.sample(states)
        policy_q1, policy_q2 = self.critic(states, new_actions)
        actor_loss = (self.config.alpha * log_prob - torch.minimum(policy_q1, policy_q2)).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
        self.actor_optimizer.step()
        with torch.no_grad():
            for target_parameter, parameter in zip(self.target.parameters(), self.critic.parameters()):
                target_parameter.mul_(1.0 - self.config.tau).add_(self.config.tau * parameter)
        return {"critic_loss": float(critic_loss.item()), "actor_loss": float(actor_loss.item())}

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict(), "config": asdict(self.config)}, path)
        return path


def train_mock(episodes: int, config: SACConfig | None = None) -> Path:
    config = config or SACConfig()
    random.seed(config.seed)
    np.random.seed(config.seed)
    backend = MockCameraBackend()
    env = CameraTuningEnv(backend, FixedCleanupEvaluator(min_component_area=8))
    schedule = CurriculumScheduler.default(seed=config.seed)
    agent = ImageSACAgent(config)
    replay = ReplayBuffer()
    recent: list[float] = []
    for episode in range(episodes):
        context = schedule.context(episode)
        state, _ = env.reset(options={"context": context})
        action = agent.act(state) if episode >= config.warmup_episodes else np.random.uniform(-1.0, 1.0, size=2).astype(np.float32)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        replay.add(state, action, reward, next_state, done)
        metrics = agent.update(replay) if len(replay) >= config.batch_size else {}
        recent.append(reward)
        if len(recent) > 25:
            recent.pop(0)
        if episode == 0 or (episode + 1) % max(1, min(25, episodes)) == 0:
            print(f"episode={episode + 1} stage={context.metadata['curriculum_stage']} angle={context.angle_degrees:g} reward={reward:.4f} mean25={np.mean(recent):.4f} {metrics}")
    return agent.save(config.checkpoint)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="camera-tuning-sac")
    parser.add_argument("--mock", action="store_true", help="run without opening hardware")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--checkpoint", default="data/tuning/camera_sac.pt")
    args = parser.parse_args(argv)
    if not args.mock:
        raise SystemExit("real hardware is explicit: construct CameraTuningEnv with a CaptureBackend")
    path = train_mock(args.episodes, SACConfig(checkpoint=args.checkpoint))
    print(f"checkpoint={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

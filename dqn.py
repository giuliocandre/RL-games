from collections import deque, namedtuple
from perudo import (
    Game,
    possible_actions_from_bid, 
    action_to_n, n_to_action, 
    Player, RLPlayer, AggressiveRoboPlayer, ConservativeRoboPlayer,
    load_policy, POLICY_MAP,
    MAX_ACTION_SIZE, policy)
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def legal_mask(tot_dices, last_bid_q, last_bid_f):
    possible_actions = possible_actions_from_bid(tot_dices, last_bid_q, last_bid_f)
    bools = [False] * MAX_ACTION_SIZE # [True, True, False, ..., True] size 180
    for q, f in possible_actions:
        bools[action_to_n(q, f)] = True

    legal_mask = torch.tensor(
        bools,  dtype=torch.bool
    )

    return legal_mask

Transition = namedtuple(
    "Transition",
    ["state", "action", "reward", "next_state", "done",
     "legal_mask", "next_legal_mask"]
)

def make_batch(transitions):
    batch = Transition(*zip(*transitions))
    return (
        torch.tensor(np.array(batch.state), dtype=torch.float32),
        torch.tensor(batch.action, dtype=torch.long),
        torch.tensor(batch.reward, dtype=torch.float32),
        torch.tensor(np.array(batch.next_state), dtype=torch.float32),
        torch.tensor(batch.done, dtype=torch.float32),
        torch.tensor(np.array(batch.legal_mask), dtype=torch.bool),
        torch.tensor(np.array(batch.next_legal_mask), dtype=torch.bool),
    )

class DQNPlayer(Player):
    def __init__(self, name="DQNPlayer", epsilon=1):
        super().__init__(name)
        self.state_size = 9 # (tot_dices, last_bid_q, last_bid_f, ...face_counts)
        self.action_size = MAX_ACTION_SIZE
        self.model = self._build_model()
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size)
        )

        return model


    def remember(self, state, action):
        # Here we keep track of the last reward and "done" as well
        self.history.append((state, action, 0, False))

    def update_last_reward(self, reward, done=False):
        last_state, last_action, _, _ = self.history[-1]
        self.history[-1] = (last_state, last_action, reward, done)

    # When winning, update reward of last action
    def win(self):
        self.update_last_reward(1, True)

    # When losing, update reward of last action
    def lose(self):
        self.update_last_reward(-1, True)

    def lose_dice(self):
        self.update_last_reward(-0.1, False)
        super().lose_dice()

    def win_bid(self):
        self.update_last_reward(0.1, False)
        super().win_bid()
    
    def make_action(self, total_dices, last_bid):
        state = [0] * (1 + 2 + 6)
        state[0] = total_dices
        state[1] = last_bid[0]
        state[2] = last_bid[1]
        state[3:] = self.dices
        state = tuple(state)

        if np.random.rand() <= max(self.epsilon, self.epsilon_min):
            return policy(state, 1)

        action = n_to_action(self.predict_action(state))
        return action

    def predict_q_values(self, state):
        i = torch.tensor(state, dtype=torch.float32)
        mask = legal_mask(state[0], state[1], state[2])
        q_values = self.model(i)
        q_values[~mask] = -1e9
        return q_values

    def predict_action(self, state):
        q_values = self.predict_q_values(state)
        return torch.argmax(q_values).item()


def dqn_update(online_net, target_net, optimizer, batch, gamma=0.99):
    """
    batch is a tuple of tensors:
    states        (B, 9)
    actions       (B,)
    rewards       (B,)
    next_states   (B, 9)
    dones         (B,)
    legal_masks   (B, 180)
    next_legal_masks (B, 180)
    """

    (
        states,
        actions,
        rewards,
        next_states,
        dones,
        legal_masks,
        next_legal_masks
    ) = batch

    # Q(s, a)
    q_values = online_net(states)
    # print("q_values:", q_values.shape)
    # print("actions:", actions.shape, actions.dtype)
    q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    # max_a' Q_target(s', a')
    with torch.no_grad():
        q_next = target_net(next_states)
        q_next[~next_legal_masks] = -1e9
        max_q_next = q_next.max(dim=1)[0]
        target = rewards + gamma * (1 - dones) * max_q_next

    # Huber loss
    loss = F.smooth_l1_loss(q_sa, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

class ReplayBuffer:
    def __init__(self, capacity=50_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

    def from_history(self, player):
        for i in range(len(player.history) - 1):
            state, action, reward, done = player.history[i]
            next_state, _, _, _ = player.history[i + 1]

            legal_mask_ = legal_mask(
                state[0], state[1], state[2]
            ).numpy()
            next_legal_mask_ = legal_mask(
                next_state[0], next_state[1], next_state[2]
            ).numpy()

            self.push(
                state,
                action_to_n(*action),
                reward,
                next_state,
                done,
                legal_mask_,
                next_legal_mask_
            )


def TrainDQN(OnlineDQNPlayer = None, num_episodes=10_000, opponent_model=AggressiveRoboPlayer):
    # Load tabular MDP policy map from disk
    POLICY_MAP = load_policy()

    if not OnlineDQNPlayer:
        OnlineDQNPlayer = DQNPlayer("OnlineDQNPlayer", epsilon=1)

    replay_buffer = ReplayBuffer(capacity=100_000)

    online_net = OnlineDQNPlayer.model
    optimizer = torch.optim.Adam(
        online_net.parameters(),
        lr=1e-4
    )

    TargetDQNPlayer = DQNPlayer("TargetDQNPlayer")
    TargetDQNPlayer.model.load_state_dict(online_net.state_dict())

    batch_size = 64
    epsilon_decay = 0.995
    target_update_freq = 1_000  # steps

    global_step = 0
    for episode in range(num_episodes):
        random_player = opponent_model("RandomOpponent")
        game = Game([OnlineDQNPlayer, random_player], quiet=True)
        done = False
        while not done:
            done = game.play_round()
            global_step += 1


            if len(OnlineDQNPlayer.history) > 1:
                replay_buffer.from_history(OnlineDQNPlayer)
                OnlineDQNPlayer.history = [OnlineDQNPlayer.history[-1]]  # Keep only the last transition

            # LEARNING STEP
            if len(replay_buffer) >= batch_size:
                transitions = replay_buffer.sample(batch_size)
                batch = make_batch(transitions)

                loss = dqn_update(
                    online_net,
                    TargetDQNPlayer.model,
                    optimizer,
                    batch
                )

            # TARGET NETWORK UPDATE
            if global_step % target_update_freq == 0:
                TargetDQNPlayer.model.load_state_dict(online_net.state_dict())
                print(f"Episode {episode}, Step {global_step}, Loss: {loss:.4f}")

        # ε decay per episode
        OnlineDQNPlayer.epsilon *= epsilon_decay

    return OnlineDQNPlayer


def save_checkpoint(path, online_net, optimizer=None, epsilon=None, step=None):
    checkpoint = {
        "model_state_dict": online_net.state_dict(),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if epsilon is not None:
        checkpoint["epsilon"] = epsilon
    if step is not None:
        checkpoint["step"] = step

    torch.save(checkpoint, path)

def load_checkpoint(path, online_net, optimizer=None, device="cpu"):
    checkpoint = torch.load(path, map_location=device)

    online_net.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epsilon = checkpoint.get("epsilon", None)
    step = checkpoint.get("step", None)

    return epsilon, step
from collections import namedtuple
import numpy as np
import random

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, state_shape=(84, 84, 4)):
        self.capacity = capacity
        self.alpha = alpha
        self.pos = 0
        self.size = 0
        
        # Numpy arrays for storing transitions
        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.bool_)
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.size > 0 else 1.0
        
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = done
        self.priorities[self.pos] = max_prio

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, beta=0.4):
        prios = self.priorities[:self.size]
        
        # Apply priority exponent and normalize
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(self.size, batch_size, p=probs)
        
        # Retrieve samples using indices
        batch = Transition(
            state=self.states[indices],
            action=self.actions[indices],
            reward=self.rewards[indices],
            next_state=self.next_states[indices],
            done=self.dones[indices]
        )

        # Importance-sampling weights
        total = self.size
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        return batch, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = max(prio, 1e-5)

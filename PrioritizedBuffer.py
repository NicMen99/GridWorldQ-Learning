import numpy as np

class PrioritizedBuffer:
    def __init__(self, capacity: int, n_episodes : int, alpha : float = 0.6, beta : float = 0.4):
        self.max_size = capacity
        self.alpha = alpha
        self.beta = beta
        self.eps = 10**-6
        self.buffer = [None] * capacity
        self.priorities = np.zeros(capacity, dtype=np.float32)

        self.pos = 0
        self.size = 0
        self.beta_increment = (1 - self.beta)/ n_episodes


    def add_experience(self, state, action, reward, next_state, done, td_error = None):
        if td_error is None:
            priority = self.priorities.max() if self.size > 0 else 1.0
        else:
            priority = (np.abs(td_error) + self.eps) ** self.alpha

        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = priority

        self.pos = (self.pos + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample_experience(self, replay_size):
        if self.size == 0:
            return None

        priors = self.priorities[:self.size]
        probs = priors / np.sum(priors)

        indices = np.random.choice(self.size, replay_size, p=probs)
        samples = [self.buffer[ids] for ids in indices]

        weights = (self.size * probs[indices]) ** -self.beta

        return samples, indices, np.array(weights)

    def update_priorities(self, indices, new_td):
        for idx, error in zip(indices, new_td):
            self.priorities[idx] = (abs(error) + self.eps) ** self.alpha

    def update_beta(self):
        self.beta += self.beta_increment
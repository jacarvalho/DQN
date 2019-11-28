import random
import torch
from collections import namedtuple

# Set the random seed
# random.seed(101)
# np.random.seed(102)

# Use the gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MDP transition
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'nextstate', 'done'))


class ReplayMemory:

    def __init__(self, capacity=100):
        self._capacity = capacity
        self._memory = []
        self._pos = 0

    def __len__(self):
        return len(self._memory)

    def add(self, t):
        """
        Add a transition to the replay memory.

        :param t: Transition tuple
        """
        if len(self._memory) >= self._capacity:
            # If the memory is full, replace as a FIFO
            self._memory[self._pos] = t
        else:
            self._memory.append(t)

        self._pos = (self._pos + 1) % self._capacity

    def sample(self, batch_size):
        return random.sample(self._memory, batch_size)


class EpsilonSchedule:

    def __init__(self, eps_start=0.9, eps_min=0.05, eps_decay=0.999):
        self._eps_start = eps_start
        self._eps_end = eps_min
        self._eps_decay = eps_decay
        self._eps = eps_start
        self._n_steps = 0


class EpsilonScheduleExponential(EpsilonSchedule):

    def __init__(self, eps_start=0.9, eps_min=0.05, eps_decay=0.999):
        super(EpsilonScheduleExponential, self).__init__(eps_start=eps_start, eps_min=eps_min, eps_decay=eps_decay)

    def __call__(self, *args, **kwargs):
        if self._n_steps > 0:
            self._eps = max(self._eps * self._eps_decay, self._eps_end)
        self._n_steps += 1
        return self._eps


class EpsilonScheduleLinear(EpsilonSchedule):

    def __init__(self, eps_start=0.9, eps_min=0.05, eps_decay=0.999):
        super(EpsilonScheduleLinear, self).__init__(eps_start=eps_start, eps_min=eps_min, eps_decay=eps_decay)

    def __call__(self, *args, **kwargs):
        if self._n_steps > 0:
            self._eps = max(self._eps - self._eps_decay * self._eps, self._eps_end)
        self._n_steps += 1
        return self._eps


import random


class ReplayBuffer:
    def __init__(self):
        self._buffer = []

    def add(self, transition):
        self._buffer.append(transition)

    def sample(self, batch_size=None):
        return (
            self._buffer if batch_size is None else random.sample(self._buffer, batch_size)
        )

    def clear(self):
        self._buffer.clear()

    def __len__(self):
        return len(self._buffer)

    def get_all(self):
        return self._buffer

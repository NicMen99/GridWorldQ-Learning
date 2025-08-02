import numpy as np

class QTable:
    def __init__(self):
        self.table = {}

    def update_table(self, state, action, model):
        if not self.table[state]:
            self.table[state] = np.zeros(4)
        next_state = model.compute_action(state, action)
        self.table[state][action] = self.table[state][action] + model.learning_rate * (model.reward + model.discount * np.max(self.table[next_state]) - self.table[state][action])

    def dic2mat(self):
        pass #Per convertire in qualcosa di leggibile e mostrabile

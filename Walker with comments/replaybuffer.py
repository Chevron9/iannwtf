import numpy as np

# Replaybuffer that stores all informations about the agents transitions
# in np arrays
class ReplayBuffer:
    #initializes the memories with zeros with sizes depending on a big maximal size,
    #the input_dimensions(output of the environment) or the numbers of possible
    #actions
    def __init__(self, max_size, input_shape, n_actions):
        self.max_size = max_size
        self.current_position = 0
        self.state_memory = np.zeros((self.max_size, *input_shape))
        self.new_state_memory = np.zeros((self.max_size, *input_shape))
        self.action_memory = np.zeros((self.max_size, n_actions))
        self.reward_memory = np.zeros(self.max_size)
        self.terminal_memory = np.zeros(self.max_size, dtype=np.bool)

    #stores new transitions in memory
    def store_transition(self, state, action, reward, state_, done):
        index = self.current_position % self.max_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.current_position+= 1

    #gives back a random batch of of transition samples
    def sample_buffer(self, batch_size):
        #makes sure to have the correct current size of the memory
        max_mem = min(self.current_position, self.max_size)

        #selects a random batch of indexes in the memory size
        batch = np.random.choice(max_mem, batch_size, replace=False)

        #retrieves the batch from memory
        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones

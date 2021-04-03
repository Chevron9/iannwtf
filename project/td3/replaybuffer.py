import numpy as np

# Replaybuffer that stores all informations about the agents transitions
# in np arrays
class ReplayBuffer:
    #initializes the memories with zeros with sizes depending on a big maximal size,
    #the input_dimensions(output of the environment) or the numbers of possible
    #actions
    def __init__(self, max_size, input_shape, n_actions, prioritize = False):
        self.max_size = max_size
        self.current_position = 0
        self.state_memory = np.zeros((self.max_size, *input_shape))
        self.new_state_memory = np.zeros((self.max_size, *input_shape))
        self.action_memory = np.zeros((self.max_size, n_actions))
        self.reward_memory = np.zeros(self.max_size)
        self.terminal_memory = np.zeros(self.max_size, dtype=np.bool)

        self.prioritize = prioritize
        if self.prioritize:
            self.error = np.zeros(self.max_size)

    #stores new transitions in memory
    def store_transition(self, state, action, reward, state_, done):
        index = self.current_position % self.max_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        if self.prioritize:
            # self.ranks[index]=(max(self.ranks)+1)
            self.error[index]=max(self.error)+1

        self.current_position+= 1

    def update_memories(self, batch, deltas):
        #print("error ",self.error)
        #print(len(self.error))
        #print("batch ",batch)
        #print(len(batch))
        #print("deltas ",deltas)
        #print(len(deltas))
        self.error[batch]=deltas

    #gives back a random batch of of transition samples
    def sample_buffer(self, batch_size):
        #makes sure to have the correct current size of the memory
        max_mem = min(self.current_position, self.max_size)
        
        if self.prioritize:
            max_error = self.error[0:max_mem]
            tmp = np.argsort(max_error)
            self.ranks = np.zeros(max_mem)
            self.p = np.zeros(max_mem)

            self.ranks[tmp] = np.arange(max_mem)

            # np is much more performant than list comprehensions
            self.ranks = np.add(1,self.ranks) # [x+1 for x in self.ranks]
            tmp_p = np.divide(1,self.ranks)
            self.p = np.divide(tmp_p,np.sum(tmp_p)) # [x/(np.sum(tmp_p)) for x in tmp_p] 
            # self.p = np.asarray(self.p)

            batch = np.random.choice(max_mem, batch_size, replace=False, p=self.p)

        #selects a random batch of indexes in the memory size
        if not self.prioritize:
            batch = np.random.choice(max_mem, batch_size, replace=False)                
            
        

        

        #retrieves the batch from memory
        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        if self.prioritize:
            probabilities = np.zeros(len(self.p))
            probabilities = self.p[batch]
            return states, actions, rewards, states_, dones, batch, probabilities

        else:
            return states, actions, rewards, states_, dones

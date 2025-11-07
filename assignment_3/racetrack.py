import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List, Tuple

class Track:
    """
    Main abstraction for the track on which the car travels. It loads the course as a numpy array. It has functionality for moving the car and updating the velocity. 
    """
    def __init__(self, course:Union[List, np.array], max_velocity:int=4, noise:float=0.0) -> None:
        """
        Initializes the Track
        Args: 
            course (List or np.arry): The course that we want to learn
            max_velocity (int): The maximum velocity that the car can attain
            noise (float): The amount of noise (i.e. stoachastic threshold) as specified in S and B 5.12 
        """
        self.course = np.array(course)
        self.max_velocity = max_velocity
        self.noise = noise

        self.velocity = np.array([0, 0])
        self.position = None

        self.load_course()
        self.random_start()

    def load_course(self) -> None:
        """
        Loads the course while taking note of its size, flipping it so that (0,0) is in the natural place, and saving the starting points
        """
        self.vert_size, self.hori_size = self.course.shape
        self.course = np.fliplr(self.course)
        self.starting_spots = np.argwhere(self.course == 1)

    def random_start(self) -> None:
        """
        Chooses a random starting spot
        """
        self.position = self.starting_spots[np.random.choice(range(len(self.starting_spots)))]
    
    def reset(self) -> None:
        """
        Resets the state of the track by selecting a new starting spot and setting the velocity in both directions to 0
        """
        self.random_start()
        self.velocity = np.array([0, 0])
    
    def is_off_track(self, pos:np.array) -> bool:
        """
        Checks if the current position is on or off the track
        Args: 
            pos (np.array): current position of the car
        Returns:
            bool: Whether we are on the track or not
        """
        try:
            return self.course[tuple(pos)] == np.int64(-1)
        except IndexError: # when the index is beyond the sides of the track
            return True
    
    def is_at_finish(self, pos:np.array) -> bool:
        """
        Checks if the current position is passed the finish line
        Args: 
            pos (np.array): current position of the car
        Returns:
            bool: Whether we are past the finish line
        """
        return self.course[tuple(pos)] == np.int64(2)
    
    def update_velocity(self, action:Tuple) -> None:
        """
        Updates the velocity based on the chosen action; ignores the update if we pass the noise threshold
        Args: 
            action (tuple): chosen action
        """
        if np.random.rand() > self.noise:
            self.velocity += np.array(action)
            self.velocity = np.minimum(self.velocity, self.max_velocity)
            self.velocity = np.maximum(self.velocity, 0)
    
    def update_position(self) -> int:
        """
        Updates the position of the car and returns the corresponding reward
        Returns:
            int: The reward of the square that we are on
        """
        for tstep in range(0, self.max_velocity+1): # must iterate through the possible next steps to see if we will leave the track
            t = tstep / self.max_velocity
            pos = np.int64(self.position - np.round(self.velocity * t)) # minus because we need to "go up" the track
            if self.is_off_track(pos):
                self.reset()
                return -1
            if self.is_at_finish(pos):
                self.position = pos
                self.velocity = np.array([0, 0])
                return 100 
        self.position = pos # actual position update
        return -1
    
    def move(self, action:Tuple) -> int:
        """
        Updates the velocity and position of the car and returns the corresponding reward
        Args: 
            action (tuple): chosen action
        Returns:
            int: The reward of the square that we are on
        """
        self.update_velocity(action)
        reward = self.update_position()
        return reward

    def get_state(self) -> Tuple:
        """
        Helper function that just returns the state of the car
        Returns:
            tuple: Current position and velocity
        """
        return self.position[0], self.position[1], self.velocity[0], self.velocity[1]

    def epsilon_greedy(self, eps, actions, Q):
        """
        Helper function that runs the epsilon policy
        Args:
            eps (float): epsilon value
            actions (list): a list of all possible actions
            Q (array): the table of Q values
        Returns:
            tuple: Chosen action
        """
        s = self.get_state()
        pos_x, pos_y, vel_x, vel_y = s[0], s[1], s[2], s[3]
        if np.random.rand() < eps:
            a = actions[np.random.choice(list(range(len(actions))))]
        else:
            a_idx = np.argmax(Q[pos_x, pos_y, vel_x, vel_y, :, :])
            a = np.unravel_index(a_idx, (3,3)) - np.array([1,1])  # Maps [0,1,2] back to [-1,0,1]
            a = (int(a[0]), int(a[1]))
        return a

class Sarsa:
    """
    Main abstraction for running the Sarsa algorithm and tracking each episode
    """
    def __init__(self, track, N=2000, gamma=0.99, alpha=0.1, epsilon=0.1, max_steps=500, log_every=200):
        """
        Initializes the Sarsa algorithm
        Args: 
            course (List or np.arry): The course that we want to learn
            N (int): The number of episodes to simulate
            epsilon (float): The probability to explore rather than exploit
            gamma (float): The discount factor
            max_steps (int): The amount of maximum steps before a single episode ends
            log_every (int): The amount of logging messages to see
        """
        self.track = Track(course=track)
        self.N = N
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_steps = max_steps
        self.log_every = log_every

        self.Q = np.zeros((self.track.vert_size,self.track.hori_size,5,5,3,3))
        self.poss_actions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
        self.counts = []
    
    def run_episode(self):
        """
        Runs a single episode
        Returns:
            list: The list of all state action reward tuples in the episode 
        """
        self.track.reset()
        s = self.track.get_state()
        a = self.track.epsilon_greedy(self.epsilon, self.poss_actions, self.Q)
        steps = 0

        while not self.track.is_at_finish(self.track.position) and steps < self.max_steps:
            steps += 1
            r = self.track.move(a)
            s_prime = self.track.get_state()
            a_prime = self.track.epsilon_greedy(self.epsilon, self.poss_actions, self.Q)

            s_a = (s[0], s[1], s[2], s[3], a[0]+1, a[1]+1)
            s_a_prime = (s_prime[0], s_prime[1], s_prime[2], s_prime[3], a_prime[0]+1, a_prime[1]+1)

            self.Q[s_a] = self.Q[s_a] + self.alpha * (r + self.gamma * self.Q[s_a_prime] - self.Q[s_a])
            s, a = s_prime, a_prime
        
        self.counts.append(steps)
    
    def learn(self):
        """
        Simple loop for running an episode and then updating the relevant tables
        """
        for e in range(self.N):
            if (e+1) % self.log_every == 0:
                print(f"Episode {e+1}")
            self.run_episode()
    
    def create_policy(self):
        """
        Converts the Q table into a policy that the agent can use.
        """
        pi = np.zeros((self.track.vert_size, self.track.hori_size, 5, 5), dtype=np.int64)
        for idx in np.ndindex(self.track.vert_size, self.track.hori_size, 5, 5):
            a = np.argmax(self.Q[idx[0], idx[1], idx[2], idx[3], :, :])
            a = np.unravel_index(a, (3, 3)) - np.array([1, 1])
            pi[idx] = self.poss_actions.index(tuple(a))
        return pi

    def infer(self, runs=10):
        """
        Runs the Sarsa Algorithm wihtout updates
        Args: 
            run (int): The amout of inference iterations to run
        Returns:
            array: A map of actions taken by the agent
        """
        pi = self.create_policy()
        pos_map = np.zeros((self.track.vert_size, self.track.hori_size))
        for _ in range(runs):
            self.track.reset()
            while not self.track.is_at_finish(self.track.position):
                pos_x, pos_y, vel_x, vel_y = self.track.get_state()
                pos_map[pos_x, pos_y] += 1
                if vel_x == 0 and vel_y == 0:
                    action = (1, 1) 
                else:
                    action = self.poss_actions[pi[pos_x, pos_y, vel_x, vel_y]]
                self.track.move(action)
        pos_map = (pos_map > 0).astype(np.float32)
        pos_map += self.track.course
        return np.flip(pos_map, axis=1)
    
    def plot_episode_lengths(self, label, window=100):
        smoothed = np.convolve([c for c in self.counts], np.ones(window)/window, mode='valid')
        plt.plot(smoothed, label=label)
        plt.ylabel("Steps")
        plt.xlabel("Episodes")
        plt.title("Steps vs. Episodes")
        plt.legend()

class NStepSarsa(Sarsa):
    def __init__(self, track, N=2000, n=3, gamma=0.99, alpha=0.1, epsilon=0.1, max_steps=500, log_every=200):
        super().__init__(track, N, gamma, alpha, epsilon, max_steps, log_every)
        self.n = n
    
    def run_episode(self):
        self.track.reset()
        # initialize and store S_0 != terminal
        states = [] 
        actions = []
        rewards = []
        steps = 0

        s = self.track.get_state()
        a = self.track.epsilon_greedy(self.epsilon, self.poss_actions, self.Q) # Select and store an action A_0 from pi
        states.append(s)
        actions.append(a)

        t = 0 
        T = np.inf # T get infinity
        while t < self.max_steps: # loop for t = 0, 1, 2 ...
            if t < T: # if t < T
                r = self.track.move(a) # take action A_t
                rewards.append(r) # observe and store the next reward as R_t+1 
                s_prime = self.track.get_state() # and the next state as S_t+1

                if self.track.is_at_finish(self.track.position): # if S_t+1 is terminal
                    T = t + 1 # T gets t + 1
                else:
                    a_prime = self.track.epsilon_greedy(self.epsilon, self.poss_actions, self.Q) # select and store an action A_t+1 from pi
                    actions.append(a_prime)
                
                # sarSA
                states.append(s_prime)
                s, a = s_prime, a_prime
                steps += 1
            
            tau = t - self.n + 1 # tau gets t - n + 1 (tau is the time whose estimate is being updated)
            if tau >= 0:
                G = 0
                for i in range(tau+1, min(tau+self.n, T)+1): # need to add one for indexing
                    G += (self.gamma ** (i - tau - 1) * rewards[i-1]) # n step udpate
                
                if tau + self.n < T:
                    # if tau + n < T then do an n step update
                    s_future = states[tau + self.n]
                    a_future = actions[tau + self.n]
                    s_a_future = (s_future[0], s_future[1], s_future[2], s_future[3], a_future[0]+1, a_future[1]+1)
                    G += (self.gamma ** self.n) * self.Q[s_a_future]
                    # we are not learning pi here
                
                s_tau = states[tau]
                a_tau = actions[tau]
                s_a = (s_tau[0], s_tau[1], s_tau[2], s_tau[3], a_tau[0]+1, a_tau[1]+1)
                self.Q[s_a] += self.alpha * (G - self.Q[s_a]) # Q table update

            if tau == T - 1: # until tau = T - 1
                break

            t += 1
        self.counts.append(steps)

def main():
    course1 = np.array([
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  -1, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1],
    ])

    sarsa = Sarsa(course1)
    three_step_sarsa = NStepSarsa(course1, n=3)
    five_step_sarsa = NStepSarsa(course1, n=5)
    seven_step_sarsa = NStepSarsa(course1, n=7)

    sarsa.learn()
    three_step_sarsa.learn()
    five_step_sarsa.learn()
    seven_step_sarsa.learn()

    # show trajectories
    pos_map1 = sarsa.infer()
    plt.imshow(pos_map1)
    pos_map2 = three_step_sarsa.infer()
    plt.imshow(pos_map2)
    pos_map3 = five_step_sarsa.infer()
    plt.imshow(pos_map3)
    pos_map4 = seven_step_sarsa.infer()
    plt.imshow(pos_map4)

    # do episode length analysis
    sarsa.plot_episode_lengths(label="Sarsa")
    three_step_sarsa.plot_episode_lengths(label="3-Step Sarsa")
    five_step_sarsa.plot_episode_lengths(label="5-Step Sarsa")
    seven_step_sarsa.plot_episode_lengths(label="7-Step Sarsa")
    
if __name__ == "__main__":
    main()
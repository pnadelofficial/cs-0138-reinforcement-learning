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
        Loads the course while taking note of its size, flipping it so that (0,0) is in the natural place, and savign the starting points
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
        except IndexError: # when the index is beyond the bottom of the track
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
                return 0
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

class MonteCarloSimulation:
    """
    Main abstraction for running the Monte Carlo simulations and tracking each episode
    """
    def __init__(self, course:Union[List, np.array], num_eps:int=20000, epsilon:float=0.1, gamma:float=0.9, max_steps:int=100, max_velocity:int=4, noise:float=0.1, non_finish_penalty:Union[None, int]=None, logging_divs:int=10) -> None:
        """
        Initializes the simulation
        Args: 
            course (List or np.arry): The course that we want to learn
            num_eps (int): The number of episodes to simulate
            epsilon (float): The probability to explore rather than exploit
            gamma (float): The discount factor
            max_steps (int): The amount of maximum steps before a single episode ends
            max_velocity (int): The maximum velocity that the car can attain
            noise (float): The amount of noise (i.e. stoachastic threshold) as specified in S and B 5.12 
            non_finish_penalty (None or int): The amount of penalty to take if the episode does not complete
            logging_divs (int): The amount of logging messages to see
        """
        self.num_eps = num_eps
        self.epsilon = epsilon
        self.gamma = gamma
        self.max_steps = max_steps
        self.noise = noise
        self.max_velocity = max_velocity
        self.track = Track(course, max_velocity=self.max_velocity, noise=self.noise)
        self.non_finish_penalty = non_finish_penalty
        self.logging_divs = logging_divs

        self.actions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
        self.Q = np.zeros((self.track.vert_size, self.track.hori_size, len(list(range(self.track.max_velocity+1))), len(list(range(self.track.max_velocity+1))), 3, 3)) # vert, hori, max_x_vel, max_y_vel, poss_x_actions, poss_y_actions
        self.C = np.zeros((self.track.vert_size, self.track.hori_size, len(list(range(self.track.max_velocity+1))), len(list(range(self.track.max_velocity+1))), 3, 3)) # vert, hori, max_x_vel, max_y_vel, poss_x_actions, poss_y_actions
        self.pi = np.zeros((self.track.vert_size, self.track.hori_size, len(list(range(self.track.max_velocity+1))), len(list(range(self.track.max_velocity+1)))), dtype=np.int64) # vert, hori, max_x_vel, max_y_vel
        self.explored = np.zeros((self.track.vert_size, self.track.hori_size))
        self.episodes = []

    def run_episode(self) -> List:
        """
        Runs a single episode
        Returns:
            list: The list of all state action reward tuples in the episode 
        """
        self.track.reset()
        episode = []
        reward = -1
        steps = 0
        while reward == -1:
            pos_x, pos_y, vel_x, vel_y = self.track.get_state()
            self.explored[pos_x, pos_y] += 1
            if np.random.rand() < self.epsilon:
                action = self.actions[np.random.choice(list(range(len(self.actions))))]
            else:
                action = self.actions[self.pi[pos_x, pos_y, vel_x, vel_y]]
            reward = self.track.move(action)
            steps += 1
            if steps >= self.max_steps: # if the episode takes too long
                if self.non_finish_penalty:
                    reward = self.non_finish_penalty
                    episode.append((((pos_x, pos_y), (vel_x, vel_y)), action, reward))
                break
            episode.append((((pos_x, pos_y), (vel_x, vel_y)), action, reward))
        self.episodes.append(episode.copy()) # need to copy because we pop later
        return episode
    
    def update(self, episode:List) -> None:
        """
        Updates Q, C and pi tables according to the on-policy Monte Carlo algorithm
        Args: 
            episode (List): A single episode in the form of a list of state action reward tuples
        """
        G = 0 
        W = 1
        while len(episode) > 0:
            state, action, reward = episode.pop()
            pos_x, pos_y = state[0][0], state[0][1]
            vel_x, vel_y = state[1][0], state[1][1]
            action_x, action_y = action
            state_action = (pos_x, pos_y, vel_x, vel_y, action_x, action_y)   
            G = self.gamma * G + reward  
            self.C[state_action] += W  
            self.Q[state_action] += (W / self.C[state_action]) * (G - self.Q[state_action])

            q_values = []
            for ax in [-1, 0, 1]:
                for ay in [-1, 0, 1]:
                    q_values.append(self.Q[(pos_x, pos_y, vel_x, vel_y, ax, ay)])
            
            best_action_idx = np.argmax(q_values)
            self.pi[pos_x, pos_y, vel_x, vel_y] = best_action_idx
    
    def simulate(self) -> None:
        """
        Simple loop for running an episode and then updating the relevant tables
        """
        for e in range(1, self.num_eps+1):
            if (e % (self.num_eps/self.logging_divs)) == 0:
                print(f"Episode {e}/{self.num_eps}")
            episode = self.run_episode()
            self.update(episode)
    
    def infer(self, num_iters:int=5000):
        """
        Runs the Monte Carlo Algorithm wihtout updates
        Args: 
            num_iters (int): 
        Returns:
            int: The reward of the square that we are on
        """
        pos_map = np.zeros((self.track.vert_size, self.track.hori_size))
        self.track.reset()
        G = 1
        for e in range(num_iters):
            pos_x, pos_y, vel_x, vel_y = self.track.get_state()
            pos_map[pos_x, pos_y] += 1
            action = self.actions[self.pi[pos_x, pos_y, vel_x, vel_y]]
            G += self.track.move(action)
            if self.track.is_at_finish(self.track.position):
                break
        pos_map = (pos_map > 0).astype(np.float32)
        pos_map +=  self.track.course
        return np.flip(pos_map, axis=1)
    
    def plot_episode_lengths(self, label, window=100):
        smoothed = np.convolve([len(e) for e in self.episodes], np.ones(window)/window, mode='valid')
        plt.plot(smoothed, label=label)
        plt.ylabel("Steps")
        plt.xlabel("Episodes")
        plt.title("Steps vs. Episodes")
        plt.legend()

# def main():
#     large_course = np.array([
#         [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2],
#         [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2],
#         [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  -1, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2],
#         [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2],
#         [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2],
#         [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2],
#         [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1],
#         [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
#         [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
#         [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
#         [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
#         [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
#         [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
#         [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
#         [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
#         [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
#         [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
#         [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
#         [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
#         [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
#         [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
#         [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
#         [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
#         [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
#         [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
#         [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
#         [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
#         [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
#         [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
#         [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
#         [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
#         [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
#         [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
#         [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
#         [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
#         [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1],
#     ])

#     mcs = MonteCarloSimulation(course=course)

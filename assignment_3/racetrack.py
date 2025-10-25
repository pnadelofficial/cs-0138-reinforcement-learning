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

# import pygame
import sys
from os import path
from typing import Optional
import numpy as np
import pandas as pd
from imitation.data import serialize
import gym
from gym import spaces
from utils import withinSight, getPositions,getDensity, getFront, getContact

width = 13
class Rush(gym.Env):
    def __init__(self, datasets_paths, trajectories_path, episode_length = 9):
        # normalization
        self.datasets = self.load_datasets(datasets_paths)
        self.trajectories = self.load_trajectories(trajectories_path)
        self.episode_length = episode_length

        self.max_speed = 6
        self.max_direction = np.pi
        movement_high = np.array([9,self.max_direction])
        movement_low = np.array([0,-self.max_direction])
        self.max_distance = 10
        self.attendant = 25
        position_high = np.array([self.max_distance,0])
        position_low = np.array([-self.max_distance,-self.max_distance])
        self.observation_space = spaces.Dict(
            {
                "position":spaces.Box(low=position_low,high=position_high,dtype=np.float32),
                "destination":spaces.Box(low=position_low,high=position_high,dtype=np.float32),
                'distance':spaces.Box(low=0, high=self.max_distance, dtype=np.float32),
                'self movement':spaces.Box(low=movement_low,high=movement_high,dtype=np.float32),
                'front movement':spaces.Box(low=movement_low, high=movement_high, dtype=np.float32),
                'density':spaces.Discrete(self.attendant),
                'contact':spaces.MultiBinary(4)
            }
        )
        action_high = np.array([self.max_speed, self.max_direction])
        self.action_space = spaces.Box(low=-action_high, high=action_high, dtype=np.float32)

    def load_datasets(self, datasets_paths):
        datasets = []
        for dataset_path in datasets_paths:
            dataset = pd.read_csv(dataset_path)
            datasets.append(dataset)
        return datasets

    def load_trajectories(self, trajectories_path):
        trajectories = serialize.load(trajectories_path)
        return trajectories

    def step(self, action):
        self.change_speed, self.change_direction = action
        self.state = self.__getstate__()
        out_of_bounds = self.state['distance'] > self.max_distance
        done = self.step >= self.episode_length or out_of_bounds
        reward = 0
        self.step += 1
        return self.state, reward, done, self.info


    def reset(self, seed=None, options=None):
        # Call the parent class's reset method to initialize self.np_random
        super().reset(seed=seed)

        # Randomly choose a trajectory
        self.current_trajectory = self.np_random.choice(self.trajectories)
        self.state = self.current_trajectory.obs[0]
        self.info = self.current_trajectory.infos[0]
        self.dataset = self.datasets[self.info[0]-1] #identify to which experiment it belongs
        self.ID = self.info[1]
        self.time_step = self.info[2]
        self.step = 1

        return self.state, self.info

    def __getstate__(self):
        speed = self.state['self movement'][0] + self.change_speed
        direction = self.state['self movement'][1] + self.change_direction
        x1 = self.state['position'][0] + speed*np.cos(direction)*0.5 #half second
        y1 = self.state['position'][1] + speed*np.sin(direction)*0.5 #half second
        dist = np.sqrt((x1)**2 + (y1)**2)
        positions = getPositions(self.dataset, self.time_step+self.step, self.ID, width=width)
        front = getFront(self.dataset, self.time_step+self.step, width, positions, x1, y1, direction)
        density = getDensity(positions, x1, y1, direction)
        contact = getContact(self.dataset, self.time_step+self.step, width, x1, y1)
        return {'position':np.array([x1,y1]),
                'destination':self.state['destination'],
                'distance':np.array([dist]),
                'self movement':np.array([speed,direction]),
                'front movement':np.array(front),
                'density':np.array([density]),
                'contact':np.array(contact)}


    # def close(self):



"""
    # Initialize Pygame
    pygame.init()

    # Set up the display
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption('Incremental Points Connector')

    # Color definitions
    background_color = (255, 255, 255)
    point_color = (0, 0, 255)
    line_color = (255, 0, 0)

    # List of predefined points
    all_points = [(100, 100), (200, 300), (400, 300), (600, 100)]
    displayed_points = []  # Points that will be displayed incrementally

    # Control the update rate
    clock = pygame.time.Clock()
    fps = 1  # Frames per second, adjust this to change the update speed

    # Index to keep track of points being displayed
    index = 0

    # Main loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Add new point to be displayed based on the timer
        if index < len(all_points):
            displayed_points.append(all_points[index])
            index += 1

        # Clear the screen
        screen.fill(background_color)

        # Draw all displayed points
        for point in displayed_points:
            pygame.draw.circle(screen, point_color, point, 5)

        # Draw lines between the displayed points
        if len(displayed_points) > 1:
            pygame.draw.lines(screen, line_color, False, displayed_points)

        # Update the display
        pygame.display.flip()

        # Control the update rate
        clock.tick(fps)

    # Quit Pygame
    pygame.quit()
    sys.exit()
"""
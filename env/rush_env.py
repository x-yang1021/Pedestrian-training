# import pygame
import sys
from os import path
from typing import Optional
import numpy as np
import pandas as pd
from imitation.data import serialize
import gymnasium as gym
from gymnasium import spaces
from env.utils import withinSight, getPositions,getDensity, getFront, getContact

width = 13
class Rush(gym.Env):
    def __init__(self, datasets_path1,datasets_path2,datasets_path3, trajectories_path, episode_length = 9):
        # normalization
        datasets_paths = [datasets_path1,datasets_path2,datasets_path3]
        self.datasets = Rush.load_datasets(datasets_paths)
        self.trajectories = Rush.load_trajectories(trajectories_path)
        self.episode_length = episode_length

        self.max_speed = 9
        self.max_direction = np.pi
        self.max_distance = 10
        movement_high = np.array([self.max_speed,self.max_direction], dtype=np.float32)
        movement_low = np.array([0,-self.max_direction], dtype=np.float32)
        self.attendant = 25
        position_high = np.array([self.max_distance,0], dtype=np.float32)
        position_low = np.array([-self.max_distance,-self.max_distance], dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "contact": spaces.MultiBinary(4),
                "density": spaces.Discrete(self.attendant),
                "destination": spaces.Box(low=position_low, high=position_high, dtype=np.float32),
                "distance": spaces.Box(low=0, high=self.max_distance, dtype=np.float32),
                "front movement": spaces.Box(low=movement_low, high=movement_high, dtype=np.float32),
                "position": spaces.Box(low=position_low, high=position_high, dtype=np.float32),
                "self movement": spaces.Box(low=movement_low, high=movement_high, dtype=np.float32)
            }
        )
        self.max_speed_change = 6
        self.max_direction_change = np.pi
        action_high = np.array([self.max_speed_change, self.max_direction_change], dtype=np.float32)
        self.action_space = spaces.Box(low=-action_high, high=action_high, dtype=np.float32)

    def step(self, actions):
        self.change_speed = actions[0]
        self.change_direction = actions[1]
        self.state = self.__getstate__()
        out_of_bounds = self.state['distance'] > self.max_distance or self.state['position'][1] > 0
        done = self.train_step >= self.episode_length or out_of_bounds
        reward = 0
        self.train_step += 1
        return self.state, reward, done, self.info


    def reset(self, seed=None, options=None):
        # Call the parent class's reset method to initialize self.np_random
        super().reset(seed=seed)

        # Randomly choose a trajectory
        self.current_trajectory = self.np_random.choice(self.trajectories)
        self.current_ob = self.current_trajectory.obs[0]
        self.state ={
            'contact': np.array(self.current_ob['contact']),
            'density': np.array(self.current_ob['density']),
            'destination': np.array(self.current_ob['destination']),
            'distance': np.array(self.current_ob['distance']),
            'front movement': np.array(self.current_ob['front movement']),
            'position': np.array(self.current_ob['position']),
            'self movement': np.array(self.current_ob['self movement'])

        }
        self.info = self.current_trajectory.infos[0]
        self.dataset = self.datasets[int(self.info['experiment']-1)] #identify to which experiment it belongs
        self.ID = self.info['ID']
        self.time_step = self.info['timestep']
        self.train_step = 1

        return self.state, self.info

    def __getstate__(self):
        speed = self.state['self movement'][0] + self.change_speed
        direction = self.state['self movement'][1] + self.change_direction
        x1 = self.state['position'][0] + speed*np.cos(direction)*0.5 #half second
        y1 = self.state['position'][1] + speed*np.sin(direction)*0.5 #half second
        dist = np.sqrt((x1)**2 + (y1)**2)
        positions = getPositions(self.dataset, self.time_step+self.train_step, self.ID, width=width)
        front = getFront(self.dataset, self.time_step+self.train_step, width, positions, x1, y1, direction)
        density = getDensity(positions, x1, y1, direction)
        contact = getContact(positions, x1, y1, direction)
        return {
            'contact': np.array(contact),
            'density': np.array([density]),
            'destination': self.state['destination'],
            'distance': np.array([dist]),
            'front movement': np.array(front),
            'position': np.array([x1, y1]),
            'self movement': np.array([speed, direction])
        }

    @staticmethod
    def load_datasets(datasets_paths):
        datasets = []
        for dataset_path in datasets_paths:
            dataset = pd.read_csv(dataset_path)
            datasets.append(dataset)
        return datasets

    @staticmethod
    def load_trajectories(trajectories_path):
        trajectories = serialize.load(trajectories_path)
        return trajectories


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
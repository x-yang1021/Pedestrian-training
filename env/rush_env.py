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
    def __init__(self, datasets_path1, datasets_path2, datasets_path3, trajectories_path, episode_length=9):
        # Load datasets and trajectories
        datasets_paths = [datasets_path1, datasets_path2, datasets_path3]
        self.datasets = Rush.load_datasets(datasets_paths)
        self.trajectories = Rush.load_trajectories(trajectories_path)
        self.episode_length = episode_length

        # Define normalization parameters
        self.max_speed = 9
        self.max_direction = np.pi
        self.max_distance = 10
        self.attendant = 25

        # Define action space
        self.max_speed_change = 6
        self.max_direction_change = np.pi
        action_high = np.array([self.max_speed_change, self.max_direction_change], dtype=np.float32)
        self.action_space = spaces.Box(low=-action_high, high=action_high, dtype=np.float32)

        # Define observation space components
        contact_shape = (4,)  # MultiBinary space with 4 elements
        density_shape = (1,)  # Discrete space as single value
        destination_shape = (2,)  # Position (x, y)
        distance_shape = (1,)  # Single value distance
        front_movement_shape = (2,)  # Speed and direction
        position_shape = (2,)  # Position (x, y)
        self_movement_shape = (2,)  # Speed and direction

        # Flattened state space shapes in the new order
        self.observation_space = spaces.Box(
            low=np.concatenate([
                np.full(position_shape, -self.max_distance, dtype=np.float32),
                np.full(destination_shape, -self.max_distance, dtype=np.float32),
                np.zeros(distance_shape, dtype=np.float32),
                np.full(self_movement_shape, -self.max_speed, dtype=np.float32),
                np.full(front_movement_shape, -self.max_speed, dtype=np.float32),
                np.zeros(density_shape, dtype=np.float32),
                np.zeros(contact_shape, dtype=np.float32)
            ]),
            high=np.concatenate([
                np.full(position_shape, self.max_distance, dtype=np.float32),
                np.full(destination_shape, self.max_distance, dtype=np.float32),
                np.full(distance_shape, self.max_distance, dtype=np.float32),
                np.full(self_movement_shape, self.max_speed, dtype=np.float32),
                np.full(front_movement_shape, self.max_speed, dtype=np.float32),
                np.full(density_shape, self.attendant, dtype=np.float32),
                np.ones(contact_shape, dtype=np.float32)
            ]),
            dtype=np.float32
        )

    def step(self, actions):
        # print('actions:', actions)
        self.change_speed = actions[0]
        self.change_direction = actions[1]
        self.state = self.__getstate__()
        # print('state:', self.state)
        # Access the new position and distance
        new_pos_x, new_pos_y = self.state[:2]
        new_dist = self.state[4]
        # Check if out of bounds
        # out_of_bounds = (new_dist > self.max_distance) or (new_pos_y > 0)
        done = self.train_step >= self.episode_length
        reward = 0
        self.train_step += 1
        return self.state, reward, done, False, self.info


    def reset(self, seed=None, options=None):
        # Call the parent class's reset method to initialize self.np_random
        super().reset(seed=seed)

        # Randomly choose a trajectory
        self.current_trajectory = self.np_random.choice(self.trajectories)
        self.state = np.array(self.current_trajectory.obs[0])
        self.info = self.current_trajectory.infos[0]
        self.dataset = self.datasets[int(self.info['experiment']-1)] #identify to which experiment it belongs
        self.ID = self.info['ID']
        self.time_step = self.info['timestep']
        self.train_step = 1

        return self.state, self.info

    def __getstate__(self):
        # Retrieve the current state values
        pos = self.state[:2]  # position (x, y)
        self_movement = self.state[5:7]  # self movement (speed, direction)
        # Calculate new state values based on actions
        speed = self_movement[0] + self.change_speed
        direction = self_movement[1] + self.change_direction
        x1 = pos[0] + speed * np.cos(direction) * 0.5  # Half second
        y1 = pos[1] + speed * np.sin(direction) * 0.5  # Half second
        dist = np.sqrt(x1 ** 2 + y1 ** 2)

        # Update other state components
        positions = getPositions(self.dataset, self.time_step + self.train_step, self.ID, width=width)
        front = getFront(self.dataset, self.time_step + self.train_step, width, positions, x1, y1, direction)
        density = getDensity(positions, x1, y1, direction)
        contact = getContact(positions, x1, y1, direction)

        # Construct the flattened state array in the specified order
        state_array = np.concatenate([
            np.array([x1, y1]),  # position (2 values)
            np.array(self.state[2:4]),  # destination (2 values)
            np.array([dist]),  # distance (1 value)
            np.array([speed, direction]),  # self movement (2 values)
            np.array(front),  # front movement (3 values)
            np.array([density]),  # density (1 value)
            np.array(contact)  # contact (4 values)
        ])

        return state_array

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
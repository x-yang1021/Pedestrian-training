# import pygame
import sys
from os import path
from typing import Optional
import numpy as np
import pandas as pd
from imitation.data import serialize
import gymnasium as gym
from gymnasium import spaces
from env.utils import get_transparency,get_green_distance,get_wall_distance


origin = [-455,52322]
North_wall = [(abs(-474 - origin[0]),52322-origin[1]), (abs(-474 - origin[0]),52468-origin[1])]
North_green = [(-455 - origin[0],52322-origin[1]), (-455 - origin[0],52468-origin[1])]
North_transparent = [(52337-origin[1],52344-origin[1]), (52401-origin[1], 52407-origin[1])]

South_origin = [-455, 52546]
South_wall = [(abs(-467 - South_origin[0]),52546-South_origin[1]), (abs(-467 - South_origin[0]), 52612-South_origin[1])]
South_green = [(-455-South_origin[0], 52546-South_origin[1]), (-455-South_origin[0], 52612-South_origin[1])]
South_transparent = [(52546-South_origin[1], 52581-South_origin[1])]

class Xinjiekou(gym.Env):

    def __init__(self, trajectory_path,North=True,  episode_length=13):
        if North:
            self.wall = North_wall
            self.green = North_green
            self.transparencies = North_transparent
        else:
            self.wall = South_wall
            self.green = South_green
            self.transparencies = South_transparent

        self.trajectories = serialize.load(trajectory_path)
        self.episode_length = episode_length

        self.max_speed = 9
        self.max_direction = np.pi
        self.position_high = np.array([self.wall[1][0], self.wall[1][1]], dtype=np.float32)
        self.position_low = np.array([self.green[0][0], self.green[0][1]], dtype=np.float32)

        self.max_width = self.wall[0][0] - self.green[0][0]

        # Define action space
        action_high = np.array([self.max_speed, self.max_direction], dtype=np.float32)
        self.action_space = spaces.Box(low=-action_high, high=action_high, dtype=np.float32)

        # Define observation space components
        regular_shape = (1,)  # Single value

        self.observation_space = spaces.Box(
            low=np.concatenate([
                self.position_low,
                np.full(regular_shape, 0, dtype=np.float32),  # Wall distance (≥ 0)
                np.zeros(regular_shape, dtype=np.float32),  # Transparency (0 stands for opaque)
                np.zeros(regular_shape, dtype=np.float32)  # heading (0 stands for south)
            ]),
            high=np.concatenate([
                self.position_high,
                np.full(regular_shape, self.max_width, dtype=np.float32),  # Wall distance
                np.ones(regular_shape, dtype=np.float32), # Transparency (1 stands for transparent)
                np.ones(regular_shape, dtype=np.float32)  # heading (1 stands for north)
            ]),
            dtype=np.float32
        )

    def step(self, actions):
        self.speed = actions[0]
        self.direction = actions[1]
        self.state = self.__getstate__()
        self.train_step += 1
        done = self.train_step >= self.episode_length
        reward = 0
        return self.state, reward, done, False, {}

    def reset(self, seed=None, options=None):
        # Call the parent class's reset method to initialize self.np_random
        super().reset(seed=seed)

        # Randomly choose a trajectory
        self.current_trajectory = self.np_random.choice(self.trajectories)
        self.state = np.array(self.current_trajectory.obs[0])
        self.heading = self.state[-1]
        self.train_step = 1

        return self.state, {}

    def __getstate__(self):
        # Retrieve the current state values
        pos = self.state[:2]  # position (x, y)
        # Calculate new state values based on actions
        speed = self.speed
        direction = self.direction
        # Normalize direction to [-pi, pi]
        if direction > np.pi:
            direction -= 2 * np.pi
        elif direction < -np.pi:
            direction += 2 * np.pi
        x1 = pos[0] + speed * np.cos(direction) * 0.4  # 0.4 second
        y1 = pos[1] + speed * np.sin(direction) * 0.4  # 0.4 second

        # Update other state components
        wall_dist = get_wall_distance(x1, self.wall)
        transparency = get_transparency(y1, self.transparencies)

        # Construct the flattened state array in the specified order
        state_array = np.concatenate([
            np.array([x1, y1], dtype=np.float32),  # position (2 values)
            np.array([wall_dist], dtype=np.float32),  # wall_dist (2 values)
            np.array([transparency], dtype=np.float32),  # transparency (1 values)
            np.array([self.heading], dtype=np.float32)  # heading (1 values)
        ])

        return state_array

    @staticmethod
    def load_trajectories(trajectories_path):
        trajectories = serialize.load(trajectories_path)
        return trajectories


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
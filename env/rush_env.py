# import pygame
import sys
from os import path
from typing import Optional
import numpy as np
import pandas as pd
from imitation.data import serialize
import gym
from gym import spaces

class Xinjiekou(gym.Env):
    def __init__(self):
        # normalization
        self.max_speed = 6
        self.max_direction = np.pi
        action_high = np.array([self.max_speed,self.max_direction])
        self.action_space = spaces.Box(low=-action_high, high=action_high, dtype=np.float32)
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
        # self.dataset = dataset2
        # self.data = trajectories

    # def step(self, action):

    # def reset(self):



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
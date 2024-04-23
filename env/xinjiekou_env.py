import pygame
import sys
from os import path
from typing import Optional
import pandas as pd
import numpy as np

import gym
from gym import spaces


sun_unity = np.array([29375,8785])
green_start1 = (np.array([29536, 9352]) - sun_unity)/10
green_end1 = (np.array([29536,10510]) - sun_unity)/10
edge_start1 = (np.array([29920,9547]) - sun_unity)/10
edge_end1 = (np.array([29688,9734]) - sun_unity)/10
wall_start1 = (np.array([29688,9734]) - sun_unity)/10
wall_end1 = (np.array([29688,10392]) - sun_unity)/10
wall_start2 = (np.array([29735,10778]) - sun_unity)/10
wall_end2 = (np.array([29738,12794]) - sun_unity)/10
wall_mid1 = (np.array([29688,10076]) - sun_unity)/10
wall_mid2 = (np.array([29738,12525])- sun_unity)/10

# dataset = pd.ExcelFile("./raw data.xlsx")
# dataset2 = pd.read_excel(dataset, 'Group 2')
# dataset3 = pd.read_excel(dataset, "Group 3")


def withinSight(x1, y1, direction, x2, y2, sight_radius=4, central_angle=np.pi):
    dx = x2 - x1
    dy = y2 - y1
    dist = np.sqrt(dx ** 2 + dy ** 2)
    if dist > sight_radius:
        return False
    direction = (direction + 2 * np.pi) % (2 * np.pi)
    angle = np.arctan2(dy, dx)
    angle = (angle + 2 * np.pi) % (2 * np.pi)
    angle_diff = min(abs(angle - direction), 2 * np.pi - abs(angle - direction))
    if angle_diff > central_angle / 2:
        return False
    else:
        return True


def get_surrounding(x, y, direction, timestep, dataset, number, threshold = 1):
    order = 0
    fast = 0
    slow = 0
    data_row = dataset.iloc[timestep:timestep + 1, :]
    for crowd in range(1, dataset.shape[1], 5):
        if order == number:
            continue
        else:
            x2 = data_row.iloc[:, crowd+1:crowd+2]
            y2 = data_row.iloc[:,crowd+2:crowd+3]
            if withinSight(x, y, direction, x2, y2):
                if data_row.iloc[:,crowd+3:crowd+4] > threshold:
                    fast += 1
                else:
                    slow += 1
        order += 1
    return np.array([fast,slow])

def get_wall(y):
    if y < wall_start1[1] or y > wall_mid1[1] and y <wall_end1[1] or y > wall_mid2[1]:
        return np.array([1])
    else:
        return np.array([0])

def get_green_space(y):
    if y < wall_end1[1]:
        return np.array([1])
    else:
        return np.array([0])

class Xinjiekou(gym.Env):

    def __init__(self):
        self.max_speed = 2
        self.max_change = np.pi
        action_high = np.array([self.max_speed,self.max_change])
        self.action_space = spaces.Box(low=-action_high, high=action_high, dtype=np.float32)
        position_high = np.array([edge_start1[0]+10,wall_end2[1]+10])
        position_low = np.array([0, 0])
        self.observation_space = spaces.Dict(
            {
                "position":spaces.Box(low=position_low,high=position_high,dtype=np.float32),
                "destination":spaces.Box(low=position_low,high=position_high,dtype=np.float32),
                "surrounding":spaces.MultiDiscrete([7,7]),
                "Wall":spaces.Discrete(2),
                "Greenspace":spaces.Discrete(2),
            }
        )

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
# import pygame
import sys
from os import path
from typing import Optional
import numpy as np
import pandas as pd
from imitation.data import serialize
import gym
from gym import spaces

mapping = { 'ID':0,
           'Trajectory':1,
           'Positionx':2,
           'Positiony':3,
            'Distance':4,
            'Up':5,
            'Right':6,
            'Down':7,
            'Left':8,
            'Speed':9,
            'Speed Change':10,
            'Direction':11,
            'Direction Change':12}

def withinSight(x1, y1, direction, x2, y2, sight_radius=10, central_angle=np.pi):
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

def getPositions(dataset, timestep, ID, width):
    positions = []
    row = dataset.iloc[timestep]
    for col in range(2, row.shape[0], width):
        data_col = row.iloc[col:col+width]
        if data_col.iloc[mapping['ID']] == ID:
            continue
        positions.append([data_col.iloc[mapping['Positionx']], data_col.iloc[mapping['Positiony']]])
    return positions

def getDensity(positions,x1,y1,direction):
    density = 0
    for position in positions:
        x2, y2 = position
        if withinSight(x1, y1, direction, x2, y2):
            density += 1
    return density

def getFront(dataset, timestep, width, positions,x1,y1,direction):
    min_distance = float('inf')
    front = None
    for position in positions:
        x2, y2 = position
        if withinSight(x1, y1, direction, x2, y2):
            dx = x2 - x1
            dy = y2 - y1
            dist = np.sqrt(dx ** 2 + dy ** 2)
            if min_distance >= dist:
                min_distance = dist
                front = (x2, y2)
    if not front:
        return [0,0]
    else:
        front_action = []
        row = dataset.iloc[timestep]
        for col in range(2, row.shape[0], width):
            data_col = row.iloc[col:col + width]
            if data_col.iloc[mapping['Positionx']] == front[0] and data_col.iloc[mapping['Positiony']] == front[1]:
                front_action.append(data_col.iloc[mapping['Speed']])
                front_action.append(data_col.iloc[mapping['Direction']])
        return front_action

class Xinjiekou(gym.Env):

    def __init__(self):
        # normalization
        self.max_speed = 1
        # actually pi before normalization
        self.max_direction = 1
        action_high = np.array([self.max_speed,self.max_direction])
        self.action_space = spaces.Box(low=-action_high, high=action_high, dtype=np.float32)
        movement_high = np.array([1,1])
        movement_low = np.array([0,-1])
        position_high = np.array([1,0])
        position_low = np.array([-1,-1])
        self.observation_space = spaces.Dict(
            {
                "position":spaces.Box(low=position_low,high=position_high,dtype=np.float32),
                "destination":spaces.Box(low=position_low,high=position_high,dtype=np.float32),
                'distance':spaces.Box(low=0, high=1, dtype=np.float32),
                'self movement':spaces.Box(low=movement_low,high=movement_high,dtype=np.float32),
                'front movement':spaces.Box(low=movement_low, high=movement_high, dtype=np.float32),
                'density':spaces.Box(low=0, high=1, dtype=np.float32),
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
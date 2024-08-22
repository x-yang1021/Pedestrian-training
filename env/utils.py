import numpy as np


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

def withinSight(x1, y1, direction, x2, y2, sight_radius=3, central_angle=np.pi):
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
        return [0.0,0.0]
    else:
        front_action = []
        row = dataset.iloc[timestep]
        for col in range(2, row.shape[0], width):
            data_col = row.iloc[col:col + width]
            if data_col.iloc[mapping['Positionx']] == front[0] and data_col.iloc[mapping['Positiony']] == front[1]:
                front_action.append(data_col.iloc[mapping['Speed']])
                front_action.append(data_col.iloc[mapping['Direction']])
        return front_action
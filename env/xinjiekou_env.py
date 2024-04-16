import pygame
import sys

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
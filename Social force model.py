import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# Simulation parameters
ROOM_SIZE = 10  # meters
EXIT_WIDTH = 0.4  # meters
NUM_PEDESTRIANS = 25
PEDESTRIAN_RADIUS = 0.2 # meters
TIME_STEP = 0.05  # seconds
MAX_SPEED = 1.5  # m/s

# Social force model parameters
A = 2500  # Strength of social repulsive force (N)
B = 0.08  # Characteristic distance of social repulsive force (m)
k = 120000  # Body compression coefficient (kg/s^2)
K = 240000  # Coefficient of sliding friction (kg/m/s)

# Define pedestrian class
class Pedestrian:
    def __init__(self, position, goal):
        self.position = np.array(position, dtype=float)
        self.velocity = np.zeros(2)
        self.goal = np.array(goal, dtype=float)
        self.reached_goal = False

    def desired_force(self):
        direction = self.goal - self.position
        distance = np.linalg.norm(direction)
        if distance > 0:
            direction = direction / distance
        desired_velocity = direction * MAX_SPEED
        force = (desired_velocity - self.velocity) / 0.5  # tau = 0.5
        return force

    def agent_repulsive_force(self, pedestrians):
        force = np.zeros(2)
        for other in pedestrians:
            if other is not self:
                direction = self.position - other.position
                distance = np.linalg.norm(direction)
                overlap = 2 * PEDESTRIAN_RADIUS - distance
                if overlap > 0:
                    if distance > 0:
                        direction = direction / distance
                    else:
                        direction = np.random.rand(2) - 0.5
                    force += direction * A * np.exp(overlap / B)
                    force += k * overlap * direction
                    force += K * overlap * np.dot(other.velocity - self.velocity, direction) * direction
        return force

    def wall_repulsive_force(self):
        force = np.zeros(2)
        # Add repulsive forces from walls
        # Left wall
        overlap = PEDESTRIAN_RADIUS - self.position[0]
        if overlap > 0:
            force[0] += k * overlap
        # Right wall
        overlap = PEDESTRIAN_RADIUS - (ROOM_SIZE - self.position[0])
        if overlap > 0:
            force[0] -= k * overlap
        # Bottom wall (excluding exit)
        if self.position[0] < (ROOM_SIZE - EXIT_WIDTH) / 2 or self.position[0] > (ROOM_SIZE + EXIT_WIDTH) / 2:
            overlap = PEDESTRIAN_RADIUS - self.position[1]
            if overlap > 0:
                force[1] += k * overlap
        return force

    def update(self, pedestrians):
        if self.reached_goal:
            return
        total_force = self.desired_force() + self.agent_repulsive_force(pedestrians) + self.wall_repulsive_force()
        acceleration = total_force  # mass is assumed to be 1
        self.velocity += acceleration * TIME_STEP
        speed = np.linalg.norm(self.velocity)
        if speed > MAX_SPEED:
            self.velocity = self.velocity / speed * MAX_SPEED
        self.position += self.velocity * TIME_STEP

        # Check if agent has reached the goal
        if self.position[1] <= 0:
            self.reached_goal = True
            self.velocity = np.zeros(2)

# Initialize pedestrians
def initialize_pedestrians(num_pedestrians, room_size, exit_width):
    pedestrians = []
    for _ in range(num_pedestrians):
        x = np.random.uniform(PEDESTRIAN_RADIUS, room_size - PEDESTRIAN_RADIUS)
        y = np.random.uniform(PEDESTRIAN_RADIUS + 1, room_size - PEDESTRIAN_RADIUS)
        goal = [room_size / 2, 0]
        pedestrians.append(Pedestrian([x, y], goal))
    return pedestrians

def run_simulation():
    pedestrians = initialize_pedestrians(NUM_PEDESTRIANS, ROOM_SIZE, EXIT_WIDTH)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, ROOM_SIZE)
    ax.set_ylim(0, ROOM_SIZE)
    exit_patch = plt.Rectangle(((ROOM_SIZE - EXIT_WIDTH) / 2, -0.5), EXIT_WIDTH, 0.5, color='green')
    ax.add_patch(exit_patch)
    pedestrian_patches = [plt.Circle(p.position, PEDESTRIAN_RADIUS, color='blue') for p in pedestrians]
    for patch in pedestrian_patches:
        ax.add_patch(patch)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    def init():
        time_text.set_text('')
        return pedestrian_patches + [time_text]

    def animate(frame):
        for p in pedestrians:
            p.update(pedestrians)
        for i, patch in enumerate(pedestrian_patches):
            if pedestrians[i].reached_goal:
                patch.set_visible(False)
            else:
                patch.center = pedestrians[i].position
        time_text.set_text(f'Time: {frame * TIME_STEP:.2f}s')
        return pedestrian_patches + [time_text]

    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=int(60 / TIME_STEP), interval=50, blit=True
    )
    anim.save('crowd_evacuation_simulation.gif', writer='imagemagick')
    plt.title('Crowd Evacuation Simulation with Arch Formation')
    plt.xlabel('Width (m)')
    plt.ylabel('Height (m)')
    plt.show()

run_simulation()

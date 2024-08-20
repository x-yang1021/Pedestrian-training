import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# Simulation parameters
NUM_AGENTS = 24
AGENT_RADIUS = 0.2  # meters
EXIT_WIDTH = 0.4  # meters
ROOM_WIDTH = 10  # meters
ROOM_HEIGHT = 10  # meters
TIME_STEP = 0.05  # seconds
MAX_SPEED = 1.5  # meters per second
MAX_TIME = 60  # seconds

# Define the exit position
EXIT_POSITION = np.array([ROOM_WIDTH / 2 - EXIT_WIDTH / 2, 0])

class Agent:
    def __init__(self, position):
        self.position = np.array(position, dtype=float)
        self.velocity = np.zeros(2)
        self.desired_speed = 1.3  # m/s, typical walking speed
        self.goal = np.array([ROOM_WIDTH / 2, -1])  # just outside the exit
        self.reached_exit = False

    def desired_force(self):
        direction = self.goal - self.position
        distance = np.linalg.norm(direction)
        if distance > 0:
            direction = direction / distance
        desired_velocity = direction * self.desired_speed
        force = (desired_velocity - self.velocity) / 0.5  # relaxation time tau = 0.5
        return force

    def agent_repulsive_force(self, agents):
        force = np.zeros(2)
        for other in agents:
            if other is not self:
                direction = self.position - other.position
                distance = np.linalg.norm(direction)
                overlap = 2 * AGENT_RADIUS - distance
                if overlap > 0:
                    if distance > 0:
                        direction = direction / distance
                    else:
                        direction = np.random.rand(2) - 0.5
                    force += direction * overlap * 200  # interaction strength
        return force

    def wall_repulsive_force(self):
        force = np.zeros(2)
        # Left wall
        overlap = AGENT_RADIUS - self.position[0]
        if overlap > 0:
            force[0] += 200 * overlap
        # Right wall
        overlap = AGENT_RADIUS - (ROOM_WIDTH - self.position[0])
        if overlap > 0:
            force[0] -= 200 * overlap
        # Bottom wall (excluding exit)
        if self.position[0] < EXIT_POSITION[0] or self.position[0] > (EXIT_POSITION[0] + EXIT_WIDTH):
            overlap = AGENT_RADIUS - self.position[1]
            if overlap > 0:
                force[1] += 200 * overlap
        # Top wall
        overlap = AGENT_RADIUS - (ROOM_HEIGHT - self.position[1])
        if overlap > 0:
            force[1] -= 200 * overlap
        return force

    def update(self, agents):
        if self.reached_exit:
            return
        total_force = self.desired_force() + self.agent_repulsive_force(agents) + self.wall_repulsive_force()
        acceleration = total_force  # mass is assumed to be 1
        self.velocity += acceleration * TIME_STEP
        speed = np.linalg.norm(self.velocity)
        if speed > MAX_SPEED:
            self.velocity = self.velocity / speed * MAX_SPEED
        self.position += self.velocity * TIME_STEP

        # Check if agent has reached exit
        if (EXIT_POSITION[0] <= self.position[0] <= EXIT_POSITION[0] + EXIT_WIDTH) and self.position[1] <= 0:
            self.reached_exit = True

def initialize_agents():
    agents = []
    while len(agents) < NUM_AGENTS:
        position = np.array([
            np.random.uniform(AGENT_RADIUS, ROOM_WIDTH - AGENT_RADIUS),
            np.random.uniform(AGENT_RADIUS + 1, ROOM_HEIGHT - AGENT_RADIUS)
        ])
        new_agent = Agent(position)
        overlap = False
        for agent in agents:
            distance = np.linalg.norm(new_agent.position - agent.position)
            if distance < 2 * AGENT_RADIUS:
                overlap = True
                break
        if not overlap:
            agents.append(new_agent)
    return agents

def run_simulation():
    agents = initialize_agents()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, ROOM_WIDTH)
    ax.set_ylim(0, ROOM_HEIGHT)
    exit_patch = plt.Rectangle((EXIT_POSITION[0], -0.5), EXIT_WIDTH, 0.5, color='green')
    ax.add_patch(exit_patch)
    agent_patches = [plt.Circle(agent.position, AGENT_RADIUS, color='blue') for agent in agents]
    for patch in agent_patches:
        ax.add_patch(patch)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    def init():
        time_text.set_text('')
        return agent_patches + [time_text]

    def animate(frame):
        for agent in agents:
            agent.update(agents)
        for i, patch in enumerate(agent_patches):
            if agents[i].reached_exit:
                patch.set_visible(False)
            else:
                patch.center = agents[i].position
        time_text.set_text(f'Time: {frame * TIME_STEP:.2f}s')
        return agent_patches + [time_text]

    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=int(MAX_TIME / TIME_STEP), interval=50, blit=True
    )

    plt.title('Crowd Evacuation Simulation using Social Force Model')
    plt.xlabel('Width (m)')
    plt.ylabel('Height (m)')
    plt.show()

if __name__ == "__main__":
    run_simulation()

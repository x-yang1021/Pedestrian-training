import numpy as np


def unit(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else np.zeros_like(vec)

class Pedestrian:
    def __init__(self, pos, vel, goal, shap_values, v_pref, direction):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.v_pref = v_pref
        self.direction = direction
        self.heading = np.array([np.cos(direction), np.sin(direction)])
        self.shap = shap_values

    def compute_driving_force(self,tau=0.5):
        dist = self.goal - self.pos
        weight_x = self.shap.get('self position x', 0)
        weight_y = self.shap.get('self position y', 0)
        force_x = weight_x * dist[0]
        force_y = weight_y * - dist[1]
        return np.array([force_x, force_y])

    def compute_repulsion_force(self, direction_flags):
        force = np.zeros(2)

        direction_map = {
            'front': self.heading,
            'left': np.array([-self.heading[1], self.heading[0]]),  # 90° left
            'right': np.array([self.heading[1], -self.heading[0]]),  # 90° right
            'back': -self.heading  # opposite
        }

        zone_keys = ['front', 'left', 'right', 'back']

        for i, active in enumerate(direction_flags):
            if not active:
                continue

            zone = zone_keys[i]
            dir_vec = unit(direction_map[zone])

            # SHAP-modulated repulsion
            if zone == 'front':
                shap_val = self.shap.get('front contact', 0)
            else:
                shap_val = self.shap.get('surrounding contact', 0)

            weight = shap_val
            force += weight * dir_vec

        return force * 0.1

    def compute_density_force(self, local_density):
        shap_density = self.shap.get('density', 0)
        return shap_density * local_density * self.heading * 0.1


    def update(self, force, dt=0.5):
        vel_vec = self.vel * self.heading
        vel_vec += force * dt
        self.pos += vel_vec * dt
        self.vel = np.linalg.norm(vel_vec)
        self.heading = unit(vel_vec)
        self.direction = np.arctan2(self.heading[1], self.heading[0])



import numpy as np

def unit(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else np.zeros_like(vec)

class Pedestrian:
    def __init__(self, pos, vel, goal, shap_values, id, v_pref, tau, A , B):
        self.id = id
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.v_pref = v_pref
        self.heading = unit(goal - pos)
        self.shap = shap_values
        self.tau = tau
        self.A = A
        self.B = B

    def compute_driving_force(self):
        v_desired = self.v_pref * unit(self.goal - self.pos)
        weight = 1 + self.shap.get('destination', 0)
        return weight / self.tau * (v_desired - self.vel)

    def compute_repulsion_force(self, others, A=2.0, B=1.0):
        force = np.zeros(2)
        for other in others:
            if other.id == self.id:
                continue
            direction = self.pos - other.pos
            dist = np.linalg.norm(direction)
            if dist < 0.001:
                continue
            direction = direction / dist
            # Angle between my heading and direction to other
            angle = np.dot(direction, self.heading)
            # Front contact emphasis
            if angle > 0.7:
                contact_shap = self.shap.get('front_contact', 0)
            else:
                contact_shap = self.shap.get('surrounding_contact', 0)
            weight = 1 + contact_shap
            f = weight * A * np.exp((0.6 - dist) / B) * direction
            force += f
        return force

    def compute_density_force(self, local_density):
        shap_density = self.shap.get('density', 0)
        return -shap_density * local_density * self.heading  # repelled from dense areas

    def compute_comfort_force(self):
        # Speed comfort
        v_mag = np.linalg.norm(self.vel)
        speed_diff = v_mag - self.v_pref
        shap_speed = self.shap.get('speed_change', 0)
        f_speed = -shap_speed * speed_diff * unit(self.vel)

        # Direction comfort
        current_heading = unit(self.vel) if v_mag > 0 else self.heading
        heading_diff = current_heading - self.heading
        shap_dir = self.shap.get('direction_change', 0)
        f_dir = -shap_dir * heading_diff

        return f_speed + f_dir

    def update(self, force, dt=0.5):
        acc = force  # assume unit mass
        self.vel += acc * dt
        self.pos += self.vel * dt



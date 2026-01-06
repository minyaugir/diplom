"""navigation_system.adaptive_tuning

Простейшая адаптивная подстройка ковариаций измерений (R) по величине инновации.

Идея:
- если инновация по GNSS резко выросла, повышаем R (меньше доверяем GNSS);
- если инновация стабильна, постепенно возвращаем R к номиналу.

Это не полноценный IAE (innovation-based adaptive estimation), но демонстрирует
принцип повышения устойчивости к выбросам и деградации GNSS.
"""

from __future__ import annotations

import numpy as np


class AdaptiveRTuner:
    def __init__(self,
                 R_pos_nom: np.ndarray,
                 R_vel_nom: np.ndarray,
                 k_up: float = 3.0,
                 k_down: float = 0.98,
                 max_scale: float = 50.0):
        self.R_pos_nom = R_pos_nom.copy()
        self.R_vel_nom = R_vel_nom.copy()
        self.R_pos = R_pos_nom.copy()
        self.R_vel = R_vel_nom.copy()
        self.k_up = float(k_up)
        self.k_down = float(k_down)
        self.max_scale = float(max_scale)

    def update(self, innov_pos: np.ndarray | None, innov_vel: np.ndarray | None):
        """Обновить масштаб R на основе инноваций."""
        scale = 1.0
        if innov_pos is not None:
            scale = max(scale, float(np.linalg.norm(innov_pos) / (np.sqrt(np.trace(self.R_pos_nom)) + 1e-9)))
        if innov_vel is not None:
            scale = max(scale, float(np.linalg.norm(innov_vel) / (np.sqrt(np.trace(self.R_vel_nom)) + 1e-9)))

        # Если слишком большая инновация — повышаем R, иначе плавно возвращаем к номиналу
        if scale > 2.0:
            factor = min(self.max_scale, self.k_up * scale)
            self.R_pos = self.R_pos_nom * factor
            self.R_vel = self.R_vel_nom * factor
        else:
            self.R_pos = self.R_pos * self.k_down + self.R_pos_nom * (1.0 - self.k_down)
            self.R_vel = self.R_vel * self.k_down + self.R_vel_nom * (1.0 - self.k_down)

        return self.R_pos, self.R_vel

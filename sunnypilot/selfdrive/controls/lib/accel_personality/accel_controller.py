"""
Copyright (c) 2021-, rav4kumar, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from cereal import custom
import numpy as np
from openpilot.common.realtime import DT_MDL
from openpilot.common.params import Params

AccelPersonality = custom.LongitudinalPlanSP.AccelerationPersonality
ACCEL_PERSONALITY_OPTIONS = [AccelPersonality.eco, AccelPersonality.normal, AccelPersonality.sport]

MAX_ACCEL_PROFILES = {
  AccelPersonality.eco:    [1.35, 1.10, 0.82, 0.62, 0.48, 0.36, 0.25, 0.12, 0.06],
  AccelPersonality.normal: [1.55, 1.30, 1.00, 0.78, 0.62, 0.47, 0.30, 0.13, 0.07],
  AccelPersonality.sport:  [1.80, 1.60, 1.28, 1.02, 0.82, 0.62, 0.40, 0.17, 0.09],
}
MAX_ACCEL_BREAKPOINTS = [0.0, 3.0, 5.0, 8.0, 12.0, 18.0, 24.0, 32.0, 42.0]

# Static decel floor per personality. MPC (COMFORT_BRAKE, A_CHANGE_COST) owns all
# brake feel shaping above this. This is a safety clip
MIN_ACCEL_BASE = {
  AccelPersonality.eco:    -0.40,
  AccelPersonality.normal: -0.62,
  AccelPersonality.sport:  -0.95,
}
MIN_ACCEL_DECAY = 0.030  # authority fades with speed: base * exp(-k * v_ego)

JERK_ACCEL = 0.55  # accel ceiling rate-limit (m/s² per s) — max only, not decel

_MIN_MAX_GAP = 0.05


class AccelPersonalityController:
  def __init__(self):
    self.params = Params()
    self.frame = 0
    self.first_run = True
    self.last_max_accel = 2.0
    self._cache_v: float | None = None
    self._cache_min: float = 0.0
    self._cache_max: float = 2.0

    val = self.params.get('AccelPersonality')
    self._accel_personality = val if val is not None else AccelPersonality.normal
    self._enabled = self.params.get_bool('AccelPersonalityEnabled')

  def update(self, sm=None):
    self.frame += 1
    self._cache_v = None
    if self.frame % max(1, int(1.0 / DT_MDL)) == 0:
      val = self.params.get('AccelPersonality')
      self._accel_personality = val if val is not None else AccelPersonality.normal
      self._enabled = self.params.get_bool('AccelPersonalityEnabled')

  @property
  def accel_personality(self) -> int:
    return self._accel_personality

  def get_accel_personality(self) -> int:
    return int(self._accel_personality)

  def set_accel_personality(self, personality: int):
    if personality in ACCEL_PERSONALITY_OPTIONS:
      self._accel_personality = personality
      self.params.put('AccelPersonality', personality)

  def cycle_accel_personality(self) -> int:
    current = self._accel_personality
    idx = ACCEL_PERSONALITY_OPTIONS.index(current) if current in ACCEL_PERSONALITY_OPTIONS else 0
    next_p = ACCEL_PERSONALITY_OPTIONS[(idx + 1) % len(ACCEL_PERSONALITY_OPTIONS)]
    self.set_accel_personality(next_p)
    return int(next_p)

  def _step(self, v_ego: float) -> tuple[float, float]:
    target_max = float(np.interp(v_ego, MAX_ACCEL_BREAKPOINTS, MAX_ACCEL_PROFILES[self._accel_personality]))

    if self.first_run:
      self.last_max_accel = target_max
      self.first_run = False
    else:
      a_step = JERK_ACCEL * DT_MDL
      target_max = float(np.clip(target_max, self.last_max_accel - a_step, self.last_max_accel + a_step))
      self.last_max_accel = target_max

    # Static floor — no rate limiting, MPC owns brake feel
    min_accel = float(MIN_ACCEL_BASE[self._accel_personality] * np.exp(-MIN_ACCEL_DECAY * v_ego))
    min_accel = min(min_accel, target_max - _MIN_MAX_GAP)

    return min_accel, target_max

  def get_accel_limits(self, v_ego: float) -> tuple[float, float]:
    v_ego = max(0.0, v_ego)
    if self._cache_v is not None and abs(self._cache_v - v_ego) < 0.01:
      return self._cache_min, self._cache_max
    self._cache_min, self._cache_max = self._step(v_ego)
    self._cache_v = v_ego
    return self._cache_min, self._cache_max

  def get_min_accel(self, v_ego: float) -> float:
    return self.get_accel_limits(v_ego)[0]

  def get_max_accel(self, v_ego: float) -> float:
    return self.get_accel_limits(v_ego)[1]

  def is_enabled(self) -> bool:
    return self._enabled

  def set_enabled(self, enabled: bool):
    self._enabled = bool(enabled)
    self.params.put_bool('AccelPersonalityEnabled', self._enabled)

  def toggle_enabled(self) -> bool:
    self.set_enabled(not self._enabled)
    return self._enabled

  def reset(self, personality: int = None):
    new_p = personality if personality in ACCEL_PERSONALITY_OPTIONS else AccelPersonality.normal
    self._accel_personality = new_p
    self.params.put('AccelPersonality', new_p)
    self.frame = 0
    self.last_max_accel = 2.0
    self._cache_v = None
    self.first_run = True

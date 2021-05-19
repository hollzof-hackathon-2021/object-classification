from typing import Tuple


class CoordinateScaler:
    def __init__(self, old_h: int, old_w: int, new_h: int, new_w: int):
        self._height_scale = new_h / old_h
        self._width_scale = new_w / old_w

    def scale(self, x, y) -> Tuple[int, int]:
        # 'x' is in width from left to right
        # 'y' is in height from top to bottom
        return int(x * self._width_scale), int(y * self._height_scale)

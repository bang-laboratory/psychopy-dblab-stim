#!/usr/bin/env python3
import pathlib

from psychopy.visual import ImageStim, Window
from psychopy.visual.basevisual import (
    BaseVisualStim,
    ColorMixin,
    ContainerMixin,
    DraggingMixin,
    # WindowMixin,
)


class CardStimulus(BaseVisualStim, DraggingMixin, ColorMixin, ContainerMixin):
    """Card task stimulus"""

    def __init__(
        self,
        win,
        card_size: float = 5,
        offset_left: tuple[float, float] = (-5, 0),
        offset_right: tuple[float, float] = (5, 0),
        aspect_ratio: tuple[float, float] = (0.6543561, 1),
    ):
        self.win: Window = win
        self.card_size = card_size
        self.offset_left = offset_left
        self.offset_right = offset_right
        self.aspect_ratio = aspect_ratio

        self.left_stim = ImageStim(
            win,
            size=(aspect_ratio[0] * card_size, aspect_ratio[1] * card_size),
            pos=offset_left,  # units=win.units
            name="Left card",
        )
        self.right_stim = ImageStim(
            win,
            size=(aspect_ratio[0] * card_size, aspect_ratio[1] * card_size),
            pos=offset_right,  # units=win.units
            name="Right card",
        )

    def draw(self):
        self.left_stim.draw()
        self.right_stim.draw()

    def update_images(self, left: str, right: str):
        self.left_stim.setImage(left)
        self.right_stim.setImage(right)

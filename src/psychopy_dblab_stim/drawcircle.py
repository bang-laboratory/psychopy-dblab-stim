#!/usr/bin/env python3

import base64
import io
import sys
from typing import Generator

import numpy as np
from psychopy import logging
from psychopy.core import Clock
from psychopy.event import Mouse
from psychopy.visual import Window
from psychopy.visual.basevisual import (
    BaseVisualStim,
    ColorMixin,
    ContainerMixin,
    DraggingMixin,
    # WindowMixin,
)
from psychopy.visual.circle import Circle
from psychopy.visual.shape import ShapeStim

# print("Imported drawcircle")
# logging.console.setLevel(logging.DEBUG)


def calc_angle_from_pos(coords: tuple, origin: tuple = (0, 0)) -> float:
    return np.arctan2(coords[0] - origin[0], coords[0] - origin[0])


class DrawingCircle(BaseVisualStim, DraggingMixin, ColorMixin, ContainerMixin):
    """Combined visual stimulus and mouse input

    Shows a thick circular outline and records "mouse" positions to emulate
    drawing. Intended to use with a wacom tablet.

    TODO(performance): The vertices for each segment is a numpy ndarray (good)
    but they are stored in a bunch of native python lists (bad for
    cache-locality at least)

    FIXME(performance): It could still be 1 vertice buffer with the segments
    just indexing in to it

    TODO(performance): does it even make sense to preallocate everything?
    """

    def __init__(
        self,
        win: Window,
        mouse: Mouse,
        clock: Clock,
        radius=4,
        fillColor="black",
        backgroundColor="grey",
        circleWidth=7,
        lineWidth=3,
        lineColor="white",
        #
        # unused/unpassed args to Circle():
        #
        # radius=0.5,
        # edges="circle",
        # units="",
        # lineWidth=1.5,
        # lineColor=None,
        # colorSpace="rgb",
        # pos=(0, 0),
        # size=1.0,
        # anchor=None,
        # ori=0.0,
        # opacity=None,
        # contrast=1.0,
        # depth=0,
        # interpolate=True,
        # draggable=False,
        # name=None,
        # autoLog=None,
        # autoDraw=False,
        # # legacy
        # lineRGB=undefined,
        # fillRGB=undefined,
        # color=undefined,
        # fillColorSpace=undefined,
        # lineColorSpace=undefined,
    ):

        # print("Entering DrawCircle.__init__")
        self.win: Window = win
        self.mouse: Mouse = mouse
        self.clock: Clock = clock

        self.radius = radius
        self.fillColor = fillColor
        self.backgroundColor = backgroundColor
        self.circleWidth = circleWidth
        self.lineWidth = lineWidth
        self.lineColor = lineColor

        # print("Basic setup for DrawCircle.__init__")

        self.inner_circle = Circle(win, radius=radius, fillColor=backgroundColor)
        self.outer_circle = Circle(
            win, radius=radius + circleWidth, fillColor=fillColor
        )

        self.reset_drawing()
        # print("Exiting DrawCircle.__init__")

    def handle_record_mouse(self):
        if self.mouse.getPressed()[0]:
            if not self.in_ongoing_line:
                self.in_ongoing_line = True
                self.drawing_vertices.append(GrowablePoints())
                self.segments_stim.append(
                    ShapeStim(
                        self.win,
                        closeShape=False,
                        lineWidth=self.__dict__["borderWidth"],
                        lineColor=self.lineColor,
                    )
                )
            elif len(self.drawing_vertices) == 0:
                self.drawing_vertices.append(GrowablePoints())
                self.segments_stim.append(
                    ShapeStim(
                        self.win,
                        closeShape=False,
                        lineWidth=self.__dict__["borderWidth"],
                        lineColor=self.lineColor,
                    )
                )

            pos = self.mouse.getPos()
            rotation = np.atan2(*pos)
            self.drawing_vertices[-1].append(
                (
                    *pos,  # idx 0,1
                    self.clock.getTime(),  # idx 2
                    rotation,  # idx 3
                    self._calc_rotation_progress(rotation),  # idx 4
                )
            )
        else:
            if self.in_ongoing_line:
                self.in_ongoing_line = False

    def _calc_rotation_progress(self, new_rotation: tuple) -> float | None:
        current_segment = self.drawing_vertices[-1]
        if len(current_segment) > 0:
            prev_rotation = current_segment[-1, 3]
            return (new_rotation - prev_rotation + np.pi) % (2 * np.pi) - np.pi

    def has_samples(self) -> bool:
        return (len(self.drawing_vertices) > 0) and (len(self.drawing_vertices[0]) > 0)

    def drawing_correct_direction(self, desired_direction: str) -> bool:
        if len(self.drawing_vertices[-1]) < 2:
            return False  # not enough samples to determine direction
        match desired_direction:
            case "clockwise":
                return self.drawing_vertices[-1][-1, 4] > 0
            case "counterclockwise":
                return self.drawing_vertices[-1][-1, 4] < 0
            case _:
                raise KeyError

    def reset_drawing(
        self,
    ):

        # reset vertice buffers
        self.drawing_vertices = []
        self.segments_stim = []

        self.in_ongoing_line = False

    def draw(self, win=None, keepMatrix=False):
        """once-per-frame tick, handles both drawing stimuli to screen and recording of drawing input

        NOTE: must be called every frame
        """
        self.outer_circle.draw(win, keepMatrix)
        self.inner_circle.draw(win, keepMatrix)
        self.handle_record_mouse()
        if len(self.segments_stim) > 0:
            self.segments_stim[-1].vertices = self.drawing_vertices[-1][:, 0:2]
            for stim in self.segments_stim:
                stim.draw(win, keepMatrix)


class GrowablePoints:
    """Extendable numpy array with preallocation starting with *capacity*
    elements allocated. When the user tries to set an item with a higher index
    than already allocated, double the size.
    """

    __slots__ = ("_buf", "_n")

    def __init__(self, capacity=64, cols=5, dtype=np.float64):
        self._buf = np.zeros((capacity, cols), dtype)
        self._n = 0

    def __setitem__(self, i, value):
        assert not isinstance(i, slice), "Slice assignments not supported"
        if i >= len(self._buf):  # geometric grow
            self._grow(max(i + 1, len(self._buf) * 2))
        if i >= self._n:
            self._n = i + 1
        self._buf[i] = value

    def _grow(self, cap):
        logger.debug(f"GROWING from {self._n} to {cap}")
        new = np.zeros((cap, self._buf.shape[1]), self._buf.dtype)
        new[: len(self._buf)] = self._buf
        self._buf = new

    @property
    def view(self):  # zero-copy valid region
        return self._buf[: self._n]

    def __getitem__(self, i):
        return self.view[i]

    def __len__(self):
        return self._n

    def append(self, value):
        self[self._n] = value


def serialise_segment_list(segments: list[GrowablePoints]) -> str:
    buf = io.BytesIO()
    np.savez(buf, *(segment.view for segment in segments))
    return base64.b64encode(buf.getvalue()).decode("ascii")


def deserialise_segment_list(str_) -> list[np.ndarray]:
    z = np.load(io.BytesIO(base64.b64decode(str_)))
    return [z[name] for name in z.files]

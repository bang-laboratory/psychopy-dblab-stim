#!/usr/bin/env python3

from typing import Generator

import numpy as np
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
        max_drawing_segments: int,
        max_drawing_samples: int,
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

        self.win: Window = win
        self.mouse: Mouse = mouse
        self.clock: Clock = clock

        self.max_drawing_segments = max_drawing_segments
        self.max_drawing_samples = max_drawing_samples
        self.radius = radius
        self.fillColor = fillColor
        self.backgroundColor = backgroundColor
        self.circleWidth = circleWidth
        self.lineWidth = lineWidth
        self.lineColor = lineColor

        self.inner_circle = Circle(win, radius=radius, fillColor=backgroundColor)
        self.outer_circle = Circle(
            win, radius=radius + circleWidth, fillColor=fillColor
        )
        self.drawing = [
            ShapeStim(win, closeShape=False, lineWidth=lineWidth, lineColor=lineColor)
            for _ in range(self.max_drawing_segments)
        ]

        self.reset_drawing(return_drawing=False)

    def handle_record_mouse(self):
        if self.mouse.getPressed()[0]:
            if not self.in_ongoing_line:
                self.in_ongoing_line = True

            pos = self.mouse.getPos()
            self.drawing_times[self.drawing_segments_count][
                self.drawing_samples_count[self.drawing_segments_count]
            ] = self.clock.getTime()
            self.drawing_vertices[self.drawing_segments_count][
                self.drawing_samples_count[self.drawing_segments_count], :
            ] = pos
            self._calc_rotation_progress(pos)
            self.drawing_samples_count[self.drawing_segments_count] += 1
        else:
            if self.in_ongoing_line:
                # end the line
                self.in_ongoing_line = False
                self.drawing_segments_count += 1

    def has_samples(self) -> bool:
        return self.drawing_samples_count[0] != 0

    def _calc_rotation_progress(self, pos: tuple):
        current_rotation = np.atan2(*pos)
        self.rotation_progress[self.drawing_segments_count][
            self.drawing_samples_count[self.drawing_segments_count]
        ] = current_rotation
        # initial_rotation = self.rotation_progress[self.drawing_segments_count][0]

    def count_rotations(self):
        """FIXME: performance when called every frame"""
        self.rotation_progress_percent = [
            [
                ((x + np.pi) / (2 * np.pi)) % 1
                for x in self.rotation_progress[segment]
                if x != 0
            ]
            for segment in range(self.drawing_segments_count + 1)
        ]
        self.derivative = [
            [
                b - a
                if (b - a) < 0.5 and (a - b) < 0.5
                else -1 + (b - a) % 1
                if b > a
                else (b - a) % 1
                for a, b in zip(
                    self.rotation_progress_percent[segment][:-1],
                    self.rotation_progress_percent[segment][1:],
                )
            ]
            for segment in range(self.drawing_segments_count + 1)
        ]
        self.segment_lengths = [sum(segment) for segment in self.derivative]
        # print(f"{self.segment_lengths=}")
        return self.segment_lengths

    def reset_drawing(
        self, return_drawing: bool = True
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]] | None:
        if return_drawing:
            # explicit list here to force the generator to resolve
            # since we want to delete the underlying data
            retval = list(self.filter_empty_vertices_and_times())
        else:
            retval = None

        # preallocate vertice buffers
        self.drawing_vertices = [
            np.zeros((self.max_drawing_samples, 2))
            for _ in range(self.max_drawing_segments)
        ]
        self.drawing_times = [
            np.zeros(self.max_drawing_samples) for _ in range(self.max_drawing_segments)
        ]

        self.rotation_progress = [
            np.zeros(self.max_drawing_samples) for _ in range(self.max_drawing_segments)
        ]

        # internal state
        self.drawing_samples_count = [0 for _ in range(self.max_drawing_segments)]
        self.drawing_segments_count = 0
        self.in_ongoing_line = False

        return retval

    def filter_empty_vertices(self) -> Generator[np.ndarray, None, None]:
        # count+1 here in order to draw ongoing, non-finalized lines
        for segment in range(self.drawing_segments_count + 1):
            if self.drawing_samples_count[segment] > 0:
                yield self.drawing_vertices[segment][
                    : self.drawing_samples_count[segment]
                ]
            else:
                # stop looking after we found 1 empty segment
                # since they are always filled from 0
                break

    def filter_empty_vertices_and_times(
        self,
    ) -> Generator[tuple[np.ndarray, np.ndarray, np.ndarray], None, None]:
        """Helper function for iterating over both verts and times in tandem"""
        for segment, vertice in enumerate(self.filter_empty_vertices()):
            yield (
                vertice,
                self.drawing_times[segment][: self.drawing_samples_count[segment]],
                self.rotation_progress[segment][: self.drawing_samples_count[segment]],
            )

    def draw(self, win=None, keepMatrix=False):
        self.outer_circle.draw(win, keepMatrix)
        self.inner_circle.draw(win, keepMatrix)
        self.handle_record_mouse()
        for segment, drawing_vertices in enumerate(self.filter_empty_vertices()):
            self.drawing[segment].vertices = drawing_vertices
            self.drawing[segment].draw(win, keepMatrix)

#!/usr/bin/env python3

import numpy as np
from psychopy import tools, visual
from psychopy.visual import dot


class DotStimMN(visual.dot.DotStim):
    """
    Extension of PsycoPy's DotStim method for show random motion
    kinematograms supporting the "Movshon/Newsome" method [1] (see overview in
    [2]) which interleaves different sets of dots to make
    it harder to focus on a single one. Set nSets > 1 to activate.

    [1] Shadlen, M. N., & Newsome, W. T. (2001). Neural basis of a perceptual
        decision in the parietal cortex (area LIP) of the rhesus monkey. Journal
        of neurophysiology, 86(4), 1916-1936.
    [2] Pilly, P. K., & Seitz, A. R. (2009). What a difference a parameter
        makes: A psychophysical comparison of random dot motion algorithms.
        Vision research, 49(13), 1599-1612.
    """

    def __init__(
        self,
        win: visual.Window,
        fixation_size: float,
        nSets: int = 1,
        save_dots: bool = False,
        **kwargs,
    ):
        self.fixation_size = fixation_size
        self.win = win
        self.nSets = nSets
        if save_dots:
            self.save_dots = []
        else:
            self.save_dots = None
        self.is_initialized = False  # workaround so we don't run the MN update_dotsXY code until initialized
        self.current_set = 0

        # these attribute setters need to be run early as part of
        # initialization (they do more than just store the variable)
        self.fieldSize = kwargs["fieldSize"]
        self.nDots = kwargs["nDots"]
        self.coherence = kwargs["coherence"]

        # NOTE: this also calls self._update_dotsXY() once
        super().__init__(win=win, **kwargs)
        self.is_initialized = True

        if nSets > 1:
            # Move everything related to dots into a nSets-length list
            # TODO: multi dimensional np arrays more efficient?
            self.multi_vertices = self.multi__verticesBases = self.multi__dotsXY = [
                self._newDotsXY(kwargs["nDots"]) for _ in range(nSets)
            ]
            self.multi__dotsSpeed = [
                np.ones(kwargs["nDots"], dtype=float) * kwargs["speed"]
                for _ in range(nSets)
            ]
            self.multi__deadDots = [
                np.zeros(kwargs["nDots"], dtype=bool) for _ in range(nSets)
            ]
            self.multi__dotsLife = [
                np.abs(kwargs["dotLife"]) * np.random.rand(kwargs["nDots"])
                for _ in range(nSets)
            ]
            self.multi__dotsDir = [
                np.random.rand(kwargs["nDots"]) * visual.dot._2pi for _ in range(nSets)
            ]
            for dots in self.multi__dotsDir:
                dots[self._signalDots] = kwargs["dir"] * visual.dot._piOver180

    def _update_dotsXY(self):
        if self.nSets == 1:
            return super()._update_dotsXY()
        if not self.is_initialized:
            return

        # cycle through all the variables that have to do with the dots...
        self.vertices = self.multi_vertices[self.current_set]
        self._verticesBase = self.multi_vertices[self.current_set]
        self._dotsXY = self.multi_vertices[self.current_set]
        self._dotsSpeed = self.multi__dotsSpeed[self.current_set]
        self._deadDots = self.multi__deadDots[self.current_set]
        self._dotsLife = self.multi__dotsLife[self.current_set]
        self._dotsDir = self.multi__dotsDir[self.current_set]
        self._signalDots = self.multi__signalDots[self.current_set]
        # print("sdsum", sum([sum(x) for x in self.multi__signalDots]))
        # print("sdsum2", sum(self._signalDots))
        # .. and run the regular update method on the current one
        self._modified_original_update_dotsXY()
        if self.save_dots is not None:
            self.save_dots.append(self.get_dots_for_saving())
        self.current_set = (self.current_set + 1) % self.nSets

    def _modified_original_update_dotsXY(self):
        """The user shouldn't call this - its gets done within draw()."""
        # Find dead dots, update positions, get new positions for
        # dead and out-of-bounds
        # renew dead dots
        if self.__dict__["dotLife"] > 0:  # if less than zero ignore it
            # decrement. Then dots to be reborn will be negative
            self._dotsLife -= 1
            self._deadDots[:] = self._dotsLife <= 0
            self._dotsLife[self._deadDots] = self.dotLife
        else:
            self._deadDots[:] = False

        # update XY based on speed and dir
        # NB self._dotsDir is in radians, but self.dir is in degs
        # update which are the noise/signal dots
        if self.signalDots == "different":
            #  **up to version 1.70.00 this was the other way around,
            # not in keeping with Scase et al**
            # noise and signal dots change identity constantly
            np.random.shuffle(self.multi__dotsDir[self.current_set])
            # and then update _signalDots from that
            self._signalDots = self.multi__dotsDir[self.current_set] == (
                self.__dict__["dir"] * visual.dot._piOver180
            )

        # update the locations of signal and noise; 0 radians=East!
        reshape = np.reshape
        if self.noiseDots == "walk":
            # noise dots are ~self._signalDots
            sig = np.random.rand(np.sum(int(~self._signalDots)))
            self._dotsDir[~self._signalDots] = sig * visual.dot._2pi
            # then update all positions from dir*speed
            cosDots = reshape(np.cos(self._dotsDir), (self.nDots,))
            sinDots = reshape(np.sin(self._dotsDir), (self.nDots,))
            self._verticesBase[:, 0] += self.speed * cosDots
            self._verticesBase[:, 1] += self.speed * sinDots
        elif self.noiseDots == "direction":
            # simply use the stored directions to update position
            cosDots = reshape(np.cos(self._dotsDir), (self.nDots,))
            sinDots = reshape(np.sin(self._dotsDir), (self.nDots,))
            self._verticesBase[:, 0] += self.speed * cosDots
            self._verticesBase[:, 1] += self.speed * sinDots
        elif self.noiseDots == "position":
            # update signal dots
            sd = self._signalDots
            sdSum = self._signalDots.sum()
            # print("sdSum", sdSum)
            cosDots = reshape(np.cos(self._dotsDir[sd]), (sdSum,))
            sinDots = reshape(np.sin(self._dotsDir[sd]), (sdSum,))
            self._verticesBase[sd, 0] += self.speed * cosDots
            self._verticesBase[sd, 1] += self.speed * sinDots
            # update noise dots
            self._deadDots[:] = self._deadDots + (~self._signalDots)

        # handle boundaries of the field

        if self.fieldShape in (None, "square", "sqr"):
            out0 = (
                np.abs(self._verticesBase[:, 0])
                > (self.fieldSize[0] + self.dotSize) / 2
            )
            out1 = (
                np.abs(self._verticesBase[:, 1])
                > (self.fieldSize[1] + self.dotSize) / 2
            )
            outofbounds = out0 + out1
        else:
            # transform to a normalised circle (radius = 1 all around)
            # then to polar coords to check
            # the normalised XY position (where radius should be < 1)
            normXY = self._verticesBase / 0.5 / (self.fieldSize + self.dotSize / 2)
            # add out-of-bounds to those that need replacing
            outofbounds = np.hypot(normXY[:, 0], normXY[:, 1]) > 1.0

        # update any dead dots
        nDead = self._deadDots.sum()
        if nDead:
            self._verticesBase[self._deadDots, :] = self._newDotsXY(nDead)

        # Reposition any dots that have gone out of bounds. Net effect is to
        # place dot one step inside the boundary on the other side of the
        # aperture.
        nOutOfBounds = outofbounds.sum()
        if nOutOfBounds:
            self._verticesBase[outofbounds, :] = self._newDotsXY(nOutOfBounds)
            # if self.fieldShape in (None, "square", "sqr"):
            #     self._verticesBase[outofbounds,:]=-self._verticesBase[outofbounds,:]
            # else:
            #     vectors = self._verticesBase[outofbounds,:]
            #     angles = np.arctan2(vectors[:, 1], vectors[:, 0])

            #     if self.dir not in [0, 180]: # left/right
            #         y = np.sin(-angles) * (self.fieldSize[0] + self.dotSize)/2
            #         coords = np.stack((self._verticesBase[outofbounds,0], y), axis=1)
            #     else: # up/down
            #         x = np.cos(-angles) * (self.fieldSize[0] + self.dotSize)/2
            #         coords = np.stack((x,self._verticesBase[outofbounds,1]), axis=1)

            #     self._verticesBase[outofbounds,:] = coords

        self.vertices = self._verticesBase / self.fieldSize

        # update the pixel XY coordinates in pixels (using _BaseVisual class)
        self._updateVertices()

    @tools.attributetools.attributeSetter
    def coherence(self, coherence: float):
        """Scalar between 0 and 1.

        Change the coherence (%) of the DotStim. This will be rounded according
        to the number of dots in the stimulus.

        :ref:`operations <attrib-operations>` are supported.
        """

        if not 0 <= coherence <= 1:
            raise ValueError("DotStim.coherence must be between 0 and 1")

        if self.nSets == 1:
            super().coherence = coherence
            return

        _cohDots = coherence * self.nDots
        # print(f"_cohDots: {_cohDots}")

        self.__dict__["coherence"] = round(_cohDots) / self.nDots
        self.multi__signalDots = [
            np.zeros(self.nDots, dtype=bool) for _ in range(self.nSets)
        ]
        for deck in range(self.nSets):
            self.multi__signalDots[deck][0 : int(self.coherence * self.nDots)] = True
        self._signalDots = self.multi__signalDots[self.current_set]

        # for 'direction' method we need to update the direction of the number
        # of signal dots immediately, but for other methods it will be done
        # during updateXY

        if not self.is_initialized:
            return
        # NB - AJS Actually you need to do this for 'walk' also
        # otherwise would be signal dots adopt random directions when the become
        # sinal dots in later trails
        # NOTE: update multi_X rather than X for use with nSets > 1
        if self.noiseDots in ("direction", "position", "walk"):
            for deck in range(self.nSets):
                self.multi__dotsDir[deck] = np.random.rand(self.nDots) * visual.dot._2pi
                self.multi__dotsDir[deck][self.multi__signalDots[deck]] = (
                    self.__dict__["dir"] * visual.dot._piOver180
                )
            self._dotsDir = self.multi__dotsDir[self.current_set]

        # print("_dotsDir are signal?", sum(self._dotsDir == (self.dir * _piOver180)))

    @tools.attributetools.attributeSetter
    def dir(self, dir: float):
        """float (degrees). direction of the coherent dots. :ref:`operations
        <attrib-operations>` are supported.
        """
        if self.nSets == 1:
            super().dir = dir
        # check which dots are signal before setting new dir
        signalDots = self._dotsDir == (self.__dict__["dir"] * visual.dot._piOver180)
        self.__dict__["dir"] = dir

        # dots currently moving in the signal direction also need to update
        # their direction
        for deck in range(self.nSets):
            self.multi__dotsDir[deck][signalDots] = (
                self.__dict__["dir"] * visual.dot._piOver180
            )
        self._dotsDir = self.multi__dotsDir[self.current_set]

    def get_dots_for_saving(self) -> np.ndarray:
        return self._verticesBase

    def _newDotsXY(self, nDots):
        """
        Don't start any new dots on top of the fixation cross
        """
        if self.fieldShape == "circle":
            fixation_size = (
                tools.monitorunittools.deg2pix(self.fixation_size, self.win.monitor)
                / (self.fieldSize[0])
                / 2
            )
            length = np.sqrt(np.random.uniform(fixation_size, 1, (nDots,)))
            angle = np.random.uniform(0.0, visual.dot._2pi, (nDots,))

            newDots = np.zeros((nDots, 2))
            newDots[:, 0] = length * np.cos(angle)
            newDots[:, 1] = length * np.sin(angle)

            newDots *= self.fieldSize * 0.5
        else:
            newDots = np.random.uniform(-0.5, 0.5, size=(nDots, 2)) * self.fieldSize

        return newDots

from __future__ import absolute_import, print_function

import collections
import math as m
import random

import numpy as np
import scipy as sy
from psychopy.data import MultiStairHandler, QuestHandler, StairHandler
from scipy.optimize import curve_fit


class ExtendedMultiStairHandler(MultiStairHandler):
    """ Handles 'pest' as the stairhandler """

    def _checkArguments(self):
        # Did we get a `conditions` parameter, correctly formatted?
        if not isinstance(self.conditions, collections.Iterable):
            raise TypeError(
                "`conditions` parameter passed to MultiStairHandler "
                "should be a list, not a %s." % type(self.conditions)
            )

        c0 = self.conditions[0]
        if not isinstance(c0, dict):
            raise TypeError(
                "`conditions` passed to MultiStairHandler should be a "
                "list of python dictionaries, not a list of %ss." % type(c0)
            )

        # Did `conditions` contain the things we need?
        params = list(c0.keys())
        if self.type.lower() not in ["simple", "quest", "pest", "vpest"]:
            raise ValueError(
                "MultiStairHandler `stairType` should be 'simple', "
                "'QUEST' or 'quest', not '%s'" % self.type
            )

        if "startVal" not in params:
            raise AttributeError(
                "MultiStairHandler needs a parameter called "
                "`startVal` in conditions"
            )
        if "label" not in params:
            raise AttributeError(
                "MultiStairHandler needs a parameter called"
                " `label` in conditions"
            )
        if self.type.lower() == "quest" and "startValSd" not in params:
            raise AttributeError(
                "MultiStairHandler needs a parameter called "
                "`startValSd` in conditions for QUEST staircases."
            )

    def _createStairs(self):
        for condition in self.conditions:
            # We create a copy, because we are going to remove items from
            # this dictionary in this loop, but don't want these
            # changes to alter the originals in self.conditions.
            args = dict(condition)

            # If no individual `nTrials` parameter was supplied for this
            # staircase, use the `nTrials` that were passed to
            # the MultiStairHandler on instantiation.
            if "nTrials" not in args:
                args["nTrials"] = self.nTrials

            if self.type.lower() == "simple":
                startVal = args.pop("startVal")
                thisStair = StairHandler(startVal, **args)
            elif self.type.lower() == "pest":
                startVal = args.pop("startVal")
                thisStair = PESTstandardHandler(startVal, **args)
            elif self.type.lower() == "vpest":
                startVal = args.pop("startVal")
                thisStair = PESTvirulentHandler(startVal, **args)
            elif self.type.lower() == "quest":
                startVal = args.pop("startVal")
                startValSd = args.pop("startValSd")
                thisStair = QuestHandler(startVal, startValSd, **args)

            # This isn't normally part of handler.
            thisStair.condition = condition

            # And finally, add it to the list.
            self.staircases.append(thisStair)
            self.runningStaircases.append(thisStair)


class PESTstandardHandler(StairHandler):
    """ Includes PEST heuristics by Taylor and Creelman [1967] """

    def __init__(
        self,
        startVal,
        nReversals=None,
        stepSizes=4,  # lin stepsize
        nTrials=100,
        extraInfo=None,
        method="2AFC",
        stepType="lin",
        minVal=None,
        maxVal=None,
        originPath=None,
        name="",
        autoLog=True,
        pest_w=1,
        **kwargs,
    ):
        nUp = 1
        nDown = 1
        self.applyInitialRule = False
        StairHandler.__init__(
            self,
            startVal=startVal,
            nReversals=nReversals,
            stepSizes=stepSizes,
            nTrials=nTrials,
            nUp=nUp,
            nDown=nDown,
            applyInitialRule=self.applyInitialRule,
            extraInfo=extraInfo,
            method=method,
            stepType=stepType,
            minVal=minVal,
            maxVal=maxVal,
            originPath=originPath,
            name=name,
            autoLog=autoLog,
            **kwargs,
        )
        self.currentStepSizeIdx = 0
        self.currentDirectionStepCount = 0
        countAlternative = int(method.lower().replace("afc", ""))
        pRandomGuess = 1.0 / countAlternative
        self.targetProb = pRandomGuess + (1.0 - pRandomGuess) / 2.0
        self.pest_w = pest_w
        self.isDoubled = False  # flag for Rule 3
        self.stepChangeidx = 0
        self.isConverged = False

    @property
    def isConverged(self):
        return self.__isConverged

    @isConverged.setter
    def isConverged(self, convergeFlag):
        self.__isConverged = convergeFlag

    def calculateNextIntensity(self):
        trialN = len(self.data)
        countTrials = trialN - self.stepChangeidx
        expectedCorrect = countTrials * self.targetProb
        upperBound, lowerBound = (
            expectedCorrect + self.pest_w,
            expectedCorrect - self.pest_w,
        )
        numCorrect = sum(self.data[self.stepChangeidx:])

        if numCorrect > upperBound:
            if self.currentDirection in ["up", "start"]:
                reversal = True
            else:
                # direction is 'down'
                reversal = False
            self.currentDirection = "down"
        elif numCorrect < lowerBound:
            if self.currentDirection in ["down", "start"]:
                reversal = True
            else:
                # direction is 'up'
                reversal = False
            # now:
            self.currentDirection = "up"
        else:
            reversal = False

        # add reversal info
        if reversal:
            self.currentDirectionStepCount = 0
            self.reversalPoints.append(self.thisTrialN)
            if not self.reversalIntensities and self.applyInitialRule:
                self.initialRule = True
            self.reversalIntensities.append(self.intensities[-1])

        # take the step
        if numCorrect > upperBound:
            self._intensityDec()
            self.stepChangeidx = trialN
        elif numCorrect < lowerBound:
            self._intensityInc()
            self.stepChangeidx = trialN

        if trialN >= self.nTrials:
            self.isConverged = False
            self.finished = True

        # compute the new step size
        if self._variableStep and not (lowerBound <= numCorrect <= upperBound):
            if reversal:
                # Rule 1: halve the stepsize after reversal
                self.currentStepSizeIdx += 1
                if self.currentStepSizeIdx >= len(self.stepSizes):
                    # we've gone beyond the list of step sizes
                    self.currentStepSizeIdx = len(self.stepSizes) - 1
                    self.isConverged = False
                    self.finished = (
                        True  # terminate when the stepsize falls below minimum
                    )
            else:
                self.currentDirectionStepCount += 1
                if self.currentDirectionStepCount >= 4:
                    # Rule 3
                    self.currentStepSizeIdx -= 1
                    self.isDoubled = True
                elif self.currentDirectionStepCount >= 3:
                    # Rule 4
                    if self.isDoubled:
                        self.isDoubled = False
                    else:
                        self.currentStepSizeIdx -= 1
                        self.isDoubled = True
                if self.currentStepSizeIdx < 0:
                    self.currentStepSizeIdx = 0
            self.stepSizeCurrent = self.stepSizes[self.currentStepSizeIdx]

        # show current status of the experiment on the console
        print(
            "Total trials: %d, Trials after step change: %d, Correct: %d (%.2f%%), Expected: %.2f (%.2f%%), Current Direction: %s, Current Stepsize: %.05f"
            % (
                trialN,
                countTrials,
                numCorrect,
                float(numCorrect) / countTrials * 100,
                expectedCorrect,
                self.targetProb * 100,
                self.currentDirection,
                self.stepSizes[self.currentStepSizeIdx],
            )
        )


class PESTvirulentHandler(StairHandler):
    """ Virulent PEST by Findlay [1978] """

    def __init__(
        self,
        startVal,
        nReversals=None,
        stepSizes=4,  # lin stepsize
        nTrials=100,
        extraInfo=None,
        method="2AFC",
        stepType="lin",
        minVal=None,
        maxVal=None,
        originPath=None,
        name="",
        autoLog=True,
        findlay_m=8,  # number of trials before decrementing the stepsize
        # value used in Wald test to determine whether a change in stimulus level should be made (larger values correspond to a slower but more accurate procedure)
        pest_w=0.5,
        **kwargs,
    ):
        nUp = 1
        nDown = 1
        self.applyInitialRule = False
        StairHandler.__init__(
            self,
            startVal=startVal,
            nReversals=nReversals,
            stepSizes=stepSizes,
            nTrials=nTrials,
            nUp=nUp,
            nDown=nDown,
            applyInitialRule=self.applyInitialRule,
            extraInfo=extraInfo,
            method=method,
            stepType=stepType,
            minVal=minVal,
            maxVal=maxVal,
            originPath=originPath,
            name=name,
            autoLog=autoLog,
            **kwargs,
        )
        self.currentStepSizeIdx = 0
        self.currentDirectionStepCount = 0
        countAlternative = int(method.lower().replace("afc", ""))
        pRandomGuess = 1.0 / countAlternative
        self.targetProb = pRandomGuess + (1.0 - pRandomGuess) / 2.0
        self.isDoubled = False  # flag for Rule 3
        self.stepChangeidx = 0
        self.stimuliLevelTrialCounts = []
        self.currentLevelTrialCount = 0
        self.findlay_m = findlay_m
        self.default_pest_w = pest_w
        self.isConverged = False

    @property
    def isConverged(self):
        return self.__isConverged

    @isConverged.setter
    def isConverged(self, convergeFlag):
        self.__isConverged = convergeFlag

    def calculateNextIntensity(self):
        trialN = len(self.data)
        countTrials = trialN - self.stepChangeidx
        expectedCorrect = countTrials * self.targetProb
        # Findlay's second modification
        if self.currentStepSizeIdx <= 0:
            self.pest_w = self.default_pest_w
        else:
            self.pest_w = int(
                (self.currentStepSizeIdx + 1) * self.default_pest_w
            )
        upperBound, lowerBound = (
            expectedCorrect + self.pest_w,
            expectedCorrect - self.pest_w,
        )
        numCorrect = sum(self.data[self.stepChangeidx:])
        self.currentLevelTrialCount += 1

        if numCorrect > upperBound:
            if self.currentDirection in ["up"]:
                reversal = True
            else:
                # direction is 'down'
                reversal = False
            self.currentDirection = "down"
        elif numCorrect < lowerBound:
            if self.currentDirection in ["down"]:
                reversal = True
            else:
                # direction is 'up'
                reversal = False
            # now:
            self.currentDirection = "up"
        else:
            reversal = False

        # Findlay's first modification
        stepChange = int(self.currentLevelTrialCount / self.findlay_m)

        if trialN >= self.nTrials:
            self.isConverged = False
            self.finished = True
            print("\tvPEST finished: The max number of trials is reached.")

        # add reversal info
        if reversal:
            self.currentDirectionStepCount = 0
            self.reversalPoints.append(self.thisTrialN)
            if not self.reversalIntensities and self.applyInitialRule:
                self.initialRule = True
            if self.intensities:
                self.reversalIntensities.append(self.intensities[-1])
            else:
                self.reversalIntensities.append(self.startVal)

        # compute the new step size
        if self._variableStep and not (lowerBound <= numCorrect <= upperBound):
            if reversal:
                # Rule 1: halve the stepsize after reversal
                self.currentStepSizeIdx += 1
                if self.currentStepSizeIdx >= len(self.stepSizes):
                    self.currentStepSizeIdx = len(self.stepSizes) - 1
                    self.isConverged = True
                    self.finished = True
                    print(
                        "\tvPEST finished: The next step size at the reversal point is smaller than the minimum."
                    )
            else:
                self.currentDirectionStepCount += 1
                if self.currentDirectionStepCount >= 4:
                    # Rule 3
                    self.currentStepSizeIdx -= 1
                    self.isDoubled = True
                elif self.currentDirectionStepCount >= 3:
                    # Rule 4
                    if self.isDoubled:
                        self.isDoubled = False
                    else:
                        self.currentStepSizeIdx -= 1
                        self.isDoubled = True
                if self.currentStepSizeIdx < 0:
                    self.currentStepSizeIdx = 0
            self.stepSizeCurrent = self.stepSizes[self.currentStepSizeIdx]

        # take the step
        if not (lowerBound <= numCorrect <= upperBound):
            self.stepSizeCurrent = self.stepSizes[self.currentStepSizeIdx]
            if numCorrect > upperBound:
                self.stimuliLevelTrialCounts.append(
                    self.currentLevelTrialCount
                )
                self.stepChangeidx = trialN
                # if self._nextIntensity <= self.minVal:
                #     self.isConverged = True
                #     self.finished = True
                #     print(
                #         "\tvPEST finished: Trying to take a step down at the min stim level."
                #     )
                self._intensityDec()
            elif numCorrect < lowerBound:
                self.stimuliLevelTrialCounts.append(
                    self.currentLevelTrialCount
                )
                self.stepChangeidx = trialN
                if self._nextIntensity >= self.maxVal:
                    self.isConverged = True
                    self.finished = True
                    print(
                        "\tvPEST finished: Trying to take a step up at the max stim level."
                    )
                self._intensityInc()
            self.currentLevelTrialCount = 0
        else:
            # lowerBound <= numCorrect <= upperBound is True
            # stop the experiment if the stepsize gets too small
            effectiveStepIdx = self.currentStepSizeIdx + stepChange
            self.currentStepSizeIdx = effectiveStepIdx
            if effectiveStepIdx >= len(self.stepSizes):
                self.currentStepSizeIdx = len(self.stepSizes) - 1
                self.stepSizeCurrent = self.stepSizes[self.currentStepSizeIdx]
                self.isConverged = True
                self.finished = True
                print(
                    "\tvPEST finished: The effective stepsize reached the minimum while testing the same level."
                )

        # show current status of the experiment on the console
        print(
            "Total trials: %d, Trials after step change: %d, Correct: %d (%.2f%%), Expected: %.2f (%.2f%%) [%.2f,%.2f], Current Direction: %s, Current Stepsize: %.02f/256, Current w: %.1f, Stepchange: %d"
            % (
                trialN,
                countTrials,
                numCorrect,
                float(numCorrect) / countTrials * 100,
                expectedCorrect,
                self.targetProb * 100,
                upperBound,
                lowerBound,
                self.currentDirection,
                self.stepSizes[self.currentStepSizeIdx] * 256,
                self.pest_w,
                stepChange,
            )
        )


# class PESTvirulentHandlerDavid(StairHandler):
#     ''' Virulent PEST by Findlay [1978] '''
#     def __init__(self,
#                  startVal,
#                  nReversals=None,
#                  stepSizes=4,  # lin stepsize
#                  nTrials=100,
#                  extraInfo=None,
#                  method='2AFC',
#                  stepType='lin',
#                  minVal=None,
#                  maxVal=None,
#                  originPath=None,
#                  name='',
#                  autoLog=True,
#                  findlay_m=8,  # number of trials before decrementing the stepsize
#                  **kwargs):
#         nUp = 1
#         nDown = 1
#         self.applyInitialRule = False
#         StairHandler.__init__(self,
#                               startVal=startVal,
#                               nReversals=nReversals,
#                               stepSizes=stepSizes,
#                               nTrials=nTrials,
#                               nUp=nUp,
#                               nDown=nDown,
#                               applyInitialRule=self.applyInitialRule,
#                               extraInfo=extraInfo,
#                               method=method,
#                               stepType=stepType,
#                               minVal=minVal,
#                               maxVal=maxVal,
#                               originPath=originPath,
#                               name=name,
#                               autoLog=autoLog,
#                               **kwargs)
#         self.currentStepSizeIdx = 0
#         self.currentDirectionStepCount = 0
#         countAlternative = int(method.lower().replace("afc", ""))
#         pRandomGuess = 1.0 / countAlternative
#         self.targetProb = pRandomGuess + (1.0 - pRandomGuess) / 2.0
#         self.isDoubled = False  # flag for Rule 3
#         self.stepChangeidx = 0
#         self.stimuliLevelTrialCounts = []
#         self.currentLevelTrialCount = 0
#         self.findlay_m=findlay_m
#         self.pest_w=0.5
#         self.isConverged = False

#     @property
#     def isConverged(self):
#         return self.__isConverged

#     @isConverged.setter
#     def isConverged(self, convergeFlag):
#         self.__isConverged = convergeFlag

#     def calculateNextIntensity(self):
#         trialN = len(self.data)
#         countTrials = trialN - self.stepChangeidx
#         expectedCorrect = countTrials * self.targetProb
#         # Findlay's second modification
#         if self.currentStepSizeIdx <= 0:
#             self.pest_w = 0.5
#         else:
#             self.pest_w = int((self.currentStepSizeIdx+1) / 2.0)
#         if self.currentLevelTrialCount < 2 and self.pest_w == 0.5:        # correct for hyper-sensitivity to first incorrect responses
#             self.pest_w = 1.0
#         upperBound, lowerBound = \
#             expectedCorrect + self.pest_w, \
#             expectedCorrect - self.pest_w
#         numCorrect = sum(self.data[self.stepChangeidx:])
#         self.currentLevelTrialCount += 1

#         if numCorrect > upperBound:
#             if self.currentDirection in ['up', 'start']:
#                 reversal = True
#             else:
#                 # direction is 'down'
#                 reversal = False
#             self.currentDirection = 'down'
#         elif numCorrect < lowerBound:
#             if self.currentDirection in ['down', 'start']:
#                 reversal = True
#             else:
#                 # direction is 'up'
#                 reversal = False
#             # now:
#             self.currentDirection = 'up'
#         else:
#             reversal = False

#         # Findlay's first modification
#         stepChange = int(self.currentLevelTrialCount / self.findlay_m)

#         effectiveStepIdx = self.currentStepSizeIdx + stepChange

#         # setup the step
#         decrement = False
#         increment = False
#         if not (lowerBound <= numCorrect <= upperBound):
#             effectiveStepIdx = self.currentStepSizeIdx + stepChange
#             if effectiveStepIdx < 0:
#                 effectiveStepIdx = 0
#             elif effectiveStepIdx >= len(self.stepSizes):
#                 effectiveStepIdx = len(self.stepSizes) - 1
#                 self.finished = True
#             self.currentStepSizeIdx = effectiveStepIdx
#             self.stepSizeCurrent = self.stepSizes[self.currentStepSizeIdx]
#             if numCorrect > upperBound:
#                 self.stimuliLevelTrialCounts.append(self.currentLevelTrialCount)
#                 decrement = True
#                 self.stepChangeidx = trialN
#             elif numCorrect < lowerBound:
#                 self.stimuliLevelTrialCounts.append(self.currentLevelTrialCount)
#                 increment = True
#                 self.stepChangeidx = trialN

#         if trialN >= self.nTrials:
#             self.isConverged = False
#             self.finished = True

#         # add reversal info
#         if reversal:
#             self.currentDirectionStepCount = 0
#             self.reversalPoints.append(self.thisTrialN)
#             #if not self.reversalIntensities and self.applyInitialRule:
#             #    self.initialRule = True
#             if self.intensities:
#                 self.reversalIntensities.append(self.intensities[-1])
#             else:
#                 self.reversalIntensities.append(self.startVal)

#         # compute the new step size
#         if self._variableStep and not (lowerBound <= numCorrect <= upperBound):
#             self.currentLevelTrialCount = 0
#             if reversal:
#                 # Rule 1: halve the stepsize after reversal
#                 self.currentStepSizeIdx += 1
#                 if self.currentStepSizeIdx >= len(self.stepSizes):
#                     self.currentStepSizeIdx = len(self.stepSizes) - 1
#                     self.isConverged = True
#                     self.finished = True
#             else:
#                 self.currentDirectionStepCount += 1
#                 if self.currentDirectionStepCount >= 4:
#                     # Rule 3
#                     self.currentStepSizeIdx -= 1
#                     self.isDoubled = True
#                 elif self.currentDirectionStepCount >= 3:
#                     # Rule 4
#                     if self.isDoubled:
#                         self.isDoubled = False
#                     else:
#                         self.currentStepSizeIdx -= 1
#                         self.isDoubled = True
#                 if self.currentStepSizeIdx < 0:
#                     self.currentStepSizeIdx = 0
#             self.stepSizeCurrent = self.stepSizes[self.currentStepSizeIdx]

#         # stop the experiment if the stepsize gets too small
#         if lowerBound <= numCorrect <= upperBound:
#             if effectiveStepIdx >= len(self.stepSizes):
#                 self.currentStepSizeIdx = len(self.stepSizes) - 1
#                 self.stepSizeCurrent = self.stepSizes[self.currentStepSizeIdx]
#                 self.isConverged = True
#                 self.finished = True

#         # take the step:
#         if decrement:
#             self._intensityDec()
#         if increment:
#             self._intensityInc()

#         # show current status of the experiment on the console
#         print("Total trials: %d, Trials after step change: %d, Correct: %d (%.2f%%), Expected: %.2f (%.2f%%), Current Direction: %s, Current Stepsize: %d, Current w: %.1f, Stepchange: %d, Next Intensity: %s, Reversal: %s" %
#             (trialN,
#              countTrials,
#              numCorrect,
#              float(numCorrect)/countTrials*100,
#              expectedCorrect,
#              self.targetProb*100,
#              self.currentDirection,
#              self.stepSizes[self.currentStepSizeIdx],
#              self.pest_w,
#              stepChange,
#              self._nextIntensity,
#              reversal))


def weibullcdf(x, lmda, k, numAlt):
    # lmda: stimulus intensity where the slope is the maximum
    # k: steepness of the function
    p_guess = 1 / numAlt
    return p_guess + (1 - p_guess) * (1 - m.exp(-m.pow(x / lmda, k)))


def weibullcdf_np(x, lmda, k, numAlt):
    p_guess = 1 / numAlt
    return p_guess + (1 - p_guess) * (1 - np.exp(-np.power(x / lmda, k)))


# def inv_weibullcdf(p, lmda, k):
#     return lmda * m.pow(-m.log(1 - p), 1 / k)


def logistic(x, beta_0, beta_1, numAlt):
    return 1 / numAlt + (1 / (1 + np.exp(-(x - beta_0) * beta_1))) * (
        1 - 1 / numAlt
    )


def main():
    """ Performs a simulation of the staircase procedure """
    import matplotlib.pyplot as plt

    converged_level = []
    ntrials = []
    hasConverged = []
    psychometric_function = "logistic"
    numAlt = 4
    procedure = "vpest"
    numRepetitions = 1000
    for iter in range(numRepetitions):
        currentVal = 64 / 256
        if psychometric_function == "weibull":
            lmda = 20.0
            k = 2.0
        elif psychometric_function == "logistic":
            beta_0 = 0.125  # transition point (should converge to this value)
            beta_1 = (
                20.0  # steepness (higher values are easier for convergence)
            )
            if iter == 0:
                x = np.array(range(numRepetitions + 1)) / numRepetitions
                y = logistic(x, beta_0, beta_1, numAlt)
                plt.plot(x, y)
                plt.show()
        if procedure == "quest":
            a = QuestHandler(
                m.log(currentVal - 1),
                m.log(100 - 1),
                nTrials=100,
                pThreshold=1.0 / numAlt + (1.0 - 1 / numAlt) / 2.0,
            )
        elif procedure == "vpest":
            a = PESTvirulentHandler(
                currentVal,
                # stepSizes=[0.05, 0.025, 0.0125, 0.00625, 0.003, 0.0015, 0.001],
                stepSizes=[10 / 256, 7 / 256, 5 / 256, 3 / 256, 2 / 256],
                method="%dAFC" % numAlt,
                stepType="lin",
                minVal=0.0,
                maxVal=0.5,
                findlay_m=12,
                pest_w=0.5,
                nTrials=120,
            )
        # a = PESTstandardHandler(currentVal, stepSizes=[16,8,4,2,1], method='%dAFC' % numAlt, stepType='lin', minVal=1, maxVal=80, pest_w=1)
        # a = StairHandler(currentVal, stepSizes=[8,4,2,1], method='%dAFC' % numAlt, stepType='lin', minVal=1, maxVal=80, nDown=3, nUp=1, nTrials=30)
        level = []
        response = []
        n = 0
        try:
            while True:
                level.append(currentVal)
                if psychometric_function == "weibull":
                    cdf = weibullcdf(currentVal, lmda, k)
                elif psychometric_function == "logistic":
                    cdf = logistic(currentVal, beta_0, beta_1, numAlt)
                print("Testing %.2f, p(detection) = %.2f" % (currentVal, cdf))
                randomnumber = random.uniform(0, 1)
                if randomnumber <= cdf:
                    response.append(1)
                else:
                    response.append(0)
                n += 1
                a.addResponse(response[-1])
                if procedure == "quest":
                    currentVal = (
                        m.exp(a._nextIntensity) - 1
                    )  # use this if testing with QUEST procedure
                    if m.exp(a.sd()) - 1 < 0.4:
                        break
                elif procedure == "vpest":
                    currentVal = a._nextIntensity
                    if a.finished:
                        break
        except StopIteration:
            print("Converged at %f after %d iterations" % (currentVal, n))

        if procedure == "quest":
            hasConverged.append(1.0)
        elif procedure == "vpest":
            hasConverged.append(int(a.isConverged))

        # fit curve to the collected data
        levels_np = np.array(level, dtype=float)
        responses_np = np.array(response, dtype=float)
        par0 = sy.array([10.0, 0.25])  # initial values
        if psychometric_function == "weibull":
            par, _ = curve_fit(
                weibullcdf_np, levels_np, responses_np, par0, maxfev=100000
            )
        elif psychometric_function == "logistic":
            def logistic_fit(x, param1, param2): return logistic(
                x, param1, param2, numAlt
            )
            par, _ = curve_fit(
                logistic_fit, levels_np, responses_np, par0, maxfev=100000
            )  # This is not MLE, it's a basic curve_fitting
        print(f"Estimated parameter values: {par}")
        converged_level.append(currentVal)
        ntrials.append(n)

    converged_level_np = np.array(converged_level, dtype=float)
    fig = plt.figure()
    plt.hist(converged_level_np, bins=20)
    plt.title("Histogram of converged values")
    plt.show()
    mean_converged_level = np.mean(converged_level_np)
    std_converged_level = np.std(converged_level_np)
    sterr_converged_level = std_converged_level / m.sqrt(len(converged_level))

    ntrials_np = np.array(ntrials, dtype=float)
    mean_ntrials = np.mean(ntrials_np)
    std_ntrials = np.std(ntrials_np)
    sterr_ntrials = std_ntrials / m.sqrt(len(ntrials))

    hasConverged_np = np.array(hasConverged, dtype=float)
    mean_hasConverged = np.mean(hasConverged_np)
    std_hasConverged = np.std(hasConverged_np)
    sterr_hasConverged = std_hasConverged / m.sqrt(len(hasConverged))
    print(
        "Mean converged level:  %.2f (sterr: %.2f, min: %.2f, max: %.2f)"
        % (
            mean_converged_level,
            sterr_converged_level,
            np.min(converged_level_np),
            np.max(converged_level_np),
        )
    )
    print("Std converged level:  %.2f" % std_converged_level)
    print(
        "Mean number of trials: %.2f (sterr: %.2f, min: %.2f, max: %.2f)"
        % (mean_ntrials, sterr_ntrials, np.min(ntrials_np), np.max(ntrials_np))
    )
    print("Std number of trials: %.2f" % std_ntrials)
    print(
        "Ratio of converged staircases: %.2f (sterr: %.2f, min: %.2f, max: %.2f)"
        % (
            mean_hasConverged,
            sterr_hasConverged,
            np.min(hasConverged_np),
            np.max(hasConverged_np),
        )
    )
    # plt.plot(levels_np, responses_np, 'ro')
    # plt.plot(np.sort(levels_np), weibullcdf_np(np.sort(levels_np), par[0], par[1]))
    # plt.show()


def simulation2():
    startVal = 64 / 256
    vpest = PESTvirulentHandler(
        startVal,
        stepSizes=[10 / 256, 7 / 256, 5 / 256, 3 / 256, 2 / 256],
        method="2AFC",
        stepType="lin",
        minVal=0.0,
        maxVal=0.5,
        findlay_m=8,
        nTrials=100,
    )
    last_level = None
    while not vpest.finished:
        if vpest._nextIntensity != last_level:
            last_level = vpest._nextIntensity
            print()
        print(
            f"Testing level: {vpest._nextIntensity}({vpest._nextIntensity*256}/256): ",
            end="",
        )
        response = int(input())
        vpest.addResponse(response)


if __name__ == "__main__":
    main()  # used for debugging
    # simulation2()

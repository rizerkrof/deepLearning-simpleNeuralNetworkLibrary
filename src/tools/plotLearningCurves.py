#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

def plotLearningCurves(fitHistory):
    learningCurves = pd.DataFrame(fitHistory)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8,8))
    fig.suptitle("loss curves")
    learningCurves.filter(regex='loss').plot(ax=ax1)
    learningCurves.filter(regex='accuracy').plot(ax=ax2)
    ax2.set_xlabel('iterations')
    fig.tight_layout()
    fig.show()

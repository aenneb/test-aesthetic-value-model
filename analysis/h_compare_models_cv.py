#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Do a preliminary, descriptive comparison between models.

Created on Tue Oct 26 2021
Last updated Feb 04 2022: adapted for cv fits
@author: aennebrielmann
"""

import os
import sys
import pandas as pd
from scipy import stats

# %% ---------------------------------------------------------
# Specify directories; settings
# ------------------------------------------------------------
os.chdir('..')
home_dir = os.getcwd()
dataDir = home_dir + '/'
figDir = dataDir + 'figures/'
plot = True

# %% ---------------------------------------------------------
# load data, functions
# ------------------------------------------------------------
df = pd.read_csv(dataDir + 'perParticipantResults_cv.csv')
sys.path.append((home_dir))
import figureFunctions as ff

# %% ---------------------------------------------------------
# Re-format df to long
# ------------------------------------------------------------
longDf = pd.wide_to_long(df,
                         stubnames=['med_rmse', 'avg_rmse'],
                         i=df.columns[:18],
                         j='model', sep='_', suffix='.*')
longDf = longDf.reset_index()
longDf['model'] = longDf['model'].str.replace('_results_', '', regex=True)
longDf['model'] = longDf['model'].str.replace('ure', '', regex=True)
longDf['model'] = longDf['model'].str.replace('glm_rating ~', '', regex=True)

# %% ---------------------------------------------------------
# Create a scatter plot comparing all median model RMSEs to LOO-avg RMSE
# ------------------------------------------------------------
plotDf = longDf.copy()
plotDf['model'] = pd.Categorical(plotDf.model)
remap_cats = {'fit_paper_draftmodel_3vggfeatalphazero_wVzero': '3D no learning',
              'fit_paper_draftmodel_2vggfeatalphazero_wVzero': '2D no learning',
              'fit_paper_draftmodel_4vggfeat': '4D full',
              'fit_paper_draftmodel_3vggfeat': '3D full',
              'fit_paper_draftmodel_2vggfeat': '2D full',
              'LOOavg': 'LOO-average'}
plotDf['model'] = plotDf.model.cat.rename_categories(remap_cats)
modelNames = plotDf.model.unique().tolist()
modelNames.pop(modelNames.index('LOO-average'))

if plot:
    fig = ff.scatter_model_comparison(plotDf, 'med_rmse', modelNames,
                                      baselineModel='LOO-average',
                                      minVal=0, maxVal=.5)
    fig.suptitle('Median RMSE')
    fig.axes[0].set_ylabel('3D no learning')
    fig.axes[1].set_ylabel('2D no learning')
    fig.axes[2].set_ylabel('3D full')
    fig.axes[3].set_ylabel('2D full')
    for axcount in range(4):
        fig.axes[axcount].set_title('')
    fig.savefig(figDir + 'model comparison scatters.png',
                dpi=300, bbox_inches="tight")

# %% ---------------------------------------------------------
# Get and print median RMSE, r per model
# ------------------------------------------------------------
tableDf = pd.DataFrame(longDf.groupby(['model'])['med_rmse'].median())
tableDf['RMSE SD'] = longDf.groupby(['model'])['med_rmse'].std()
# latex format because we want to paste this into the manuscript
print(tableDf.to_latex(float_format="{:0.3f}".format))

# %% ---------------------------------------------------------
# run paired t-tests betweeen all models and LOO-avg.
# just as a preliminary check on possibly meaningful differences.
# ------------------------------------------------------------
for model in plotDf.model.unique():
    _, p = stats.shapiro(plotDf[plotDf.model == model]['med_rmse'])
    if model != 'LOO-average':
        if p > 0.05:
            print('Comparison to ' + model)
            print(stats.ttest_rel(plotDf[plotDf.model == 'LOO-average']['med_rmse'],
                                  (plotDf[plotDf.model == model]['med_rmse'])))
        else:
            print('Comparison to ' + model)
            print(stats.wilcoxon(plotDf[plotDf.model == 'LOO-average']['med_rmse'],
                                 (plotDf[plotDf.model == model]['med_rmse'])))

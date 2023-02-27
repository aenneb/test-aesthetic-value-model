# Testing a computational model of aesthetic value
Data and analysis files for an experiment testing the viability of [a computational model of aesthetic value developed by Brielmann and Dayan (2022)](https://psycnet.apa.org/fulltext/2022-78031-001.html)(go to psyarxiv for the [free preprint](https://psyarxiv.com/eaqkc/) if you do not have accesss).
Analyses and plotting functions used here depend on the [main package containing the core functions of the computational model of aesthetic value](https://github.com/aenneb/intro-aesthetic-value-model).

For convenience, this repo contains copies of the necessary core package files. So, you do not need to download any additional scripts from another repo and all paths are set so that you can run everything with this repo only. 

If you want to replicate all analyses, you can run all scripts located in the "analysis" folder in alphabetic order. However, I would recommend not to re-fit the model for all participants unless you have a substantial amount of time and/or processing power.

# Running the code

If you want run the code, try the following steps:

```
git clone https://github.com/aenneb/test-aesthetic-value-model.git 
cd test-aesthetic-value-model 
python -m venv .env 
.env/Scripts/activate # windows
# or 
source .env/bin/activate # mac/linux
pip install -r requirements.txt 
cd analysis 
python a_get_complete_data.py #etc
```

You should then be able to run any of the analysis scripts. You will also note the `requirements.txt` file which lists all packages that are required to run all scripts. NOTE that pingouin may not be compatible with the latest numpy version. To run the analyses that require pingouin, downgrade your environment's numpy version to < 1.24.

## Which code to run and what it does

In general, you can run all analysis code in alphabetical order as indicated by the prefix in front of the .py files to replicate all analyses.

NOTE that running model fits is expensive and may take a long time.

a_get_complete_data.py reads in raw data files and converts them into usable .csv files.

b_get_descriptives_ratings.py gives you a few summary stats about the ratings (ICC, distribution per image, checks normal distribution).

c_get_descriptives_participants.py gives you information about individual participants' answers at the end of the experiment and the correlations between ratings and initial slider positions.

d_fit_custom_model_ratings_cv.py fits one version of the model to all participants and saves the results.

e_evaluate_costum_model_fits.py fetches the individual model fitting results and stores them in a summary .csv for later use.

f_merge_fit_results_participantInfo_cv.py merges the summary of the model fit results of all variations with the general participant info and saves it as 'perParticipantResults_cv.csv'.

g_get_leaveOneOut_Average_predictions.py computes the predictions for the LOO-average baseline model and appends it to the same file that contains summaries of all model fits (perParticipantResults_cv.csv).

h_compare_models_cv.py does a preliminary model comparison with t-test and identifies the nominally best model per participant.

i_sim_scrambled_trial_order_with_refit.py simulates data using fit model parameters in a scrambled order, evaluates the resulting RMSE and R-squared and saves these results.

j_eval_sim_scrambled_trial_order_with_refit.py evaluates the results of the simulated scrambled order fits and compares the results to the true-order fits.

k_model_selection_cv.py performs proper model comparison using rmmANOVAs.


## Meaning and use of other scripts

The main directory also contains some .py files that contain helper functions and necessary components from the core package.

simExperiment.py contains the most basic model functionality and calculates the aesthetic value.

fitPilot.py coontains model functionality that is needed for fitting rating data.

figureFunctions.py is a collection of functions that help with visualization. Here, we only use scatter_model_comparison().

plot_images_inVGGspace.py visualizes the location of the stimulus images in reduced DNN-feature space. It is not used for anything presented in the paper but I thought you might be curious about what these DNN features mean (like I have been).


# Folder content

The main directory contains the pre-processed data from all participants as well as the results of the simulated, random-order refits.

It also contains the two components of the core aesthetic value model you need to reproduce the analyses.

In addition, it contains "map_imgName_imgIdx.csv" which provides a safe way to ensure that DNN-derived image features are mapped unto the correct images in the data files. 

As a bonus, it contains the "plot_images_inVGGspace.py" script which visualizes the location of the stimulus images in reduced DNN-feature space.

You will also note the requirements.txt file which lists all packages that are required to runn all scripts. NOTE that pingouin may not be compatible with the latest numpy version. To run the analyses that require pingouin, downgrade your environment's numpy version to < 1.24.

## analysis

Contains all scripts needed to reproduce the analyses reported in the paper as well as the code for re-creating the figures. 

NOTE that the model fitting may take a substantial amount of time, so if you do not want to tweak this, you might want to use the available model fitting results from the "results" folder.

## data_experiment

Contains all raw data that was collected. 

NOTE that these files also contain participants that were excluded from analyses.

## results

Contains the .csv and .npy files for all preliminary and final results of the analyses that are the basis for statistical analyses and plots.

## experiment_code

Contains the complete code for running the experiment as described in the paper *except* for the core jspsych files. Download jspsych 6.3.1 and save the folder inside the experiment_code folder to run the experiment.

## VGG_features

Contains the feature values and their PCA-reduced version for all stimuli based on vanilla, pretrained VGG-16.

## ResNet50_features

Contains the feature values and their PCA-reduced version for all stimuli based on vanilla, pretrained ResNet50.

## figures

Contains the figures from the paper.



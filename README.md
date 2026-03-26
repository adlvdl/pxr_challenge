# pxr_challenge
Personal repository on my participation on the OpenADME PXR blind challenge. 
This challenge contains two strands: property prediction and structure elucidation. 
I plan to participate only in the property prediction strand.
More information about the challenge can be found at 
https://openadmet.ghost.io/announcing-the-next-openadmet-blind-challenge-predicting-pxr-induction/

# Why am I doing this?

- To create an educational resource on the use of ML for drug discovery
- To create a demonstration of my work process

# Technical setup

- Use of open software: rdkit, matplotlib, chemprop, scikit-learn, ...
- Work on marimo notebooks
- Use Claude Code to write code, not direct the analysis
- Post thoughts, progress and ideas about the challenge at my blog https://adlvdl.github.io/blog.html

# Expected workflow

This is a preliminary plan of the work to be done:

1. Data analysis and preprocessing: downloading the data, exploring general SAR character of the dataset, think whether any compound data is better removed or altered for training, explore public datasets that might enhance the prediction

2. Generate data splits: as we will be comparing ML model predictions to chose the best one, I plan to follow the process outline in a recent paper by the Polaris group to generate 5x5 cross-validated data splits to make statistical sound comparisons. The main point to explore is whether to generate random, scaffold or time based splits

3. Generate single task baseline: this will likely be a comparison from different fingerprints used on RF and XGB as well as chemprop

4. Explore multitask settings and/or finetuning models for property prediction: previous challenges showed the impact of external data to improve predictions so this will be an important aspect. It is also possible that different data available in the challenge can be modeled separately in this manner

5. Provide predictions for analog set 1: this held out dataset will be unblinded in the middle of the challenge. This will provide important information for how the different models performed prospectively and might suggest alterations before submitting predictions for analog set 2

6. Provide predictions for analog set 2: this will be the final step and will be the set on which the participant will be ranked.
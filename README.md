# Navigating the Disagreement Space

This repository contains all data-processing scripts, analytical code, and supplementary materials for the study Navigating the Disagreement Space: A Case Study on Persistent YouTube Users’ Interactions in Immigration-Related Discussions.
The project examines how persistent YouTube commenters navigate contentious political environments, how their stance positions evolve, and how interactional choices shape patterns of polarization.

The dataset has been uploaded here for anonymity during peer review. Upon acceptance, it will be made publicly available in a Zenodo repository under the Creative Commons Attribution 4.0 International license.

For peer-review transparency, we also upload the Annotation Guidelines used to augment our test set. However, we do not own them: they are publicly available from the repository of the paper
A Pipeline for the Analysis of User Interactions in YouTube Comments: A Hybridization of LLMs and Rule-Based Methods


## Repository Folder Descriptions
### Analysis

Contains all scripts and notebooks used for the empirical analyses reported in the paper. This includes:
1) stance-based user clustering
2) media-exposure and interaction-pattern analyses
3) temporal activity and peak-event alignment tests
4) toxicity and linguistic-style analyses
5) polarization-trajectory estimation for general audiences and persistent users

These materials reproduce the statistical results, figures, and tables presented in the manuscript.

### Data

Provides processed and intermediate datasets required to run the analysis pipeline, including:
1) cleaned comment datasets and stance-labeled outputs
2) channel-level metadata (political leaning, category, upload timelines)
3) recency-bias audit results assessing stability of video retrieval over time
4) aggregated yearly stance distributions for channels and user groups

NB: Raw identifiers are anonymized in accordance with ethical guidelines.

### Data_Crawling&Annotation

Includes scripts used for YouTube data collection and manual annotation resources, specifically:

1) YouTube API crawling scripts for retrieving videos, comments, and partial thread structures
2) procedures for reconstructing parent–child comment chains

### ML_Model_Scripts

Contains all machine-learning components for stance detection and toxicity scoring, including:
1) NLI-based stance-classification pipeline (DeBERTa-v3-large and RoBERTa baselines)
2) hyperparameter configurations and best settings identified via Optuna
3) training and evaluation scripts
4) inference scripts used to scale stance predictions to the full dataset

### Results

Stores computed outputs and final metrics used in the paper, divided by Research Questions, such as:
1) stance distributions by channel type and year
2) user-level stance summaries and cluster assignments
3) toxicity attribute aggregations
4) intermediate result tables used for plotting and statistical tests

These files enable transparent replication of the reported findings.

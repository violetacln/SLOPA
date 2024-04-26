# SLOPA
Statistical Learning for Overestimated Population Adjustments

Here we share our workflows/R-scripts for using classifiers/ensembles and evaluating them, in order to predict the status of individuals (as present or absent) from a given population, provided a set of their attributes (demographic, social) are known.
The earliest [script](https://github.com/violetacln/SLOPA/blob/main/SLOPA_pub.R) established the exploratory and modeling steps necessary to fit and evaluate any such ML models, ensembles included. An additional [code]( https://github.com/MargheritaZ/ML-Census2021) was focused on the random forest classifier and optimised for several performance measures.

Two papers presented at [NTTS-2021](https://coms.events/NTTS2021/en/) and [NSM-2022](https://www.nsm2022.is/) conferences are shared here as well as our [paper](https://content.iospress.com/articles/statistical-journal-of-the-iaos/sji230090) in the Statistical Journal of the IAOS (2023). This is an open access paper and uses the code shown in the [file above](https://github.com/violetacln/SLOPA/blob/main/ML_classif_forSJIAOS.R). It is the most general and it compares multiple ML models, it reports on their performance measures and associated uncertainty and chooses the optimum regimes, depending on the goal of the analysis.



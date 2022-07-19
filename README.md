# Ellipsoidal conformal inference for Multi-Target Regression

INTRODUCTION
------------

This code is a supplementary material for the paper ["Ellipsoidal conformal inference for Multi-Target Regression"](https://copa-conference.com/papers/COPA2022_paper_7.pdf) accepted in COPA 2022. It includes necessary source code for reproducing synthetic data results, as well as 3 real data set results (other data sets were omitted for size restrictions).

FOLDERS
-------

This project is composed of 3 folders:

- code : contains main code for synthetic data and real data sets' experiments.
- utilities : contains files with essential functions used in the "code" folder for simulating synthetic data, real data preprocessing, empirical and ellipsoidal non-conformity measures' functions.
- input : contains a folder for each real data set with its data and config files, and stores log and visualization files when produced by the "code" folder. There are 3 data sets to test with (residential building, enb and scpf).

EXECUTING FILES
---------------

To reproduce the results in the paper, you can run the files in the "code" folder :

- conformal_synthetic_data : produces results for synthetic data. Visualization files are stored in the "code" folder.
- conformal_real_data : produces results for real data sets in the "input" folder. You can run the "run.sh" or "run.bat" file to execute the code.

REQUIREMENTS
------------

pandas==0.24.2
matplotlib==3.0.3
tensorflow==2.0.1
scipy==1.2.1
copulae==0.6.0
numpy==1.16.2
scikit_learn==1.0.2

NOTES
-----

- Code related to data preprocessing and empirical copula non-conformity measures is taken from [our previous work.](https://github.com/M-Soundouss/CopulaConformalMTR) 
- Results may slightly vary from presented results in the paper.

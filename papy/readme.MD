This is a single file implementation.

All functions are moved into pa.py.

"plotSurface.py" is a separate function for users to plot interactive surface plots on each variables.

The rest other files used to be modules called by pa.py. The files are in the folder "files_from_oldversion".

To run the Power Analysis tool:

simple usage: python pa.py TutorialData.csv 8, TutorialData.csv is input test data set, can be replaced by actual data set name, 8 means the first 8 variables, which can be a range, e.g., 8-16

full usage: python pa.py TutorialData.csv 2-9 0:100:500 0.05:0.05:0.7 20 4

"0:100:500" means the range of sample sizes from 0 to 500 (not inclusive) with interval of 100

"0.05:0.05:0.7" means the range of effect sizes from 0.05 to 0.7 (not inclusive) with interval of 0.05

"20" is an integer number of repeats.

"4" is an integer number as number of CPU cores to use.

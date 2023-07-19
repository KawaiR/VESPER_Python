# VESPER_Python

VESPER_Python is a computational tool which uses a local vector based algorithm as well as secondary structure matching that can accurately identify the global and local alignment of cryo-electron microscopy (EM) maps. This version provides an option to use probability maps generated by Emap2sec+, another software developed by the Kihara Lab, for secondary strcuture matching.

Emap2sec+: https://github.itap.purdue.edu/kiharalab/Emap2secPlus_prob \
Original VESPER: https://github.com/kiharalab/VESPER \
Data: https://drive.google.com/drive/folders/1os3i7YmlMew3dmfovlzRU4WVuJ7nKzPX?usp=sharing

## Requirements:
* matplotlib
* mrcfile
* numba
* numpy
* pyFFTW
* scipy

## Software Compatibility:
To ensure all versions are compatible, run:
```
pip3 install -r requirements.txt --user
```
or
```
pip install -r requirements.txt --user
```

## VESPER_Python protocol

VESPER_Python uses vector based algorithm same as the previous version and along with that it also uses secondary structure mapping in deciding the best superimposition between two maps. The secondary structure mapping is done with the help of Emap2sec+. Emap2sec+ gives the probability of the secondary structure of each voxel of the input mrc map. These probabilities are used to make probability mrc maps which contain the probabilities of each structure. 

## Probability Map Generation Pipeline
Using secondary structure matching requires probability predictions generated by Emap2sec+ at each voxel. To generate the two probability map files required, follow the process below:
1. Run Emap2sec+ modes 5 and 6 to generate a .npy file.
2. Run the interpolation script using the following command. The interpolation script has two modes: insert and graph </br>
a. insert: only interpolates the map given. </br>
```
Usage: interpolation.py insert [-h] -f F -s S
---Options---
-h, --help  show this help message and exit
  -f F        Map array (.npy)
  -s S        Name for new map to be saved (string)
```
b. graph: interpolates the given map and saves each secondary structure prediction map individually.
```
Usage: interpolation.py graph [-h] -f F -s S -m M
---Options---
-h, --help  show this help message and exit
  -f F        Map array (.npy)
  -s S        Name for individual maps (string)
  -m M        Name of density map file. NOTE: VOXEL SPACIING MUST BE 1. (.mrc)
```
## Usage
(1) Mode 1: No additional file generation is required. Only the maps used for matching are needed to run the program. (No secondary structure matching.)
```
Usage: main.py orig [-h] -a A -b B [-t T] [-T T] [-g G] [-s S] [-A A] [-N N] [-S S] [-M M] [-E E]
---Options---
-h, --help  show this help message and exit
  -a A        MAP1.mrc (large)
  -b B        MAP2.mrc (small)
  -t T        Threshold of density map1
  -T T        Threshold of density map2
  -g G        Bandwidth of the Gaussian filter def=16.0, sigma = 0.5*[value entered]
  -s S        Sampling voxel spacing def=7.0
  -A A        Sampling angle spacing def=30.0
  -N N        Refine Top [int] models def=10
  -S S        Show topN models in PDB format def=false
  -M M        V: vector product mode (default)
              O: overlap mode
              C: Cross Correlation Coefficient
              Mode P: Pearson Correlation Coefficient
              Mode L: Laplacian Filtering Mode
  -E E        Evaluation mode of the current position def=false
  -gpu gpuid  GPU ID to use, if not present, use CPU
  -nodup      Remove duplicate positions using heuristics
```
(2) Mode 2: Probability maps generated for each secondary structure in both maps used for alignment must be generated using Emap2sec+ first and then converted into .npy format using the interpolation script before running this mode. (With Secondary structure matching.)
```
Usage: main.py prob [-h] -a A -npa NPA -b B -npb NPB [-t T] [-T T] [-g G] [-s S] [-A A] [-N N] [-S S] [-M M] [-E E]
[-vav VAV] [-vstd VSTD] [-pav PAV] [-pstd PSTD]
---Options---
 -h, --help  show this help message and exit
  -a A          MAP1.mrc (large)
  -npa NPA      Numpy array for Predictions for map 1
  -b B          MAP2.mrc (small)
  -npb NPB      Numpy array for Predictions for map 2
  -alpha ALPHA  The weighting parameter
  -t T          Threshold of density map1
  -T T          Threshold of density map2
  -g G          Bandwidth of the Gaussian filter def=16.0, sigma = 0.5*[value entered]
  -s S          Sampling voxel spacing def=7.0
  -A A          Sampling angle spacing def=30.0
  -N N          Refine Top [int] models def=10
  -S S          Show topN models in PDB format def=false
  -M M          V: vector product mode (default)
                O: overlap mode
                C: Cross Correlation Coefficient Mode
                P: Pearson Correlation Coefficient Mode
                L: Laplacian Filtering Mode
  -E E          Evaluation mode of the current position def=false
  -vav VAV      Pre-computed average for density map
  -vstd VSTD    Pre-computed standard deviation for density map
  -pav PAV      Pre-computed average for probability map
  -pstd PSTD    Pre-computed standard deviation for probability map
```
## Identify the best fitting of two EM maps.
Run VESPER in either probability or original mode.
#### Output Format: 
By default, VESPER_python writes the vector information for each of top 10 models after local refinement into VESPER output. Vector information for the first model starts with two lines like the ones shown below.
```
Overlap= 0.02156028368794326 76/3525 CC= 0.012639913 PCC= -0.09214708 Scoreplusprob= 0.00019255988782371103 Scoreprobonly= 3.5619823131948194e-05
Score=  58.680508
```
Score shows the DOT score, which is the summation of dot products of matched vectors between two maps. Scoreplusprob shows the normalized score which is the sum of the normalised dot score as well as the normalised probability dot score. Scoreprobonly represents only the normalized probability dot score.

## Visualizing the transformed positions of the query maps.
VESPER_Python gives the pdb files of the top 10 alignments. They are named as "model_x.pdb". These are the query maps transformed to the best alignment position according to VESPER_Python. These can be visualised uing PyMol or Chimera.


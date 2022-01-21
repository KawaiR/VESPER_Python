# VESPER_Python (*README still in progress*)

VESPER_Python is a computational tool which uses local vector based algorithm as well as secondary structure matching that can accurately identify the global and local alignment of cryo-electron microscopy (EM) maps. 

## Requirements:
* matplotlib
* mrcfile
* numba
* numpy
* pyFFTW
* scipy

To ensure all versions are compatible, run:
```
pip3 install -r requirement.txt --user
```

## VESPER_Python protocol

VESPER_Python uses vector based algorithm same as the previous version and along with that it also uses secondary structure mapping in deciding the best superimposition between two maps. The secondary structure mapping is done with the help of Emap2sec+. Emap2sec+ gives the probability of the secondary structure of each voxel of the input mrc map. These probabilities are used to make probability mrc maps which contain the probabilities of each structure. 

## Input File generation

VESPER_Python uses input mrc maps as well as probability maps. The probability maps are generated using Emap2sec+.

# VESPER_Python

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
## Usage:
```
python3 main.py --a [Map A (large)] --b [Map B (small)] [Other options]
--Options--
--t [float ]  : Threshold of density map1 def=0.000
--T [float ]  : Threshold of density map2 def=0.000
--g [float ]  : Bandwidth of the gaussian filter
                def=16.0, sigma = 0.5*[float]
--s [float ]  : Sampling grid space def=7.0
--A [float ]  : Sampling Angle interval def=30.0
--c [int   ]  : Number of cores for threads def=2
--N [int   ]  : Refine Top [int] models def=10
--S [boolean] : Show topN models in PDB format def=false
--M [string]  : V: vector product mode (default)
                O: overlap mode
                C: Cross Correlation Coefficient Mode
                P: Pearson Correlation Coefficient Mode
                L: Laplacian Filtering Mode
--E [boolean] : Evaluation mode of the current position def=false
```

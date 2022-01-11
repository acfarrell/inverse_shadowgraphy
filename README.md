# Inverse Shadowgraphy

This repository contains methods and classes for the inverse shadowgraphy analysis outlined in Kasim et. al's 2017 PRE (whose code can be found on Github [here](https://github.com/mfkasim1/invert-shadowgraphy)), using Python instead of Matlab to resolve dependency issues when running on clusters.

## Requirements
This code has been written to require as few external libraries as possible to avoid compatibility issues. The requirements are:

-Python version 3.8 or higher (3.7 and lower will be considerably slower due to changes in the multiprocessing module)
-natplotlib
-numpy
-scipy
-cv2

If you're using conda to manage virtual environments, the requirements.txt file can be used to create a new conda environment compatible with this code by running
```
conda create --name new_env --file requirements.txt</code>
```

## Running
The input parameters for this script are all defined in files located in the `input` directory. The file `input/example_input.py` demonstrates all the input parameters and how they should be used. Once you have an input file in the input directory, you can run the code from the main directory using the line (using `input/example_input.py` as an example):
```
python3 invert.py example_input
```
The code will then run with the given parameters and save the deflection potential, displacement maps, site locations, and weights into a `.npz` file that can be loaded into other scripts for plotting and analysis. It also saves an image showing the final power diagram on the source plane and the displacement of the cell centroids relative to the initial site positions.

![Example plot of the site displacement image.](/output/site_displacement_test.png)

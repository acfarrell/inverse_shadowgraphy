'''
Define all the input parameters for the analysis in this file!

This example file  will explain each parameter and define a quick sample analysis that can
be run to check that the script is working.
'''
# Define the path to the data you want to analyze
data_directory = 'test_data/'

# Define the source plane image file. Passing None will assume a uniform background
source_image_file = None

# Rather than initializing new sites on the source plane, you can use an existing
# power diagram as the source by feeding in the saved data from a previous run
use_previous_source = True
# If True, give the path to the .npz file to load:
previous_source = 'test_data/output/test_output_test.npz'

# Define the target plane image file you want to analyze.
target_image_file = 'test.png'

# Define the number of sites you'd like to use
# If passed None or omitted the default is 0.8 * N_pixels in the image
N_sites = 100

# Define the threshold for Lloyd relaxation on the source plane
# If passed None or omitted the default is .1 (10%)
lloyd_threshold = 0.1

# By default all the output will be saved as a single .npz numpy file (to read into other python scripts)
# To also save the raw data in several .txt files, set save_txt to True
save_txt = False

# Define the directory where you want the results of the analysis saved
output_directory = 'test_data/output/'

# Define the suffix you'd like to put at the end of any output filenames
# For example output_suffix = 'test' would change 'log.txt' to 'log_test.txt'
# Passing None will default to using the current date and time when the script is run
output_suffix = 'test'

# Give the filename where you'd like the script to log its system output.
# If passed None the code will print to the console as it runs.
log_file = None

# Set the desired logging level for the run. The possible levels are:
# DEBUG: Detailed information needed when diagnosing problems
# INFO: Confirmation that thigs are working as expectsd
# WARNING: Indications that something unexpected has happened but the code is still running
# ERROR: Something bad happened and a function was not able to be completed
# CRITICAL: Something really bad happened and the whole program cannot finish running

# The default logging level is INFO
log_level = 'INFO'


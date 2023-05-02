import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import os
import importlib
import logging as log
import cv2
import multiprocessing as mp
import shutil
# Import the optimal transport module
from src import optimal_transport as ot
importlib.reload(ot)

def initialize():
    '''
    Get all the user defined parameters from the input file, resolve any errors or missing values,
    configure the logging properties, and return a dictionary of all the loaded variables
    '''
    # Record the start time
    import time
    start_time = time.strftime("%Y%m%d_%T", time.localtime())

    if len(sys.argv) != 2:
        raise Exception("Wrong number of arguments given! Expected to be run with 'python3 invert.py <input_filename>'")
    else:
        # Initialize a dictionary to store all the parameters
        p = {}
        # Grab the name of the input file
        input_fname = 'input.' + str(sys.argv[1])
        
        # Import the input file so we can get all the user defined variables
        init = importlib.import_module(input_fname)
        # Get the output directory to save to (do this first in case log file
        # is inside an output directory that doesn't exist yet)
        try:
            output_dir = init.output_directory
        except AttributeError:
            # If no output directory is given, set it to a directory called 'output' in the same
            # directory as invert.py
            output_dir = os.path.dirname(os.path.realpath(__file__))+'/output'
        
        # If the output directory doesn't exist, make it
        try:
            os.mkdir(output_dir)
            outdir_status = 'new'
        except OSError:
            outdir_status = 'existing'
            
        # Initialize the log file first so we can catch any errors in the log while loading parameters
        try:
            log_file = init.log_file
        except AttributeError:
            log_file = None
        try:
            log_level = getattr(log, init.log_level.upper(), None)
        except AttributeError:
            log_level = getattr(log, 'INFO', None)
        
        
        if log_file is None:
            log.basicConfig(level=log_level, format='')
            log.info('No log file was provided, printing the log to console...')
        else:
            
            log.basicConfig(filename = log_file, filemode='w', level=log_level, format='')
        log.info(f'Beginning analysis at {start_time}...\n')
        log.info(f'All output will be saved to the {outdir_status} directory {output_dir}\n')
        
        # Copy the input deck into the output directory
        shutil.copy('input/'+str(sys.argv[1])+'.py', output_dir+'input-deck.py')
        log.info(f'Input deck copied into output directory as input-deck.py')
        
        try:
            data_dir = str(init.data_directory)
        except AttributeError:
            # If no directory is given for the data files, assume they're in the current working directory
            data_dir = str(os.getcwd())
            log.debug(f'No directory was given for the data, assuming the current working directory {data_dir}...\n')
        
        # Load the source plane image file name
        try:
            source_file = init.source_image_file
        except AttributeError:
            source_file = None
        if source_file is not None:
            source_file = data_dir + source_file
        
        # Load the target plane image file name
        try:
            target_fname = init.target_image_file
            target_file = data_dir + target_fname
            log.info(f'Target image: {target_file}')
        except AttributeError:
            log.exception('No target image file was given!\n')
        
        # Load the target image into a numpy array using cv2
        target_image = cv2.imread(target_file, cv2.IMREAD_GRAYSCALE)
        if target_image is None:
            log.exception(f'No target image was found at {target_file}!\n')
        else:
            target_image = target_image.astype(float)
        
        shape = target_image.shape
        N_pix = shape[0] * shape[1]
        
        # Load the source image into a numpy array if an image is given, otherwise initialize a
        # uniform source image the same shape as the target
        if source_file is None:
            log.info('No source image was provided, assuming a uniform background...\n')
            source_image = np.ones(shape)
        else:
            log.info(f'Using the source image {source_file}...\n')
            source_image = cv2.imread(source_file, cv2.IMREAD_GRAYSCALE)
            if source_image is None:
                log.exception(f'No source image was found at {source_file}\n')
            else:
                source_image = source_image.astype(float)
        
        try:
            N = init.N_sites
        except AttributeError:
            N = None
        
        if N is None:
            N = int(np.floor(0.8 * N_pix))
            
        try:
            use_previous = init.use_previous_source
        except AttributeError:
            use_previous = False
        
        sites = None
        if use_previous:
            # If we want to load a previous source plane tesselation, we need an .npz file to load
            try:
                previous_source = init.previous_source
                log.info(f'Using the source plane tesselation saved in {previous_source}\n')
            except AttributeError:
                log.exception("To use a previous run's source plane tesselation you must provide a path to the .npz file containing that run's data.\n")
            previous_data = np.load(previous_source)
            source_image = previous_data['source_image']
            sites = previous_data['sites']
            N = sites.shape[0]
            
        # Check that source and target plane images are the same size
        if source_image.shape != shape:
            log.exception(f'Source and target plane images must be the same dimensions, but instead were {source_image.shape} and {shape}!\n')
            
        # Normalize the source and target images:
        source_image *= N_pix / np.sum(source_image)
        target_image *= N_pix / np.sum(target_image)
        
        # Get the threshold for lloyd relaxation     
        try:
            lloyd_thresh = init.lloyd_threshold
        except AttributeError:
            lloyd_thresh = None
        if lloyd_thresh is None:
            lloyd_thresh = 0.1

        try:
            save_txt = init.save_txt
        except AttributeError:
            save_txt = False
        
        # Load the suffix to tag onto any output files
        try:
            suffix = init.output_suffix
        except AttributeError:
            suffix = None

        if suffix is None:
            # If no filename suffix is provided, use the current time and date
            import time
            suffix = time.strftime("%Y%m%d_%T", time.localtime())
            log.info(f'No output file suffix was provided, using the current time and date: {suffix}\n')
        
        # Add a leading underscore to the suffix if it's not already there
        if suffix[0] != '_':
            suffix = '_' + suffix
            
        params = {'target_image': target_image,
                  'source_image': source_image,
                  'N': N,
                  'lloyd_thresh': lloyd_thresh,
                  'sites': sites}
        
        # Fix plotting warnings
        mpl.rcParams['pcolor.shading'] = 'auto'
        log.info(f'Successfully pulled input parameters from {input_fname}.\n')
        return params, target_fname, output_dir, suffix, save_txt
    
    
def save_run(target_fname, output_dir, suffix, save_txt, results):
    '''Save all the data for this run'''
    fname = output_dir + target_fname[:-4] + '_output' + suffix + '.npz'
    log.info(f'Saving run data to {fname}')
    np.savez(fname, **results)
    return

def main():
    mp.set_start_method('spawn')

    params, target_fname, output_dir, suffix, save_txt = initialize()

    target, phi_c, alpha_x, alpha_y, result = ot.get_deflection_potential(**params, output_dir = output_dir)
    
    results = {'target_image':target.image,
               'source_image': params['source_image'],
               'sites': target.sites,
               'weights': target.weights,
               'phi': phi_c,
               'alpha_x':alpha_x,
               'alpha_y':alpha_y}
    
    #ot.plot_displacements(target)
    #plt.savefig(output_dir+'site_displacement'+suffix+'.png', 
    #            dpi=300, 
    #            bbox_inches='tight')
    save_run(target_fname, output_dir, suffix, save_txt, results)
    import time
    end_time = time.strftime("%Y%m%d_%T", time.localtime())
    log.info(f'Analysis finished at {end_time}.')

    return

if __name__ == '__main__':
    main()

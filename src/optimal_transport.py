import numpy as np
import os
import logging as log
from . import voronoi as vor
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import griddata
#from scipy.interpolate import RectBivariateSpline
from scipy.spatial import distance

def initialize_sites(source_image = None, N=100, lloyd_thresh = 0.1,uniform=False):
    '''
    Deploy N random sites on the target image and perform Lloyd relaxation
    '''
    log.info(f'Deploying N = {N} sites on the source plane...')
    if source_image is None:
        # if there is no source plane image, assume a uniform background and normalize to sum to 1
        source_image = np.ones_like(target_image)
        source_image /= np.sum(source_image)
        uniform = True
    
    shape = source_image.shape
    h, w = shape
    Y, X = np.indices(shape).astype(np.float64)
    
    # Randomly deploy sites on the source plane 
    weights = source_image.flatten() / np.sum(source_image)
    # Randomly pick an index
    idx = np.arange(len(weights))
    rand_idx = np.random.choice(idx, N, replace = False, p = weights)

    # Get the coordinates of each sampled point
    sites = np.array([X.flatten()[rand_idx], Y.flatten()[rand_idx]], dtype=float).T
    # add some noise to the positions of initial sites within their pixel, 
    # since pure integer values can mess up the convex hull
    noise = np.random.rand(*sites.shape) - 0.5
    sites += noise
    
    if uniform:
        source = vor.Voronoi(sites, shape, clip = True)
    else:
        source = vor.Voronoi(sites, shape, image = source_image, clip=False)
    log.info('Performing Lloyd relaxation on the source plane...')
    source.lloyd(threshold = lloyd_thresh)
    
    source_fname = 'source_plane.npz'
    if not os.path.isfile(source_fname):
        np.savez('source_plane.npz', 
                 sites=source.sites,
                 source_image=source_image)
    
    return source

def get_deflection_potential(target_image,
                             source_image = None, 
                             N=100,
                             lloyd_thresh = 0.1, 
                             interp_method = 'linear',
                             sites = None,
                             output_dir = None):
   
    shape = source_image.shape
    h, w = shape
    Y, X = np.indices(shape)

    if sites is not None:
        source = vor.Voronoi(sites, shape, image = source_image, clip=False)
    else:
        #Deploy sites on source plane and lloyd relax
        source = initialize_sites(source_image, N, lloyd_thresh)

    sites = source.sites
    # Initial guess of the weights
    if output_dir is not None:
        if os.path.isfile(output_dir + 'restart.npz'):
            log.info(f'Loading restart file at {str(output_dir) + "restart.npz"}')
            restart = np.load(output_dir + 'restart.npz')
            if np.array_equal(restart['sites'], sites, equal_nan=True):
                log.info('Restart file loaded with correct sites, continuing where optimization left off...')
                w0 = restart['weights']
            else:
                log.info('Restart file contained different sites than current run, starting with zero weights...')
                w0 = np.zeros(N, dtype=np.float64)  
        else:
            log.info(f'No restart file found, initializing weights as zero...')
            w0 = np.zeros(N, dtype=np.float64)  
    else:
        log.info(f'No output directory given for restart files, initializing weights as zero...')
        w0 = np.zeros(N, dtype=np.float64)
    
    target = vor.Voronoi(source.sites, shape, image = target_image, weights = w0, clip=False)
    
    # the weight should never be larger than the image itself
    bounds = [(0, None) for site in sites]
    max_iter = max(shape) * 4
    
    log.info('Optimizing cell weights on the target plane (this will take a while)...')
    result = minimize(f, w0, args=(source, target,{'Nfeval':0}, output_dir), 
                            jac = True, 
                            bounds = bounds,
                            method='L-BFGS-B',#L-BFGS-B
                            options={'maxiter':max_iter, 'maxls':30},#'ftol':1e-8, 'gtol':1e-8, 'maxls':30}
                     )
# maxls=30
    if result.success:
        log.info(f'Minimization succeeded after {result.nit} iterations and {result.nfev} power diagrams')
        log.info(result.message)
    else:
        log.warning(f'Minimization failed after {result.nit} iterations and {result.nfev} power diagrams')
        log.info(result.message)
        #return target, np.zeros_like(source_image), np.zeros_like(source_image), np.zeros_like(source_image)
    
    log.info('Calculating the centroids of the optimized cells...')
    target.weights = result.x
    # Remove any invalid areas (corresponding to sites whose Voronoi cells vanished)
    centroids = np.copy(target.c)
    valid = target.A > 0
    valid *= (centroids[:,0] < w)  * (centroids[:,1] < h)
    centroids = centroids[valid]
    sites = target.sites[valid]

    log.info('Moving nearest sites to the corners of the image...')
    sites = move_to_corners(source_image.shape, sites)
    
    log.info('Interpolating the displacements...')
    interp_order = 1
    x_map = griddata(sites,centroids[:,0],(X, Y), method= interp_method )
    y_map = griddata(sites,centroids[:,1],(X, Y), method= interp_method )
    #interp_x = RectBivariateSpline(sites.T[0], sites.T[1],centroids[:,0],
    #                               kx=interp_order, ky=interp_order )    
    #interp_y = RectBivariateSpline(sites.T[0], sites.T[1],centroids[:,1],
    #                               kx=interp_order, ky=interp_order )
    #x_map = interp_x.ev(X.flatten(), Y.flatten()).reshape(X.shape)
    #y_map = interp_y.ev(X.flatten(), Y.flatten()).reshape(X.shape)
    
    alpha_x = x_map - X
    alpha_y = y_map - Y
    
    log.info('Calculating the deflection potential...')
    phi_x = -1 * np.cumsum(alpha_x, axis=1)
    phi_y = -1 * np.cumsum(alpha_y, axis=0)
    
    phi = (phi_x + phi_y) / 2
    
    #Set the minimum of the potential to zero
    phi -= np.min(phi)
    
    log.info('Finished')
    
    return target, phi, alpha_x, alpha_y, result
    
def f(weights, source, target, minInfo, output_dir):
    """ Returns the minimization function and its gradient"""
    S = source.A
    target.weights = weights
    
    T = target.A
    I = target.I
    c = target.c
    p = target.sites
    
    I_0 = I - T * np.sum(c**2, axis=1) + T * np.sum((p - c)**2, axis=1)
    
    f = np.sum(weights * (T - S) - I_0)
    grad = T - S
    nfev = minInfo['Nfeval']
    if nfev%5 == 0:
        n_vanished = np.sum(T==0)
        log.info(f"nfev = {nfev}, f = {f:.2f}, n_vanished = {n_vanished}")
        if output_dir is not None:
            np.savez(output_dir + 'restart.npz', 
                     sites=source.sites,
                     weights=weights)
        
        with np.printoptions(precision=1, suppress=True):
            log.debug(f"A = {T[:5]}")
            log.debug(f"I = {I[0:5]}")
            log.debug(f"c = {c[0:5]}")
            log.debug(f"p = {p[0:5]}")
            log.debug(f"Ix0 = {I_0[0:5]}")
    minInfo['Nfeval'] += 1
    return f, grad

def move_to_corners(shape, sites):
    h, w = shape
    corners = np.array([[0,0], [0,h],[w,h],[w,0]]) - 0.5
    from scipy.spatial import distance
    for corner in corners:
        closest_index = distance.cdist([corner], sites).argmin()

        sites[closest_index] = corner
    return sites
                             
def plot_displacements(target, **kwargs):
    '''
    Plot the Voronoi diagram on the target plane showing the displacement between initial sites and
    the final positions of the centroids.
    '''
    fig, ax = plt.subplots()
    centroids = target.c
    # Remove any invalid centroids (corresponding to sites whose Voronoi cells vanished)
    valid = np.all(centroids != -1, axis=1)
    centroids = target.c[valid]
    sites = target.sites[valid]
    
    target.plot(ax=ax, plot_image=True, sites=False, lw=0.5, **kwargs)
    
    for site, centroid in zip(sites, centroids):
        ax.plot((site[0], centroid[0]),(site[1], centroid[1]), 'r-', lw=.5)
    
    ax.plot(centroids[:,0], centroids[:,1], 'ro', ms=.5)
    ax.set_title('Displaced Centroids on the Target Plane')

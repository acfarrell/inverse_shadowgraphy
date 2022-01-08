import numpy as np
import voronoi as vor
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import griddata
from scipy.spatial import distance

def initialize_sites(source_image = None, N=100, lloyd_thresh = 0.1):
    '''
    Deploy N random sites on the target image and perform Lloyd relaxation
    '''
    print(f'Deploying N = {N} sites on the source plane...')
    shape = source_image.shape
    if source_image is None:
        # if there is no source plane image, assume a uniform background and normalize to sum to 1
        source_image = np.ones_like(target_image)
        source_image /= np.sum(source_image)
    h, w = shape
    Y, X = np.indices(shape)
    
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
    
    source = vor.Voronoi(sites, shape, image = source_image, clip=False)
    print('Performing Lloyd relaxation on the source plane...')
    source.lloyd(threshold = lloyd_thresh)
    return source

def get_deflection_potential(target_image,source_image = None, N=100, lloyd_thresh = 0.1):
    shape = source_image.shape
    Y, X = np.indices(shape)

    
    #Deploy sites on source plane and lloyd relax
    source = initialize_sites(source_image, N, lloyd_thresh)

    sites = source.sites
    # Initial guess of the weights
    w0 = np.zeros(N)
    
    target = vor.Voronoi(source.sites, shape, image = target_image, weights = w0, clip=False)
    
    bounds = [(0,None) for site in sites]
    max_iter = 100
    
    print('Optimizing cell weights on the target plane (this will take a while)...')
    result = minimize(f, w0, args=(source, target), 
                            jac = True, 
                            bounds = bounds,
                            method='L-BFGS-B',
                            options={'maxiter':max_iter}
                     )

    if result.success:
        print(f'Minimization succeeded after {result.nit} iterations and {result.nfev} power diagrams')
    else:
        print(f'Minimization failed after {result.nit} iterations and {result.nfev} power diagrams')
        print(result.message)
        return target, np.zeros_like(source_image)
    
    print('Calculating the centroids of the optimized cells...')
    target.weights = result.x
    centroids = np.copy(target.c)
    # Remove any invalid centroids (corresponding to sites whose Voronoi cells vanished)
    valid = np.all(centroids != -1, axis=1)
    centroids = centroids[valid]
    sites = target.sites[valid]

    print('Moving nearest sites to the corners of the image...')
    centroids = move_to_corners(source_image.shape, centroids)
    
    print('Interpolating the displacements...')
    points = np.stack((centroids, sites))
    disp = np.diff(points, axis=0)[0]

    disp_x = griddata(centroids,disp[:,0],(X, Y), method= 'cubic' )
    disp_y = griddata(centroids,disp[:,1],(X, Y), method= 'cubic' )
    
    print('Calculating the deflection potential...')
    phi_x = np.cumsum(disp_x, axis=1)
    phi_y = np.cumsum(disp_y, axis=0)
    
    phi = (phi_x + phi_y) / 2
    
    #Set the minimum of the potential to zero
    phi -= np.min(phi)
    
    print('Finished')
    
    return target, phi
    
def f(weights, source, target):
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

    return f, grad

def move_to_corners(shape, centroids):
    h, w = shape
    corners = np.array([[0,0], [0,h],[w,h],[w,0]]) - 0.5
    from scipy.spatial import distance
    for corner in corners:
        closest_index = distance.cdist([corner], centroids).argmin()

        centroids[closest_index] = corner
    return centroids
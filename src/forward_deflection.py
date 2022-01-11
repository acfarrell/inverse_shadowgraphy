'''
Functions for taking a given source image and deflection potential and projecting forward to determine the resulting target image.
'''
import numpy as np
from multiprocessing import Pool
from . import shapes
from . import rasterize

def forward_map(source, phi_func):
    '''
    Given a source image (represented as a 2D numpy array) and a deflection potential function (of the 
    form phi(x, y)) project the source plane forward a distance L to determine the image at the 
    target plane.t
    '''
    h, w = source.shape
    # Get the indices of each pixel in the image
    Y, X = np.indices((h, w))
    idxs = [(i,j) for i, j in zip(X.flatten(), Y.flatten())]

    # Get the coordinates of each corner of the pixels in the image
    pix_y, pix_x = np.indices((h+1, w+1), dtype=float) - 0.5

    # Evaluate the deflection potential at each pixel corner
    phi = phi_func(pix_x, pix_y)
    # Determine the deflection amount in each direction from the gradient of phi
    ay, ax = np.gradient(phi)
    
    # Calculate the final positions of each pixel corner after deflection
    x = pix_x - ax 
    y = pix_y - ay 
    points = np.stack((x, y)).T # [w, h, axis]
    
    # Get the contribution to the target image from each pixel in the source image
    args = [(i,j, points, source) for i, j in zip(X.flatten(), Y.flatten())]
    target_maps = Pool().map(forward_map_pixel_wrapper, args)
    
    target_image = sum(target_maps)
    return target_image

def forward_map_pixel_wrapper(args):
    return forward_map_pixel(*args)

def forward_map_pixel(i, j, points, source_image):
    '''
    Get the four corners of the pixel at j, i after being deflected from the source to image plane
    and from those vertices determine that pixels contributions to the target plane intensity profile
    '''
    
    # Get the coordinates of the four vertices of the pixel
    p1 = points[i, j]
    p2 = points[i+1, j]
    p3 = points[i, j+1]
    p4 = points[i+1, j+1]

    verts = np.array([p1, p2, p3, p4])
    # Calculate the mean coordinate to estimate the center of the polygon (for clockwise sorting)
    center = np.mean(verts, axis=0)
    
    # sort the vertices in clockwise order
    verts = shapes.clockwise_sort(verts)
    pixel = shapes.Polygon(verts)
    I_0 = source_image[j, i] / pixel.A
    A_map, _, _ = rasterize.rasterize(pixel, source_image.shape)
    return I_0 * A_map

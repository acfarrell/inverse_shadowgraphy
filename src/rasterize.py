'''
Functions for rasterizing polygons onto a pixelated grid and determining the portion of
each pixel inside and outside a given polygon. Clipping individual pixels like this is
important for cases where the polygons and pixels are of similar volume, which is the 
case in invert shadowgraphy calculations.
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from . import shapes

def rasterize(polygon, shape, plot = False, **kwargs):
    ''' Pixelate the given polygon to an integer grid of given shape '''
    Y, X = np.indices(shape)

    area_map = np.zeros(shape)
    centroid_map = np.stack((X, Y)).astype(float)
    I_map = np.sum(centroid_map**2, axis = 0) + 1/6.
    #print(f'c_map = {centroid_map[:, 5,5]}, I_map = {I_map[5,5]}')
    enclosed = get_enclosed_pixels(polygon, shape)
    area_map[enclosed] = 1
    centroid_map[:,~enclosed] = -1
    I_map[~enclosed] = 0
    
    edges = get_edge_pixels(polygon, shape)
    
    edge_pixels = np.stack((X[edges], Y[edges])).T
    if plot:
        ax = plt.gca()
    for pixel in edge_pixels:
        idx = (pixel[1], pixel[0])
        pix = shapes.Pixel(pixel)
        pix.clip_to(polygon)
        if pix.N == 0:
            continue
        area_map[idx] = pix.A
        centroid_map[:, pixel[1], pixel[0]] = pix.c[0], pix.c[1]
        I_map[idx] = pix.I
        
        if plot:
            pix.plot(**kwargs)
    return area_map, centroid_map, I_map

def get_edge_pixels(polygon, shape):
    ''' Find all the pixels that are intersected by the edges of the polygon '''
    h, w = shape

    # Make a boolean array that tells whether each pixel is on the edge or not
    pixels = np.zeros(shape).astype(bool)
    if polygon.N == 0:
        return pixels
    
    # get the endpoints of each edge of the polygon
    edges = np.stack((polygon.vertices, np.roll(polygon.vertices, -1, axis=0)), axis =1)

    
    for edge in edges:
        p1, p2 = edge[0], edge[1]
        if (abs(p1[0] - p2[0]) < 1e-3):
            y1, y2 = min(p1[1], p2[1]), max(p1[1], p2[1])
            if y1 > h - 1 or y2 < 0:
                continue
            # Take care of vertical lines first
            y_int = np.arange(max(0,y1), min(h-1, y2))
            x_int = np.array([int(p1[0])] * y_int.size)
            valid = (x_int <= w-1) & (x_int >= 0)
        else:
            # Interpolate the line between two vertices and cast back onto an integer grid
            m = (p1[1] - p2[1]) / (p1[0] - p2[0]) # get slope of line
            
            x1, x2 = min(p1[0], p2[0]), max(p1[0], p2[0])
            if x1 > w - 1 or x2 < 0:
                continue
            x_int = np.linspace(max(0, x1), min(w-1,x2), 2*max(h, w))
            y_int = (m * (x_int - p1[0]) + p1[1]) # get y values along line
            valid = ((y_int <= h-1) & (y_int >= 0))
        pix_edge = np.around(np.stack((x_int[valid], y_int[valid]))).astype(int) # cast to integers
        pixels[pix_edge[1], pix_edge[0]] = True # make a boolean mask of edges
    return pixels

def get_enclosed_pixels(polygon, shape):
    if polygon.N == 0:
        return np.zeros(shape, dtype=bool)
    
    Y, X = np.indices(shape).astype(float)
    points = np.stack((X.flatten(), Y.flatten())).T
    
    poly_path = Path(polygon.vertices)
    enclosed = poly_path.contains_points(points)
    return enclosed.reshape(shape)

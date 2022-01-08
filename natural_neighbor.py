import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import voronoi
import shapes
import multiprocessing as mp

def interpolate(sites, values, shape):
    '''
    Perform 2D natural neighbor interpolation of scattered data with a given value at each site
    onto a pixel grid of the given shape.
    '''
    h, w = shape
    
    # First get the Voronoi tesellation of the given sites
    vor = voronoi.Voronoi(sites, shape)
    
    # Grab the vertices of all the Delaunay triangles from the Voronoi object
    indices = vor.triangles # indices of the sites defining each triangle
    triangles = sites[indices]# (x,y) coordinates of the vertices of each triangle
    centers = vor.circumcenters
    radii = vor.radii   
    
    # Define the pixel grid to interpolate on
    Y, X = np.indices(shape)
    
    # Flatten grid into a list of coordinates
    interp_points = np.stack((X.flatten(), Y.flatten())).T
    # Initialize an array for the interpolated values
    interp_values = np.zeros(h*w)

    in_circle = []

    for i, triangle in enumerate(triangles):
        # Grab the circumcircle
        c = centers[i]
        r = radii[i]
        circle = mpl.path.Path.circle(center = c, radius = r)

        # Get a boolean map of all the pixels enclosed by the circumcircle
        enclosed = circle.contains_points(interp_points, radius= -.01)#.reshape(shape)
        in_circle.append(enclosed)

    in_circle = np.array(in_circle).T # Now we have an array of format [point_idx, in_circle]
    # Find all the unique combinations of triangles that are natural neighbors to the same points
    unique_envelopes, npix = np.unique(in_circle, axis = 0, return_counts = True)
    idxs = np.arange(triangles.shape[0])

    for i, envelope in enumerate(unique_envelopes):
        if np.sum(envelope) == 0:
            continue
        # find all the pixels corresponding to this envelope
        pixels = np.where((in_circle == envelope).all(1))[0]

        # get the indices of the triangles in the envelope
        envelope_triangles = idxs[envelope]

        # get the indices of the sites defining each triangle
        triangle_vertices = np.array(indices[envelope_triangles])

        # Define the Bowyer-Watson envelope
        envelope_vertices = np.unique(triangle_vertices)
        envelope_values = values[envelope_vertices]
        n_verts = envelope_vertices.shape[0]

        # get the coordinates of each vertex in the envelope
        vert_coords = shapes.clockwise_sort(sites[envelope_vertices])

        # get the midpoint of each edge
        midpoints = (vert_coords + np.roll(vert_coords, -1, axis=0)) / 2

        A_0 = np.zeros_like(envelope_vertices)
        # calculate the areas of the original voronoi cells inside the envelope
        for i, site in enumerate(envelope_vertices):
            # find the triangles that have the site as a vertex        
            tri_idx = np.argwhere(triangle_vertices == site)[:,0]
            # get the midpoints and circumcenters of the site's triangles
            c = centers[envelope_triangles[tri_idx]]
            m = [midpoints[i], midpoints[i-1]]
            verts = shapes.clockwise_sort(np.concatenate((m, c)))
            A_0[i] = shapes.area(verts)

        for pixel in pixels:
            point = interp_points[pixel]

            A_1 = np.zeros_like(envelope_vertices)        

            new_triangles = np.stack((vert_coords, np.roll(vert_coords, -1,axis=0), [point] * n_verts))
            new_triangles = new_triangles.transpose((1,0,2))

            new_centers = [shapes.circumcenter(verts) for verts in new_triangles]
            new_cells = np.stack((midpoints, 
                                   new_centers, 
                                   np.roll(new_centers, 1, axis=0), 
                                   np.roll(midpoints, 1, axis=0))).transpose((1,0,2))
            A_1 = [shapes.area(verts) for verts in new_cells]
            weights = A_0 - A_1
            lambdas = weights / np.sum(weights)
            value = np.sum(lambdas * envelope_values)
            interp_values[pixel] = value

    return interp_values.reshape((h,w))

if __name__ == "__main__":
    h, w = 100,100
    N = 500
    shape = (h, w)

    Y, X = np.indices(shape, dtype=float)
    # Give each site a value that we want to interpolate
    # In our real use case these values would be the displacement of each centroid in x or y
    sigma = .25*h
    gauss = lambda x, y: np.exp(-((x - w//2)**2 + (y - h//2)**2)/sigma**2)

    source_image = 10 * gauss(X,Y)

    # Use the image intensities as the weights for each pixel
    weights = np.sqrt(source_image).flatten() / np.sum(np.sqrt(source_image))
    # Randomly pick an index
    idx = np.arange(len(weights))

    rand_idx = np.random.choice(idx, N, replace = False, p = weights)

    # Get the coordinates of each sampled point
    points = np.array([X.flatten()[rand_idx], Y.flatten()[rand_idx]], dtype=float).T
    noise = np.random.rand(*points.shape) - 0.5
    points += noise
    values = source_image.flatten()[rand_idx]

    vor = voronoi.Voronoi(points, shape)
    vor.lloyd(threshold=.1)

    # Move some sites to the corners
    pts = vor.sites

    pts[0] = [0,0] 
    pts[1] = [h,w]
    pts[2] = [h,0]
    pts[3] = [0,w]
    pts[:4] -= 0.5

    values[0] = source_image[0,0] 
    values[1] = source_image[-1,-1]
    values[2] = source_image[-1,0]
    values[3] = source_image[0,-1]

    interp_values = interpolate(pts, values, shape)

    fig, (ax1, ax2) = plt.subplots(1,2)

    ax1.set_aspect('equal')
    ax2.set_aspect('equal')

    ax1.set_title('Original Image')
    ax1.pcolormesh(X,Y,source_image, cmap='gray', vmin=0, vmax=10, shading='auto')

    ax2.set_title(f'Natural Neighbor Interpolation with N = {N} Sites')
    ax2.pcolormesh(X,Y, interp_values, cmap='gray', vmin=0, vmax=10, shading='auto')
    #ax2.plot(pts[:,0], pts[:,1], 'ro', ms=1)

    plt.show()
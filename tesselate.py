import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from numba import jit, jit_module

from multiprocessing import Process, Queue

class Voronoi:
    """
    Class for calculating 2D power diagrams, i.e. weighted Voronoi tesselations
    """
    def __init__(self, sites, shape, weights=None, image = None):
        self._sites = sites
        self.N = sites.shape[0]
        self.shape = shape
        h, w = shape
        # Define the corner points of the box in clockwise order
        self.box = np.array([[0,0],[0,h-1],[w-1,h-1],[w-1,0]], dtype=float)
        # Define each edge of the box
        self.box_edges = np.stack((self.box, np.roll(self.box, -1, axis=0)), axis =1)

        if weights is None:
            self._weights = np.zeros(self.N)
        else:
            self._weights = weights
            
        if image is None:
            self.image = np.ones(shape)
        else:
            self.image = image
        
        # Calculate the weighted Voronoi diagram
        self.regions = self.get_Voronoi()
        self.A, self.c, self.I = weighted_stats(self)
    
    @property
    def sites(self):
        """The sites (generators) of the power diagram."""
        return self._sites
    
    @sites.setter
    def sites(self, points, point_weights = None):
        """Setter function for the sites of the power diagram."""
        if points.size != self._sites.size:
            #raise ValueError('The number of sites for this power diagram has changed!')
            print('The number of sites for this power diagram has changed!')
            self.N = points.shape[0]
        self._sites = points
        if point_weights is not None:
            self._weights = point_weights
        
        # if the sites change, we need to recalculate the power diagram.
        self.regions = self.get_Voronoi()
        self.A, self.c, self.I = weighted_stats(self)

    
    @property
    def weights(self):
        """The weights corresponding to each site in the power diagram."""
        return self._weights
    
    @weights.setter
    def weights(self, values):
        """Setter function for the weights of each sites."""
        self._weights = values
        
        # if the weights change, we need to recalculate the power diagram.
        self.regions = self.get_Voronoi()
        self.A, self.c, self.I = weighted_stats(self)
    
    def get_Voronoi(self):
        """ Calculate the Voronoi tesselation using the projection onto a 3D paraboloid"""
        # Project the 2D sites onto a 3D weighted paraboloid
        points_3D = self.lift_points(self.sites, self.weights)
        # Calculate the convex hull in 3D
        self.hull = ConvexHull(points_3D)
        
        # Calculate the dual vertex of each face in the convex hull
        # Suppress divide by zero errors for this bit since there will be some infs
        old_settings = np.seterr(invalid = 'ignore', divide = 'ignore')
        # Calculate all the vertices from the hull equations
        all_vertices = -0.5 * self.hull.equations[:,:2] / self.hull.equations[:,2][:, None]
        # Revert to the old error settings
        np.seterr(**old_settings)
        
        # Determine which faces of the hull are downward-facing, 
        # Downward-facing faces of the hull give finite Delaunay triangles
        self.lowers = self.get_lower_facing()
        
        # Determine the faces defining each edge of the Delaunay triangulation from the convex hull
        neighbors, infinite = self.get_delaunay_edges()
        
        triangles = self.hull.simplices[neighbors] # get the indices of the vertices of each Delaunay triangle
        # Find the common edge between each pair of neighboring Delaunay triangles
        edge_sites = np.array([np.intersect1d(pair[0], pair[1]) for pair in triangles])
        self.delaunay_edges = edge_sites

        # Get the dual vertices corresponding to each Delaunay edge
        vertices = all_vertices[neighbors]

        # Overwrite the infinite vertices to finite points along the edges
        vertices[infinite] = [self.get_infinite_vertices(faces, sites) for \
                              faces, sites in zip(neighbors[infinite], edge_sites[infinite])]
        self.vertices = vertices
        # Get all the edges containing each site
        cell_vertices = []
        for i, site in enumerate(self.sites):
            # Find the indices of the Delaunay edges containing each site
            edge_idx = np.argwhere(edge_sites == i)[:,0]
            
            # Get the Voronoi vertices from each of those edges
            verts = vertices[edge_idx]
            # Remove duplicate vertices
            verts = np.unique(np.reshape(verts, (-1,2)), axis = 0)

            # Sort the vertices in clockwise order
            verts = np.array(sorted(verts,
                                    key = lambda v: np.arctan2((v[0] - site[0]), (v[1] - site[1]))))
            
            # Clip the vertices of the cell to the bounding box
            verts = self.clip(verts)
            cell_vertices.append(verts)
        
        return cell_vertices
        
    def get_delaunay_edges(self):
        """ 
        Find all the edges in the Delaunay triangulation by grabbing all the pairs of faces
        in the convex hull that share an edge.
        If one of the faces in a pair is not downward-facing, the edge is infinite.
        Returns:
            neighbors:  array of pairs of hull face indices that are unique neighbors 
                        defining a delaunay edge
            infinite:   boolean mask giving all the edges that are infinite
        """
        hull, lowers = self.hull, self.lowers
        # the neighbors attribute contains the indices of the 3 faces that are the neighbors
        # of the face corresponding to each position in the array
        neighbors = np.zeros((lowers.shape[0], 2 * hull.ndim), dtype = int)

        # Each finite face has three neighbors
        neighbors[:, 0::2] = lowers[:, None] # set every other element to the finite face index
        neighbors[:, 1::2] = hull.neighbors[lowers] # set the remaining elements to its neighbors' indices

        # Reshape array to be a list of neighboring faces defining each Delaunay triangle edge
        neighbors = np.sort(neighbors.reshape([-1,2]), axis = 1)
        # Remove any duplicate pairs
        neighbors = np.unique(neighbors, axis = 0)

        # Make a mask of edges that are infinite (i.e. connect faces that are *not* both downward facing)
        infinite = np.isin(neighbors, lowers).sum(axis=1) != 2

        return neighbors, infinite
    
    def get_infinite_vertices(self, faces, sites):
        """
        Calculate the dual vertices  of a Delaunay edge in the case where one of the 
        faces defining the edge is not downward-facing (has an infinite dual vertex)
        """
        hull, lowers = self.hull, self.lowers
        h, w = self.shape
        # Determine which face defining the given edge is the nasty one
        is_lower = np.isin(faces, lowers)
        f1, f2 = faces[is_lower], faces[~is_lower] # f1 is finite, f2 is infinite

        # Check that only one of the faces is infinite
        if np.sum(is_lower) != 1:
            print("An edge is only infinite if one of the two faces is infinite.")
            return

        # Get the two vertices the faces have in common (the Delaunay edge)
        p1, p2 = hull.points[sites[0]], hull.points[sites[1]] # edge vertices in form [x y z]
        edge = np.stack((p1[:2],p2[:2]))

        # get the one point in the triangle f1 that *isn't* on the shared edge
        triangle = hull.simplices[f1]
        p3 = hull.points[triangle[~np.isin(triangle, sites)]][0, :2]

        # the dual vertex of the lower-facing plane is calculated normally
        v1 = (-0.5 * hull.equations[f1, :2] / hull.equations[f1, 2])[0]

        # Calculate another point on the line to define the infinite edge
        # Calculate the slope of the line perpendicular to the line connecting p1 and p2
        if (p2 -p1)[1] == 0:
            # line is purely vertical, slope undefined 
            delta = np.array([0,h])
        else:
            m = -(p2 - p1)[0] / (p2 - p1)[1]
            # Define the second vertex at the width of the image away from v1 in x, so v2 is guaranteed
            # to be outside the boundary of our box
            dx = w
            dy = m * dx
            # Calculate the offset of v2 relative to v1
            delta = np.array([dx, dy])
        
        # Make sure we're drafting the edge in the right direction (outward)
        if np.dot(delta, np.mean(edge, axis=0) - p3) < 0:
            # if delta is pointing towards p3, flip it around
            delta *= -1
        
        v2 = v1 + delta

        return np.stack((v1, v2))

    def mirror_points(self):
        """
        DEPRECIATED
        Reflect all the sites over the edges of the bounding box
        """
        h, w = self.shape
        pts = np.copy(self.sites)
        
        pts = np.tile(pts, (5,1,1))
        pts[[1,2],:,1] *= -1
        pts[[3,4],:,0] *= -1

        pts[2,:,1] += 2*h
        pts[4,:,0] += 2*w

        return pts.reshape((5 * self.N ,2))
                    
    def lift_points(self, points, weights):
        """
        Lift 2D points onto 3D paraboloid for calculating weighted 
        Delaunay triangulations and Voronoi diagrams

        Arguments:
            points: 2D numpy array of point coordinates with shape (N, 2)
            weights: 1D numpy array of length N holding each points corresponding weight

        Returns:
            3D numpy array of coordinates with shape (N, 3) where the third column is 
            the z projection
        """
        z = np.sum(points * points, axis=1) - weights 
        return np.hstack((points, z[np.newaxis,:].T))
        
    def get_lower_facing(self):
        """
        Return the indices of simplexes in hull that are lower-facing
        """
        return np.where(self.hull.equations[:,2] < 0)[0]
    
    def circumcenter(self, points):
        """
        Calculates the circumcenter of the triangle defined by points
        Arguments:
            points: 2D ndarray of triangle vertices with form [[Ax, Ay],[Bx, By],[Cx, Cy]]

        Returns:
            circumcenter: [cx, cy] coordinates of the triangle's circumcenter
        """

        # Get the permutation needed for Ax(By - Cy), etc
        shifted = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
        flipped = np.fliplr(shifted)
        flipped[:,1] *= -1

        # Calculate the magnitude squared of each vertex vector
        mag = np.sum(points**2, axis=1)

        # Calculate the circumcenter
        D = 2 * np.sum(points[:,0] * flipped[:,0])
        return (mag @ flipped) / D
    
    def clip(self, shape):
        """
        Clip a polygon defined by its vertices to the bounding box of the diagram
        Arguments:
            shape:  numpy array of the vertices of the polygon to be clipped,
                    given in clockwise order
        """
        if shape.size == 0:
            # There is no shape, exit,
            return shape
        clipped = shape.copy()
        if shape[0].size == 1:
            # there's only one edge, don't close the polygon
            shape_edges = [shape]
        else:
            shape_edges = np.stack((shape, np.roll(shape, -1, axis=0)), axis =1)
        
        for box_edge in self.box_edges:
            edge_clipped = []
            for i, v1 in enumerate(clipped):
                v2 = clipped[i-1]
                is_inside = [inside(v1, box_edge), inside(v2, box_edge)]

                if np.all(is_inside):
                    # Both vertices are inside the box, add the second vertex to final shape
                    edge_clipped.append(v2)
                    continue
                elif is_inside[0]:
                    # Only the first vertex is inside, add the intersect to final shape
                    edge_clipped.append(get_intersect(np.array([v1, v2]), box_edge))
                    continue
                elif is_inside[1]:
                    # Only the second vertex is inside, add both the second vertex and
                    # the intersect to the final shape
                    edge_clipped.append(v2)
                    edge_clipped.append(get_intersect(np.array([v1, v2]), box_edge))
            edge_clipped = np.array(edge_clipped)
            # update the polygon after clipping to each edge
            clipped = edge_clipped
        return clipped
    def get_centroids(self):
        """Calculates the centroids of all the Voronoi regions using
        multiprocessing to speed things along."""
        centroids = np.zeros(self.sites.shape)
        processes = []
        for i, region in enumerate(self.regions):
            p = Process(target = self.centroid, args = (i,centroids,))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        print(centroids[:5])
        return centroids
            
    def centroid(self, i):
        """ Calculates the centroid of the ith Voronoi region"""
        vertices = self.regions[i]
        
        # if the site has no vertices, return the site's position
        if vertices.size == 0:
            return self.sites[i]
        v_i = vertices
        v_ip1 = np.roll(vertices, 1, axis=0)

        xy = v_ip1 * np.fliplr(v_i)
        xy = np.array([xy[:,1] - xy[:,0]])

        c = ((v_i + v_ip1).T @ xy.T).T[0]
        
        return c / (6 * area(vertices))
        
    def lloyd(self, threshold = 0.05, MAXDEPTH=100, verbose = True):
        """
        Performs lloyd relaxation on a voronoi diagram object
        """
        shape = self.shape
        weights = self.weights
        centroids = self.c#np.array([self.centroid(i) for i in range(self.N)])
        for i in range(MAXDEPTH):
            old_centroids = np.copy(centroids)
            self.sites = centroids
            centroids = self.c#np.array([self.centroid(i) for i in range(self.N)])
            convergence = abs(centroids - old_centroids)/ old_centroids
            if verbose:
                print(f'Centroids shifted by up to {np.max(convergence)*100:.2f}% after {i+1} iterations', end='\r')
            if np.all(convergence < threshold):
                print(f"All centroids within {threshold * 100}% after {i+1} iterations of Lloyd relaxation")
                break
        return
    
    def plot(self, ax = None, transparent = False, plot_weights = False, plot_delaunay = False):
        """
        Plot the power diagram
        """
        h, w = self.shape

        if ax is None:
            ax = plt.gca()

        ax.set_ylim((0,h-1))
        ax.set_xlim((0,w-1))
        
        colors = mpl.cm.get_cmap(name='rainbow')(np.linspace(0,1,self.N))
        
        if transparent:
            alph = 0
            linecol = 'cyan'
            sitecol = 'red'
        else:
            alph = 0.5
            linecol = 'black'
            sitecol = 'black'
        
        if plot_delaunay:
            for idx in self.delaunay_edges:
                pts = self.sites[idx]
                ax.plot(pts[:,0], pts[:,1], 'b', lw = 0.5)
        for i, site in enumerate(self.sites):
            vertices = self.regions[i]
            if vertices.size == 0:
                print(f"Site at i = {i} has no corresponding region. Site is shown in red on the plot.")
                ax.plot(site[0], site[1], 'ro')
                if plot_weights:
                    ax.add_patch(plt.Circle((site[0], site[1]), 
                                 np.sqrt(self.weights[i]), 
                                 color='red', fill=False, lw=0.5, ls='--'))
                continue

            ax.fill(vertices[:,0], vertices[:,1], 
                    facecolor=mpl.colors.to_rgba(colors[i],alph),
                    edgecolor=linecol)

            ax.plot(site[0], site[1], 'o', c = sitecol)
            if plot_weights:
                ax.add_patch(plt.Circle((site[0], site[1]), 
                                        np.sqrt(self.weights[i]), 
                                        color='black', fill=False, lw=0.5, ls='--'))
        ax.set_title('Power Diagram')
        ax.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)
        # Fix aspect ratio so pixels are always square
        ax.set_aspect('equal')
        
class TessellatedImage(Voronoi):
    """
    Subclass for applying weighted voronoi tesselations onto an image 
    using the pixel intensities as weights which update with the locations of the sites
    """
    def __init__(self, sites, image):
        self.image = image
        # Define the weights as the intensity of the image in the pixel that each site sits on
        idxs = sites.astype(int)
        weights = self.image[idxs[:,1], idxs[:,0]]
        super().__init__(sites, self.image.shape, weights)
        
    @Voronoi.sites.setter
    def sites(self, points):
        """
        Setter function for the sites of the power diagram, updating the weights 
        with the image intensities.
        """
        if points.size != self._sites.size:
            #raise ValueError('The number of sites for this power diagram has changed!')
            print('The number of sites for this power diagram has changed!')
            self.N = points.shape[0]
        self._sites = points
        idxs = points.astype(int)
        self._weights = self.image[idxs[:,1], idxs[:,0]]
        
        # if the sites change, we need to recalculate the power diagram.
        self.regions = self.get_Voronoi()
    

        

def area(vertices):
    """
    Calculate the area of a polygon from its vertices
    """
    v_i = vertices
    v_ip1 = np.roll(vertices, 1, axis=0)
    
    xy = v_ip1 * np.fliplr(v_i)
    xy = np.array([xy[:,1] - xy[:,0]])
    return np.sum(xy)/2.
        
def lloyd(vor, MAXDEPTH=100):
    """
    DEPRECIATED
    lloyd is now a class function to fix  issue where some sites with small weights get swallowed up 
    by larger voronoi cells, don't use this one
    Performs lloyd relaxation on a voronoi diagram object
    """
    shape = vor.shape
    weights = vor.weights
    centroids = np.array([centroid(region) for region in vor.regions])
    for i in range(MAXDEPTH):
        old_centroids = np.copy(centroids)
        vor.sites = centroids
        #vor = Voronoi(centroids, shape, weights)
        centroids = np.array([centroid(region) for region in vor.regions])
        if np.array_equal(centroids,old_centroids):
            print(f"Centroids no longer changing at i={i+1} iterations")
            break
    return vor

def inside(point, line):
    """
    Determine if a point is on the right side of a line
    
    When going clockwise, being on the right side of an edge means 
    the point is inside the box
    Arguments:
        point:  numpy array in format [x, y]
        line:   numpy array in format [[x1,y1],[x2,y2]]
                where the line starts at (x1,y1) and ends at (x2,y2)
    Returns:
        True if point is on or to the right of line, else False
    """
    P = np.flip(np.diff(line, axis=0)[0]) # gives [(y2-y1),(x2-x1)]
    P *= point - line[0] # gives [(y2-y1)(x - x1), (x2-x1)(y-y1)]
    P = np.diff(P)[0]
    return P <= 0

def get_intersect(line1, line2):
    """
    Determine the coordinates of the intersection between two lines
    by projecting into 3D space and taking the cross product
    Arguments:
        line1, line2:   numpy arrays in format [[x1,y1],[x2,y2]]
    Returns:
        P:  intersection point in format [x,y]
    """
    # get the equation of each line from the cross product of two points
    # line t = p1 x p2 must pass through both p1 and p2
    # since t . p1 = (p1 x p2) . p1 = 0
    points = np.vstack((line1, line2))
    
    # Add a third homogeneous coordinate
    points_3D = np.hstack((points, np.ones((4,1))))
    
    # Get vectors representing each line
    l1 = np.cross(points_3D[0], points_3D[1])
    l2 = np.cross(points_3D[2], points_3D[3])
    
    # Get the intersection of the two lines
    x,y,z = np.cross(l1, l2)
    
    if z == 0:
        return np.array([float('inf'), float('inf')])
    return np.array([x / z, y / z])

def clip(boundary, polygon):
    """
    Clip a polygon defined by its vertices to the given bounding box
    Arguments:
        boundary:   numpy array of the vertices of the bounding polygon to clip to,
                    given in clockwise order
        polygon:    numpy array of the vertices of the polygon to be clipped,
                    given in clockwise order
        
    """
    clipped = polygon.copy()
    
    box_edges = np.stack((boundary, np.roll(boundary, -1, axis=0)), axis =1)

    for box_edge in box_edges:
        edge_clipped = []
        for i, v1 in enumerate(clipped):
            v2 = clipped[i-1]
            is_inside = [inside(v1, box_edge), inside(v2, box_edge)]
            
            # if both vertices are outside, move along
            if np.all(is_inside):
                # Both vertices are inside the box, add the second vertex to final shape
                edge_clipped.append(v2)
                continue
            elif is_inside[0]:
                # Only the first vertex is inside, add the intersect to final shape
                edge_clipped.append(get_intersect(np.array([v1, v2]), box_edge))
                continue
            elif is_inside[1]:
                # Only the second vertex is inside, add both the second vertex and
                # the intersect to the final shape
                edge_clipped.append(v2)
                edge_clipped.append(get_intersect(np.array([v1, v2]), box_edge))
        edge_clipped = np.array(edge_clipped)
        # update the polygon after clipping to each edge
        clipped = edge_clipped
    return clipped

def clip_with_line(point, line):
    x0, y0 = point[0], point[1]
    t,b,l,r = y0 + .5, y0 - .5, x0 - .5, x0 + .5
    pixel = np.array([[l, b], [l, t], [r, t], [r, b]])
    
    clipped = pixel.copy()
    edge_clipped = []
    for i, v1 in enumerate(clipped):
        v2 = clipped[i-1]
        is_inside = [inside(v1, line), inside(v2, line)]

        # if both vertices are outside, move along
        if np.all(is_inside):
            # Both vertices are inside the box, add the second vertex to final shape
            edge_clipped.append(v2)
            continue
        elif is_inside[0]:
            # Only the first vertex is inside, add the intersect to final shape
            edge_clipped.append(get_intersect(np.array([v1, v2]), line))
            continue
        elif is_inside[1]:
            # Only the second vertex is inside, add both the second vertex and
            # the intersect to the final shape
            edge_clipped.append(v2)
            edge_clipped.append(get_intersect(np.array([v1, v2]), line))
    # update the polygon after clipping to each edge
    return np.unique(np.array(edge_clipped), axis=0)

def get_centroids(sites, regions):
    """Calculates the centroids of all the given Voronoi regions using
    multiprocessing to speed things along. Placed outside of the class because multiprocessing
    has some odd behaviors inside the class structure"""
    centroids = np.zeros(sites.shape)
    Q = Queue()
    def centroid(i, site, vertices, returnarray):
        """ Calculates the centroid of the ith Voronoi region"""

        # if the site has no vertices, return the site's position
        if vertices.size == 0:
            return sites
        v_i = vertices
        v_ip1 = np.roll(vertices, -1, axis=0)

        xy = v_ip1 * np.fliplr(v_i)
        xy = np.array([xy[:,1] - xy[:,0]])

        c = ((v_i + v_ip1).T @ xy.T).T[0]

        Q.put((i, c / (6 * area(vertices))))
    processes = []
    for i, region in enumerate(regions):
        p = Process(target = centroid, args = (i,sites[i],region,centroids,))
        processes.append(p)
        p.start()
        #print(Q.get())
        j, c = Q.get()
        centroids[j] = c
    for p in processes:
        p.join()
    return centroids

def pixel_centroid(vertices, A):
    """ Calculates the centroid of the polygon with given vertices"""

    v_i = vertices
    v_ip1 = np.roll(vertices, 1, axis=0)

    xy = v_ip1 * np.fliplr(v_i)
    xy = np.array([xy[:,1] - xy[:,0]])

    c = ((v_i + v_ip1).T @ xy.T).T[0]

    return c / (6 * A)

from numba import jit, njit
import numba

@jit(nopython=True)
def point_inside(polygon, point):
    length = len(polygon)-1
    dy2 = point[1] - polygon[0][1]
    intersections = 0
    ii = 0
    jj = 1

    while ii<length:
        dy  = dy2
        dy2 = point[1] - polygon[jj][1]

        # consider only lines which are not completely above/bellow/right from the point
        if dy*dy2 <= 0.0 and (point[0] >= polygon[ii][0] or point[0] >= polygon[jj][0]):

            # non-horizontal line
            if dy<0 or dy2<0:
                F = dy*(polygon[jj][0] - polygon[ii][0])/(dy-dy2) + polygon[ii][0]

                if point[0] > F: # if line is left from the point - the ray moving towards left, will intersect it
                    intersections += 1
                elif point[0] == F: # point on line
                    return 2

            # point on upper peak (dy2=dx2=0) or horizontal line (dy=dy2=0 and dx*dx2<=0)
            elif dy2==0 and (point[0]==polygon[jj][0] or (dy==0 and (point[0]-polygon[ii][0])*(point[0]-polygon[jj][0])<=0)):
                return 2

        ii = jj
        jj += 1

    #print 'intersections =', intersections
    return intersections & 1  


@njit(parallel=True)
def points_inside(points, polygon):
    ln = len(points)
    D = np.empty(ln, dtype=numba.boolean) 
    for i in numba.prange(ln):
        D[i] = point_inside(polygon,points[i])
    return D

def get_pixel_areas(vertices, h, w):
    '''
    Makes an array of the area of each pixel, where the area of pixels that are intersected
    by edges of the voronoi diagram are given as the larger of two areas on either side of the
    edge.
    '''
    areas = np.ones((w, h))
    cx, cy = np.indices((w, h)).astype(float)
    big_centroids = np.stack((cx, cy), axis=0)
    lil_centroids = np.copy(big_centroids)
    big_I = np.sum(big_centroids**2, axis = 0) + 1/6
    lil_I = np.copy(big_I)
    for i, edge in enumerate(vertices):
        #print(f'{i = },{edge =}')
        
        p1, p2 = edge[0], edge[1]
        if (abs(p1[0] - p2[0]) < 1e-6):
            # Take care of vertical lines first
            y_int = np.arange(min(edge[:,1]), max(edge[:,1]))
            x_int = np.array([int(p1[0])] * y_int.size)
        else:
            # Interpolate the line between two vertices and cast back onto an integer grid
            m = (p1[1] - p2[1]) / (p1[0] - p2[0]) # get slope of line

            x_int = np.linspace(p1[0], p2[0], w//2)
            y_int = (m * (x_int - p1[0]) + p1[1]) # get y values along line
        pix_edge = np.around(np.stack((x_int, y_int)).T).astype(int) # cast to integers

        pix_edge = pix_edge[(pix_edge[:,0] < w)&(pix_edge[:,1] < h)&(pix_edge[:,1] >= 0)&(pix_edge[:,0] >= 0)]
        if pix_edge.shape[0] == 0:
            continue
        for pixel in np.unique(pix_edge,axis=0):
            if (abs(areas[pixel[0],pixel[1]] - 1) > 1e-6):
                continue
            #print(f'pixel = {pixel}')
            clipped = clip_with_line(pixel, edge)
            if clipped.shape[0] < 3:
                continue
            A1, c1, I1 = shape_stats(clipped)
            
            clipped = clip_with_line(pixel, np.flipud(edge))
            if clipped.shape[0] < 3:
                continue
            A2, c2, I2 = shape_stats(clipped)
            
            if A1 > A2:
                A = A1
                big_centroids[:,pixel[0], pixel[1]] = c1
                lil_centroids[:,pixel[0], pixel[1]] = c2
                big_I[pixel[0], pixel[1]] = I1
                lil_I[pixel[0], pixel[1]] = I2
            else:
                A = A2
                big_centroids[:,pixel[0], pixel[1]] = c2
                lil_centroids[:,pixel[0], pixel[1]] = c1
                big_I[pixel[0], pixel[1]] = I2
                lil_I[pixel[0], pixel[1]] = I1
            areas[pixel[0],pixel[1]] = A
    return areas.T, np.stack((big_centroids.T, lil_centroids.T)), np.stack((big_I.T, lil_I.T))

def get_edge_pixels(polygon, shape):
    '''
    For a given set of polygon vertices, get all the pixels that are intersected by the edges
    '''
    h, w = shape
    edges = np.stack((polygon, np.roll(polygon, -1, axis=0)), axis =1)
    
    # make a bounding box that holds the polygon and has pixel indices as values
    #xmax, ymax = np.ceil(polygon.max( axis=0 )).astype(int)
    #xmin, ymin = np.floor(polygon.min( axis=0 )).astype(int)

    pixels = np.zeros((h, w)).astype(bool)
    for edge in edges:
        p1, p2 = edge[0], edge[1]
        if (abs(p1[0] - p2[0]) < 1e-6):
            # Take care of vertical lines first
            y_int = np.arange(min(edge[:,1]), max(edge[:,1]))
            x_int = np.array([int(p1[0])] * y_int.size)
        else:
            # Interpolate the line between two vertices and cast back onto an integer grid
            m = (p1[1] - p2[1]) / (p1[0] - p2[0]) # get slope of line

            x_int = np.linspace(p1[0], p2[0], w//2)
            y_int = (m * (x_int - p1[0]) + p1[1]) # get y values along line
        pix_edge = np.around(np.stack((x_int, y_int)).T).astype(int) # cast to integers
        pixels[pix_edge[:,1], pix_edge[:,0]] = True # make a boolean mask of edges
    return pixels

def clip_pixel(point, polygon):
    x0, y0 = point[0], point[1]
    t,b,l,r = y0 + .5, y0 - .5, x0 - .5, x0 + .5
    pixel = np.array([[l, b], [l, t], [r, t], [r, b]])
    #print(f'Clipping pixel at {point}, vertices = \n{pixel}')
    return clip(polygon, pixel)

def get_pixels_in_cell(cell, areas, centroids, Is, shape):
    if cell.size == 0:
        print(f'Cell has zero vertices! cell = {cell}')
        return np.zeros_like(areas), np.zeros_like(centroids[0]), np.zeros_like(Is[0])
    
    # First get a basic set of pixels whos centers are inside the polygon
    h, w = shape
    Y, X = np.indices(shape)
    pix_areas = np.zeros((h, w))
    
    points = np.stack((X.flatten(),Y.flatten())).T
    
    # Inside 
    cell_closed = np.append(cell, [cell[0]], axis=0)
    insides = points_inside(points, cell_closed).reshape((h,w))
    pix_areas[insides] = 1
    
    # get all the edge pixels:
    edges = get_edge_pixels(cell, shape)
    pix_areas[edges] = areas[edges]
    pix_centroids = np.copy(centroids[0])
    pix_Is = np.copy(Is[0])

    # if edge pixels are not inside the polygon, use the smaller area
    pix_areas[edges * ~insides] = 1 - areas[edges * ~insides]
    pix_centroids[edges * ~insides,:] = centroids[1,edges * ~insides,:]
    pix_Is[edges * ~insides] = Is[1,edges * ~insides]

    # fix the pixel areas containing vertices
    for vertex in cell:
        vpix = np.around(vertex).astype(int)
        clipped = clip_pixel(vpix, cell)
        if clipped.size == 0:
            print(f'Clipped vertex pixel has no vertices')
            A = 0
            c = vertex
        else:
            A, c, _ = shape_stats(clipped)
        pix_areas[vpix[1], vpix[0]] = A
        pix_centroids[vpix[1], vpix[0],:] = c
    return pix_areas, pix_centroids, pix_Is

def shape_stats(vertices):
    #print(f'vertices = {vertices.T}')

    x1,y1 = vertices.T
    x2,y2 = np.roll(x1, -1), np.roll(y1, -1)
    
    # Calculate the area
    A = np.sum(x2 * y1 - x1 * y2) / 2.
    
    # Calculate the center of mass
    if A == 0:
        cx = np.mean(x1)
        cy = np.mean(y1)
    else:
        cx = np.sum((x1 + x2) * (x2 * y1 - x1 * y2)) / (6*A)
        cy = np.sum((y1 + y2) * (x2 * y1 - x1 * y2)) / (6*A)
    
    Ix = np.sum((y1**2 + y1 * y2 * y2**2) * (x2 * y1 - x1 * y2)) / 12.
    Iy = np.sum((x1**2 + x1 * x2 * x2**2) * (x2 * y1 - x1 * y2)) / 12.
    I0 = Ix + Iy
    #print(f'A = {A:.4e}, c = ({cx:.2f}, {cy:.2f}), I_0 = {I0:.2e}')
    return A, [cx,cy], I0

def cell_weighted_stats(cell, areas, centroids, Is, image):
    area_map, centroid_map, I_map = get_pixels_in_cell(cell, areas, centroids, Is, image.shape)
    A = np.sum(image * area_map)
    cx = np.sum(image * centroid_map[:,:,0] * area_map)/A
    cy = np.sum(image * centroid_map[:,:,1] * area_map)/A
    
    I = np.sum(image * I_map)
    return A, [cx, cy], I
    
def weighted_stats(vor):
    areas, centroids, Is = get_pixel_areas(vor.vertices, *vor.shape)
    weighted_areas = np.zeros(vor.N)
    weighted_Is = np.zeros(vor.N)
    weighted_centroids = np.zeros((vor.N,2))
    for i in range(vor.N):
        cell = vor.regions[i]
        weighted_areas[i], weighted_centroids[i,:], weighted_Is[i] = cell_weighted_stats(cell, areas, centroids, Is, vor.image)
    return weighted_areas, weighted_centroids, weighted_Is

if __name__ == "__main__":
    
    # set plotting style
    plt.style.reload_library()
    plt.style.use('ucla_presentations')

    # define a handy function for wiping the ticks and labeles off of plot axes
    def no_ticks(ax=None):
        if ax is None:
            ax = plt.gca()
        return ax.tick_params(left=False,
                              bottom=False,
                              labelleft=False,
                              labelbottom=False)
    # and another for setting the axis limits to stay where they are when plotting multiple things
    def freeze_ax(ax=None):
        if ax is None:
            ax = plt.gca()
        ax.set_xlim(ax.get_xlim())
        ax.set_ylim(ax.get_ylim())
        return

    # Generate some funky test data
    h, w = 500,700
    shape = (h,w)

    Y, X = np.indices(shape)
    def makeGaussian(shape, fwhm = 3, center=None):
        """ Make a square gaussian kernel.

        size is the length of a side of the square
        fwhm is full-width-half-maximum, which
        can be thought of as an effective radius.
        """
        y, x = np.indices(shape)
        if center is None:
            y0, x0 = shape[0] / 2, shape[1] / 2
        else:
            x0 = center[0]
            y0 = center[1]

        return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

    z = np.sin(w+3*X/w)**10 + 2*np.cos(9*Y*X/ (h*w)) * np.cos(w+3*X/w)
    test_bg = makeGaussian((h,w), fwhm = h//2)
    test_sig = abs((np.max(z) - abs(z)))

    noise = 0.01 * np.random.rand(h,w)
    noise_bg = 0.01 * np.random.rand(h,w)
    test_sig += noise
    test_bg += noise_bg

    # normalize the signal to sum to 1
    test_sig /= np.sum(test_sig)
    test_bg /= np.sum(test_bg)

    fig, (ax1, ax2) = plt.subplots(1,2)

    ax1.pcolormesh(X,Y, test_bg, cmap='gray')
    ax1.set_aspect('equal')
    ax1.set_title("Test Background Signal")

    ax2.pcolormesh(X,Y, test_sig, cmap='gray')
    ax2.set_aspect('equal')
    ax2.set_title("Test Signal")
    no_ticks(ax1)
    no_ticks(ax2)
    plt.show()
    
    N = 100

    # Use the image intensities as the weights for each pixel
    weights = test_sig.flatten() / np.sum(test_sig)
    # Randomly pick an index
    idx = np.arange(len(weights))

    rand_idx = np.random.choice(idx, N, replace = False, p = weights)

    # Get the coordinates of each sampled point
    points = np.array([X.flatten()[rand_idx], Y.flatten()[rand_idx]], dtype=float).T

    bg = Voronoi(points, shape=(h,w), image=test_bg)

    fig, ax1 = plt.subplots()

    ax1.pcolormesh(X,Y, test_bg, cmap='gray')
    ax1.set_aspect('equal')
    bg.plot(ax=ax1, transparent=True)
    ax1.set_title("Background Signal Before Lloyd Relaxation")
    no_ticks(ax1)
    ax1.plot(bg.c[:,0], bg.c[:,1], 'cx')

    plt.show()
    '''
    bg.lloyd()
    fig, ax2 = plt.subplots()

    ax2.pcolormesh(X,Y, test_bg, cmap='gray')
    ax2.set_aspect('equal')
    bg.plot(ax=ax2, transparent=True)
    ax2.set_title("Background Signal After Lloyd Relaxation")
    no_ticks(ax2)
    plt.show()'''

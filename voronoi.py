'''
Functions for calculating and managing Voronoi/power diagrams
'''
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import multiprocessing as mp

import shapes
class Voronoi:
    """
    Class for calculating 2D power diagrams, i.e. weighted Voronoi tesselations
    """
    def __init__(self, sites, shape = None, weights=None, image = None, clip = True):
        self._sites = sites
        self.N = sites.shape[0]
        
        #h, w = shape
        # Define the corner points of the box in clockwise order
        #self.box = shapes.Polygon(np.array([[0,0],[0,h],[w,h],[w,0]], dtype=float)-0.5)
        # Define each edge of the box

        if weights is None:
            self._weights = np.zeros(self.N)
        else:
            self._weights = weights
            
        if image is None:
            self.image = np.ones(shape)
            self.shape = shape
        else:
            self.image = image
            self.shape = image.shape
        self.clip = clip
        # Calculate the weighted Voronoi diagram
        self.get_Voronoi()
    
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
        self.get_Voronoi()
    
    @property
    def weights(self):
        """The weights corresponding to each site in the power diagram."""
        return self._weights
    
    @weights.setter
    def weights(self, values):
        """Setter function for the weights of each sites."""
        # if the weights change, we need to recalculate the power diagram.
        if not np.array_equal(values, self._weights):
            self._weights = values
            self.get_Voronoi()
    
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
        
        # Save the finite triangles for use in natural neighbor interpolation and their
        # corresponding circumcenters
        delaunay_triangles = self.hull.simplices[self.lowers]
        
        self.triangles = delaunay_triangles
        self.circumcenters = all_vertices[self.lowers]
        # Get the circumradius of each delaunay triangle from its vertices and circumcenter
        r = np.sqrt(np.sum((self.sites[self.triangles[:,0]] - self.circumcenters)**2, axis=1))
        self.radii = r
        
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
        return self.get_cells()
    
    def get_cells(self):
        '''
        From all of the delaunay edges and Voronoi vertices get the vertices defining each
        cell in the power diagram and define a Polygon object for that cell
        '''
        args = [(i, self.delaunay_edges, self.vertices, self.image, self.clip) for i in range(self.N)]
        pool = mp.Pool()
        cells = pool.map(get_cell_wrapper, args)
        
        self.regions = cells
        stats = [(cell.A, cell.c, cell.I) for cell in cells]
        stats = list(zip(*stats))
        
        self.A, self.c, self.I = np.array(stats[0]), np.array(stats[1]), np.array(stats[2])
        
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
        
    def lloyd(self, threshold = 0.05, MAXDEPTH=50, verbose = True):
        """
        Performs lloyd relaxation on a voronoi diagram object
        
        Convergence occurs when the maximum relative shift in a centroid
        """
        # Get threshold distance in pixels
        centroids = self.c
        for i in range(MAXDEPTH):
            old_centroids = np.copy(centroids)
            self.sites = centroids
            centroids = self.c
            #error = np.std(self.A)/(np.sqrt(self.N) * np.mean(self.A))
            
            convergence = abs(centroids - old_centroids)/ old_centroids
            if verbose:
                print(f'Centroids shifted by up to {np.max(convergence)*100:.2f}% after {i+1} iterations', end='\r')
            if np.all(convergence < threshold):
                print(f"All centroids within {threshold * 100}% after {i+1} iterations of Lloyd relaxation")
                break
        return
    
    def plot(self, ax = None, sites = True, transparent = False, plot_weights = False, plot_delaunay = False, plot_cells = True, plot_image = False,  **kwargs):
        """
        Plot the power diagram
        """
        h, w = self.shape

        if ax is None:
            ax = plt.gca()
        
        # Fix the axes of the plot to the size of the image (taking into account pixel size on the edges)
        ax.set_ylim((-.5,h-.5))
        ax.set_xlim((-.5,w-.5))
        
        if plot_image:
            transparent = True
            Y, X = np.indices(self.shape)
            ax.pcolormesh(X, Y, self.image, cmap='gray')

        colors = mpl.cm.get_cmap(name='rainbow')(np.linspace(0,1,self.N))
        
        if transparent:
            alph = 0
            linecol = 'cyan'
            sitecol = 'red'
        else:
            alph = 0.3
            linecol = 'black'
            sitecol = 'black'
        
        if plot_delaunay:
            for idx in self.delaunay_edges:
                pts = self.sites[idx]
                ax.plot(pts[:,0], pts[:,1], 'b', lw = 0.5)
        for i, site in enumerate(self.sites):
            cell = self.regions[i]
            if plot_cells:
                cell.plot(ax=ax,facecolor=mpl.colors.to_rgba(colors[i],alph),plot_points=False, edgecolor=linecol, **kwargs)
            if sites:
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
        
def get_cell(i, edges, vertices, image, clip = True):
    # Find the indices of the Delaunay edges containing each site
    edge_idx = np.argwhere(edges == i)[:,0]

    # Get the Voronoi vertices from each of those delaunay edges
    verts = vertices[edge_idx]
    if verts.size == 0:
        return shapes.WeightedPolygon(verts, image)

    # Remove any duplicate vertices
    verts = np.unique(np.reshape(verts, (-1,2)), axis = 0)
    c = np.mean(verts, axis = 0)

    # Sort the vertices in clockwise order
    verts = np.array(sorted(verts,
                            key = lambda v: np.arctan2((v[0] - c[0]), (v[1] - c[1]))))
    cell = shapes.WeightedPolygon(verts, image)
    
    if clip:
        # Crop to bounding box
        h, w = image.shape
        box = shapes.Polygon(np.array([[0,0],[0,h],[w,h],[w,0]], dtype=float)-0.5)
        cell.clip_to(box)
    return cell

def get_cell_wrapper(args):
    return get_cell(*args)
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

class Voronoi:
    """
    Class for calculating 2D power diagrams, i.e. weighted Voronoi tesselations
    """
    def __init__(self, sites, shape, weights=None):
        self.sites = sites
        self.N = sites.shape[0]
        self.shape = shape
        if weights is None:
            self.weights = np.zeros(self.N)
        else:
            self.weights = weights
        
        # Calculate the power diagram
        # Reflect sites over the edges of the box
        points = self.mirror_points()
        # Project the 2D sites onto a 3D paraboloid
        points_3D = self.lift_points(points, np.tile(self.weights,5))
        
        # Calculate the Convex Hull
        hull = ConvexHull(points_3D)
        # Get the downward facing faces of the hull
        lowers = self.get_lower_facing(hull)
        # Get the voronoi vertices from the Delaunay triangles
        triangles = hull.simplices[lowers]
        eqns = hull.equations[lowers]
        vertices = np.array([-0.5 * eq[:2] / eq[2] for eq in eqns])

        self.triangles = triangles
        self.regions = []
        
        for i, site in enumerate(self.sites):
            # get the indices of all triangles that have point i as a vertex
            indices = np.where(triangles==i)[0]
            
            # get the circumcenter of each of those triangles
            verts = vertices[indices]

            # Sort the vertices in clockwise order
            verts = np.array(sorted(verts,
                                    key = lambda v: np.arctan2((v[1] - site[1]), (v[0] - site[0]))))
            self.regions.append(verts)


    def mirror_points(self):
        """
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
        
    def get_lower_facing(self, hull):
        """
        Return the indices of simplexes in hull that are lower-facing
        """
        return np.where(hull.equations[:,2] < 0)[0]
    
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
    
    def plot(self, ax = None, transparent = False, plot_weights = False, plot_delaunay = True):
        """
        Plot the power diagram
        """
        h, w = self.shape

        if ax is None:
            ax = plt.gca()

        ax.set_ylim((0,h))
        ax.set_xlim((0,w))
        
        colors = mpl.cm.get_cmap(name='rainbow')(np.linspace(0,1,self.N))
        
        if transparent:
            alph = 0
            linecol = 'cyan'
            sitecol = 'red'
        else:
            alph = 0.5
            linecol = 'black'
            sitecol = 'black'
        
        """if plot_delaunay:
            for i, s in enumerate(self.triangles):
                s = np.append(s, s[0])  # Here we cycle back to the first coordinate

                # Projecting these back to z=0 gives the Delaunay Triangulation
                ax.plot(self.sites[s, 0], self.sites[s, 1],  'b-', lw=0.5)"""
        for i, site in enumerate(self.sites):
            vertices = self.regions[i]
            if vertices.size == 0:
                print(f"Site at i = {i} is empty, moving along")
                continue

            ax.fill(vertices[:,0], vertices[:,1], 
                    facecolor=mpl.colors.to_rgba(colors[i],alph),
                    edgecolor=linecol)

            ax.plot(site[0], site[1], 'o', c = sitecol)
            if plot_weights:
                ax.add_patch(plt.Circle((site[0], site[1]), 
                                        np.sqrt(self.weights[i]), 
                                        color='black', fill=False, lw=0.5))

        ax.set_title('Power Diagram')
        ax.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)

def centroid(vertices):
    """
    Calculate the centroid of a polygon from its vertices
    """
    if vertices.size == 0:
        return np.nan
    
    v_i = vertices
    v_ip1 = np.roll(vertices, -1, axis=0)
    
    xy = v_ip1 * np.fliplr(v_i)
    xy = np.array([xy[:,1] - xy[:,0]])
    
    c = ((v_i + v_ip1).T @ xy.T).T[0]
    
    return c / (6 * area(vertices))

def area(vertices):
    """
    Calculate the area of a polygon from its vertices
    """
    v_i = vertices
    v_ip1 = np.roll(vertices, -1, axis=0)
    
    xy = v_ip1 * np.fliplr(v_i)
    xy = np.array([xy[:,1] - xy[:,0]])
    return np.sum(xy)/2.
        
def lloyd(vor, MAXDEPTH=100):
    """
    Performs lloyd relaxation on a voronoi diagram object
    """
    shape = vor.shape
    weights = vor.weights
    centroids = np.array([centroid(region) for region in vor.regions])
    for i in range(MAXDEPTH):
        vor = Voronoi(centroids, shape, weights)
        old_centroids = np.copy(centroids)
        centroids = np.array([centroid(region) for region in vor.regions])
        if np.array_equal(centroids,old_centroids):
            print(f"Centroids no longer changing at i={i+1} iterations")
            break
    return vor
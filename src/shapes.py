'''
Classes and functions for creating and manipulating polygons, with and without weighted
statistics based on an image
'''
import numpy as np
import matplotlib.pyplot as plt
from . import rasterize as ras

class Polygon:
    '''
    Each polygon is defined by vertices given in clockwise order
    '''
    def __init__(self, vertices):
        self._vertices = vertices
        self.N = vertices.shape[0]

        self.A, self.c, self.I = calculate_shape_stats(self)
    
    @property
    def vertices(self):
        return self._vertices
    
    @vertices.setter
    def vertices(self, vertices):
        ''' Anytime the vertices change, we need to recalculate the shape stats '''
        self._vertices = vertices
        self.N = vertices.shape[0]
        self.A, self.c, self.I = calculate_shape_stats(self)

    def clip_to(self, boundary):
        ''' clip the polygon to the given boundary polygon using Hodgman-Sutherland '''
        bound_verts = boundary.vertices
        if boundary.N == 2:
            # clipping to a line, rather than a shape: don't close it
            box_edges = [bound_verts]
        else:
            box_edges = np.stack((bound_verts, np.roll(bound_verts, -1, axis=0)), axis =1)
        clipped = np.copy(self.vertices)
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
        self.vertices = clipped
        return
    
    def plot(self, ax = None, plot_points = True, **kwargs):
        if ax is None:
            ax = plt.gca()
        if 'facecolor' not in kwargs:
            kwargs['facecolor'] = (0,0,0,0)
        if 'edgecolor' not in kwargs:
            kwargs['edgecolor'] = 'k'
        verts = self.vertices.T
        ax.fill(verts[0], verts[1], **kwargs)
        
        if plot_points:
            ax.plot(verts[0], verts[1], 'o', c = kwargs['edgecolor'], zorder=3)
        
        ax.set_aspect('equal')

def clockwise_sort(verts):
    '''Sort an array of polygon vertices into clockwise order'''
    # Calculate the mean coordinate to estimate the center of the polygon (for clockwise sorting)
    if verts.shape[0] == 1:
        return verts
    center = np.mean(verts, axis=0)
    # sort the vertices in clockwise order
    sorted_verts = np.array(sorted(verts,
                                   key = lambda v: np.arctan2((v[0] - center[0]), 
                                                              (v[1] - center[1]))))
    return sorted_verts

class Pixel(Polygon):
    ''' A pixel is defined by a single point at its center, with assumed area of 1 '''
    def __init__(self, point):
        x0, y0 = point
        t,b,l,r = y0 + .5, y0 - .5, x0 - .5, x0 + .5
        vertices = np.array([[l, b], [l, t], [r, t], [r, b]])
        super().__init__(vertices)
        
class Triangle(Polygon):
    '''A triangle is the same as a polygon, but has functions for circumcircles that don't 
    apply to all polygons'''
    def __init__(self, vertices):
        # Check that there are precisely three vertices
        if vertices.shape[0] != 3:
            print(f'WARNING: A triangle must have three vertices, but {vertices.shape[0]} were given.')
        super().__init__(vertices)
        self._circumcenter, self._circumradius = self.get_circumcircle()
    
    @property
    def circumcenter(self):
        return self._circumcenter
    
    @property
    def circumradius(self):
        return self._circumradius

    def get_circumcircle(self):
        p_x, p_y = self.vertices[:,0][:, None], self.vertices[:,1][:, None]

        side_lengths = (p_x**2 + p_y**2)
        ones = np.ones((3,1))
        S_x = np.hstack((side_lengths, p_y, ones))
        S_y = np.hstack((p_x, side_lengths, ones))
        
        S_x = 1/2 * np.linalg.det(S_x)
        S_y = 1/2 * np.linalg.det(S_y)
        
        a = np.linalg.det(np.hstack((p_x, p_y, ones)))
        b = np.linalg.det(np.hstack((p_x, p_y, side_lengths)))
        
        center = np.array([S_x, S_y]) / a
        radius = np.sqrt(b/a + (S_x**2 + S_y**2)/a**2)
        
        return center, radius
    
    def inside_circle(self, point):
        c = self._circumcenter
        d = np.sum((point - c)**2)
        if d < self._circumradius**2:
            return True
        return False
    
    def plot(self, ax = None, plot_points = True, plot_circle = False, plot_center = False, circle_options = {}, **kwargs):
        if self.N == 0:
            return
        if ax is None:
            ax = plt.gca()
        if 'facecolor' not in kwargs:
            kwargs['facecolor'] = (0,0,0,0)
        if 'edgecolor' not in kwargs:
            kwargs['edgecolor'] = 'k'
        
        if plot_circle:
            if 'facecolor' not in circle_options:
                circle_options['facecolor'] = (0,0,0,0)
            if 'edgecolor' not in circle_options:
                circle_options['edgecolor'] = 'k'
            ax.add_patch(plt.Circle(self._circumcenter, self._circumradius, 
                                    lw=0.5,
                                    **circle_options))
        if plot_center:
            ax.plot(self._circumcenter[0], self._circumcenter[1], 'x', ms=2, mew=.5, 
                    c=kwargs['edgecolor'])
            
        verts = self.vertices.T
        ax.fill(verts[0], verts[1], **kwargs)
        
        if plot_points:
            if 'zorder' not in kwargs:
                zo = 1
            else:
                zo = kwargs['zorder']
            ax.plot(verts[0], verts[1], 'o', c = kwargs['edgecolor'], zorder = zo)
        
        ax.set_aspect('equal')
        
class WeightedPolygon:
    '''
    Polygons defined by clockwise vertices, properties are weighted by the intensities
    of each pixel in the rasterized polygon
    '''
    def __init__(self, vertices, image):
        self.image = image
        self._vertices = vertices
        self.N = vertices.shape[0]
        
        self.A, self.c, self.I = calculate_weighted_stats(self)
        
    @property
    def vertices(self):
        return self._vertices
    
    @vertices.setter
    def vertices(self, vertices):
        ''' Anytime the vertices change, we need to recalculate the shape stats '''
        self._vertices = vertices
        self.N = vertices.shape[0]
        self.A, self.c, self.I = calculate_weighted_stats(self)

    def clip_to(self, boundary):
        ''' clip the polygon to the given boundary polygon using Hodgman-Sutherland '''
        bound_verts = boundary.vertices
        if boundary.N == 2:
            # clipping to a line, rather than a shape: don't close it
            box_edges = [bound_verts]
        else:
            box_edges = np.stack((bound_verts, np.roll(bound_verts, -1, axis=0)), axis =1)
        clipped = np.copy(self.vertices)
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
        self.vertices = clipped
        return
     
    def plot(self, ax = None, plot_points = True, **kwargs):
        if self.N == 0:
            return
        if ax is None:
            ax = plt.gca()
        if 'facecolor' not in kwargs:
            kwargs['facecolor'] = (0,0,0,0)
        if 'edgecolor' not in kwargs:
            kwargs['edgecolor'] = 'k'
        verts = self.vertices.T
        ax.fill(verts[0], verts[1], **kwargs)
        
        if plot_points:
            ax.plot(verts[0], verts[1], 'o', c = kwargs['edgecolor'], zorder=3)
        
        ax.set_aspect('equal')

        
def calculate_shape_stats(polygon):
    ''' calculate the area, centroid, and moment of inertia of a 2D convex polygon given 
    by its vertices in clockwise order'''
    if polygon.N == 0:
        return 0, (-1,-1), 0
    x1,y1 = polygon.vertices.T
    x2,y2 = np.roll(x1, -1), np.roll(y1, -1)

    # Calculate the area
    A = np.sum(x2 * y1 - x1 * y2) / 2.

    # Calculate the center of mass
    if A == 0:
        cx = np.mean(x1)
        cy = np.mean(y1)
        Ix = 0
        Iy = 0
    else:
        cx = np.sum((x1 + x2) * (x2 * y1 - x1 * y2)) / (6*A)
        cy = np.sum((y1 + y2) * (x2 * y1 - x1 * y2)) / (6*A)

        Ix = np.sum((y1**2 + y1 * y2 + y2**2) * (x2 * y1 - x1 * y2)) / 12.
        Iy = np.sum((x1**2 + x1 * x2 + x2**2) * (x2 * y1 - x1 * y2)) / 12.

    I0 = Ix + Iy
    return A, np.array([cx, cy]), I0

def calculate_weighted_stats(weighted_poly):
    shape = weighted_poly.image.shape
    image = weighted_poly.image
    A_map, c_map, I_map = ras.rasterize(weighted_poly, shape)
    
    A = np.sum(image * A_map)
    if A == 0:
        return calculate_shape_stats(weighted_poly)
    cx = np.sum(image * A_map * c_map[0]) / A
    cy = np.sum(image * A_map * c_map[1]) / A
    I0 = np.sum(image * I_map)
    
    return A, np.array([cx, cy]), I0
    
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

def area(vertices):
    x1,y1 = vertices.T
    x2,y2 = np.roll(x1, -1), np.roll(y1, -1)

    # Calculate the area
    A = np.sum(x2 * y1 - x1 * y2) / 2.
    return A

def circumcenter(vertices):
    p_x, p_y = vertices[:,0][:, None], vertices[:,1][:, None]

    side_lengths = (p_x**2 + p_y**2)
    ones = np.ones((3,1))
    S_x = np.hstack((side_lengths, p_y, ones))
    S_y = np.hstack((p_x, side_lengths, ones))

    S_x = 1/2 * np.linalg.det(S_x)
    S_y = 1/2 * np.linalg.det(S_y)

    a = np.linalg.det(np.hstack((p_x, p_y, ones)))
    #b = np.linalg.det(np.hstack((p_x, p_y, side_lengths)))

    center = np.array([S_x, S_y]) / a
    #radius = np.sqrt(b/a + (S_x**2 + S_y**2)/a**2)

    return center#, radius

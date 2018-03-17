class kdtree:
    def __init__(self, points, dimensions):
        self.dimensions = dimensions
        self.tree = self._build([(i, p) for i,p in enumerate(points)])
        
    def closest(self, point):
        return self._closest(self.tree, point)
        
    def _build(self,points, depth=0):
        n = len(points)

        if n<= 0:
            return None

        dimension = depth % self.dimensions
        sorted_points = sorted(points, key=lambda point:point[1][dimension])

        mid = n//2

        return {
            'mid':sorted_points[mid],
            'left':self._build(sorted_points[:mid], depth+1),
            'right':self._build(sorted_points[mid+1:], depth+1)
        }
    
    def _distance(self, point, node):
        if point == None or node == None:
            return float('Inf')
        
        d = 0
        for p,n in zip(list(point),list(node[1])):
            d += (p - n)**2
        
        return d
    
    def _closest(self, root, point, depth=0):
        if root is None:
            return None

        dimension = depth % self.dimensions   
        mid_point = root['mid']

        next_branch     = root['left'] if point[dimension] < mid_point[1][dimension] else root['right']
        opposite_branch = root['right'] if point[dimension] < mid_point[1][dimension] else root['left']

        best_in_next_branch = self._closest(next_branch, point, depth+1)

        distance_to_mid_point = self._distance(point, mid_point)
        distance_to_next_branch = self._distance(point, best_in_next_branch)

        best = best_in_next_branch if distance_to_next_branch < distance_to_mid_point else mid_point
        best_distance = min(distance_to_next_branch, distance_to_mid_point)

        if best_distance > abs(point[dimension] - mid_point[1][dimension]):
            best_in_opposite_branch = self._closest(opposite_branch, point, depth+1)
            distance_to_opposite_branch = self._distance(point, best_in_opposite_branch)
            best = best_in_opposite_branch if distance_to_opposite_branch < best_distance else best

        return best
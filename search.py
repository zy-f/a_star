__author__ = 'cpolzak'

import numpy as np
from heapQueue import MinPriorityQueue
from linkedList import LinkedList

class Node(object):
    def __init__(self, pos, parent):
        self.pos = pos
        self.parent = parent
        self.g = self.parent.g if parent is not None else 0
        self.f = self.g
    
    def __repr__(self):
        return f"POS={self.pos}  G={self.g}  F={self.f}"

    def __eq__(self, other):
        return isinstance(other, Node) and self.f == other.f
    def __lt__(self, other):
        return isinstance(other, Node) and self.f < other.f
    def __gt__(self, other):
        return not self.__eq__(other) and not self.__lt__(other)
    def __le__(self, other):
        return not self.__gt__(other)
    def __ge__(self, other):
        return not self.__lt__(other)
    def __ne__(self, other):
        return not self.__eq__(other)

class FringeNode(Node):
    def __init__(self, *args):
        super(FringeNode, self).__init__(*args)
    
    def __eq__(self, other):
        return isinstance(other, Node) and tuple(self.pos) == tuple(other.pos)

class AStarSearch(object):
    # higher elev_lambda => more weight on elevation change
    def __init__(self, elev_grid, pixel_size=100, heuristic_lambda=.5, fwd_only=False):
        # converts grid to x = row, y = col
        self.elev_grid = elev_grid.transpose(1,0)
        self.pixel_size = pixel_size
        self.h_lambda = heuristic_lambda
        dir_grid = np.indices((3,3)).transpose((2,1,0)) - 1
        if fwd_only:
            dir_grid = dir_grid[:,-1]
        # flattens to list of coordinates, removes current position
        dir_grid = dir_grid.reshape(-1,2)
        self.dir_grid = dir_grid[np.any(dir_grid,axis=-1)]

    def get_neighbor_positions(self, pos):
        neighbor_pos = pos + self.dir_grid
        # excludes positions that don't exist on the grid
        grid_dims = self.elev_grid.shape
        in_grid_mask = (neighbor_pos[:,0] < grid_dims[0]) & (neighbor_pos[:,1] < grid_dims[1]) & (neighbor_pos>=0).all(axis=-1)
        return [tuple(p) for p in neighbor_pos[in_grid_mask]]
    
    def movement_cost(self, pos1, pos2):
        dist_cost = np.sqrt((pos2[0]-pos1[0])**2 + (pos2[1]-pos1[1])**2)
        height_delta = np.abs(self.elev_grid[pos2] - self.elev_grid[pos1])
        height_cost =  height_delta/self.pixel_size
        return dist_cost + height_cost

    def search(self, start_y, end_y):
        start = (0,start_y)
        end = (self.elev_grid.shape[0]-1, end_y)
        open_nodes = MinPriorityQueue()
        root = Node(start, None)
        seen_nodes = {}

        open_nodes.add(root)
        iters = 0
        current = root
        while current.pos != end: # while not at right edge
            # print(iters)
            current = open_nodes.pop_root()
            # print(open_nodes)
            # if iters > 10:
            #     0/0
            if iters%1000 == 0 and iters>0:
                print('DEBUG', iters, len(seen_nodes), open_nodes[0].pos)
            seen_nodes[current.pos] = current
            for pos in self.get_neighbor_positions(current.pos):
                new_node = Node(pos, current)
                new_node.g += self.movement_cost(current.pos, pos)
                # if seen_nodes.get(pos) and new_node.g < seen_nodes.get(pos).g:
                #     del seen_nodes[pos]
                if not seen_nodes.get(pos):
                    new_node.f = new_node.g + self.h_lambda*self.movement_cost(pos, end)
                    seen_nodes[pos] = new_node
                    open_nodes.add(new_node)
            iters += 1
        # get coordinates to draw path
        path = []
        while current is not None:
            path.append(current.pos[::-1]) # convert (x,y) to (y,x) for rendering
            current = current.parent

        return path, [pos[::-1] for pos in seen_nodes.keys()]

    if __name__ == '__main__':
        search(0,None)


class FringeSearch(AStarSearch):
    def __init__(self, elev_map, **kwargs):
        super(FringeSearch,self).__init__(elev_map, **kwargs)
        self.fringe = LinkedList(double_link=True)
        self.node_cache = {}
    
    def get_neighbor_positions(self, pos):
        neighbor_pos = pos + self.dir_grid
        # excludes positions that don't exist on the grid
        grid_dims = self.elev_grid.shape
        in_grid_mask = (neighbor_pos[:,0] < grid_dims[0]) & (neighbor_pos[:,1] < grid_dims[1]) & (neighbor_pos>=0).all(axis=-1)
        return neighbor_pos[in_grid_mask]
    
    def movement_cost(self, pos1, pos2):
        dist_cost = np.sum((pos2 - pos1)**2)
        height_delta = self.elev_grid[tuple(pos2)] - self.elev_grid[tuple(pos1)]
        height_cost = (height_delta**2)/self.pixel_size
        return dist_cost + height_cost
    
    def search(self, start_y, end_y):
        self.start = np.array((0,start_y))
        self.end = np.array((self.elev_grid.shape[0]-1, end_y))
        
        root = FringeNode(self.start, None) # pos, parent
        self.fringe.append(root)
        self.node_cache[tuple(self.start)] = root
        now_len = 1
        
        f_limit = self.movement_cost(root.pos, self.end)
        goal_node = None
        while goal_node is None:
            f_min = float('inf')
            k = 0
            link_el = self.fringe.start
            print(len(self.node_cache.keys()))
            while k < now_len:
                node = link_el.v
                f = node.g + self.movement_cost(node.pos, self.end)
                if f > f_limit:
                    f_min = min(f, f_min)
                    k += 1
                    continue
                if tuple(node.pos) == tuple(self.end):
                    goal_node = node
                    break
                for child_pos in self.get_neighbor_positions(node.pos):
                    child_node = FringeNode(child_pos, node)
                    child_node.g += self.movement_cost(node.pos, child_pos)
                    if self.node_cache.get(tuple(child_pos)) is not None and child_node.g >= self.node_cache[tuple(child_pos)].g:
                        continue
                    self.fringe.remove(child_node)
                    self.fringe.insert(k+1, child_node)
                    now_len += 1
                    self.node_cache[tuple(child_node.pos)] = child_node
                self.fringe.remove(node)
                k += 1
                link_el = link_el.r
            f_limit = f_min
        
        path = []
        current = goal_node
        while current is not None:
            path.append(tuple(current.pos[::-1])) # convert (x,y) to (y,x) for rendering
            current = current.parent

        return path, [pos[::-1] for pos in node_cache.keys()]
        

                    
                    
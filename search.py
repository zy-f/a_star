__author__ = 'cpolzak'

import numpy as np
from heapQueue import MinPriorityQueue

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

class AStarSearch(object):
    # higher elev_lambda => more weight on elevation change
    def __init__(self, elev_grid, elev_lambda=.5, pixel_size=100, heuristic_lambda=1, fwd_only=False):
        # converts grid to x = row, y = col
        self.elev_grid = elev_grid.transpose(1,0)
        self.pixel_size = pixel_size
        self.h_lambda = heuristic_lambda
        self.e_lambda = elev_lambda
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
        return (1-self.e_lambda)*dist_cost + self.e_lambda*height_cost

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
                # if seen_nodes.get(pos) is not None and new_node.g < seen_nodes.get(pos).g:
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

class Fringe(object):
    def __init__(self, full_copy=None, size=1):
        if full_copy is not None:
            self.poss, self.gs, self.fs, self.parents = full_copy
        else:
            size **= 2
            self.poss = np.empty((size, 2), dtype=int)
            self.gs = np.empty(size, dtype=float)
            self.fs = np.empty(size, dtype=float)
            self.parents = np.empty((size, 2), dtype=int)
    
    def __getitem__(self, i):
        return Fringe(full_copy=(self.poss[i], self.gs[i], self.fs[i], self.parents[i]))
    
    def get(self, i):
        return (self.poss[i], self.gs[i], self.fs[i], self.parents[i])

    def __setitem__(self, i, value):
        self.poss[i], self.gs[i], self.fs[i], self.parents[i] = value
    
    def __len__(self):
        return len(self.gs)
    

class SemiFringeSearch(AStarSearch):
    def __init__(self, elev_map, **kwargs):
        super(SemiFringeSearch,self).__init__(elev_map, **kwargs)
        self.fringe = Fringe(size=len(elev_map.flatten()))
        self.node_cache = {}

    def movement_cost(self, pos1, pos2):
        dist_cost = np.sum((pos2 - pos1)**2, axis=1)
        height_delta = self.elev_grid[tuple(np.moveaxis(pos2,-1,0))] - self.elev_grid[tuple(np.moveaxis(pos1,-1,0))]
        height_cost = (height_delta**2)/self.pixel_size
        return (1-self.e_lambda)*dist_cost + self.e_lambda*height_cost

    def get_children(self, idxs):
        idx_fringe = self.fringe[idxs]
        n_child = self.dir_grid.shape[0]
        parent_poss = np.repeat(idx_fringe.poss[:,np.newaxis,:], n_child, axis=1)
        child_poss = parent_poss + self.dir_grid
        parent_poss = parent_poss.reshape(-1,2)
        child_poss = child_poss.reshape(-1,2)
        
        grid_dims = self.elev_grid.shape
        in_grid_mask = (child_poss[:,0] < grid_dims[0]) & (child_poss[:,1] < grid_dims[1]) & (child_poss>=0).all(axis=-1)
        child_poss = child_poss[in_grid_mask]
        parent_poss = parent_poss[in_grid_mask]

        gs = np.repeat(idx_fringe.gs, n_child)[in_grid_mask]
        fs = gs + self.movement_cost(child_poss, self.end)
        child_fringe = Fringe(full_copy=(child_poss, gs, fs, parent_poss))
        return child_fringe
    
    def search(self, start_y, end_y):
        self.start = np.array((0,start_y))
        self.end = np.array((self.elev_grid.shape[0]-1, end_y))
        root = (self.start, 0, 0, np.array([-1, -1])) # pos, g, f, parent pos
        self.node_cache[tuple(root[0])] = root
        self.fringe[0] = root
        now_len = 1
        later_len = now_len

        f_limit = float('inf')
        goal_node = None
        while goal_node is None:
            print(len(self.node_cache.keys()))
            usable_mask = self.fringe.fs[:now_len] <= f_limit
            child_fringe = self.get_children(np.where(usable_mask))

            # preexisting + goal_node check
            for k in range(len(child_fringe)):
                if (child_fringe.poss[k] == self.end).all():
                    goal_node = child_fringe.get(k)
                    break
                preexisting = self.node_cache.get(tuple(child_fringe.poss[k]))
                if preexisting is None:
                    self.node_cache[tuple(child_fringe.poss[k])] = child_fringe.get(k)
                elif preexisting is not None and preexisting[1] < child_fringe.gs[k]:
                    child_fringe.fs[k] = -1
            child_fringe = child_fringe[child_fringe.fs >= 0]
            
            unchecked_fringe = self.fringe[np.where(~usable_mask)]
            self.fringe[:len(unchecked_fringe)] = unchecked_fringe.get(slice(0,len(unchecked_fringe)))
            self.fringe[len(unchecked_fringe):len(unchecked_fringe)+len(child_fringe)] = (child_fringe.poss, child_fringe.gs, child_fringe.fs, child_fringe.parents)
            now_len = len(unchecked_fringe)+len(child_fringe)

            f_max = np.max(self.fringe.fs[:now_len])
            f_min = np.min(self.fringe.fs[:now_len])
            f_limit = f_min + (f_max - f_min)*.5
        
        path = []
        pos = goal_node[0]
        while np.sum(pos) >= 0:
            path.append(tuple(pos[::-1]))
            pos = self.node_cache[tuple(pos)][-1]
        
        seen = [tuple(node[0][::-1]) for node in self.node_cache.values()]

        return path, seen

# f_min = float('inf')
# now = self.fringe[:now_len]
# for node in self.fringe[:now_len]:
#     f = node.g + self.movement_cost(node.pos, self.end)
#     if f > f_limit:
#         f_min = min(f, f_min)
#         continue
#     if node.pos == self.end:
#         goal_node = node
#         break
#     for pos in self.get_neighbor_positions(node.pos):
#         child_node = Node(pos, node)
#         child_node.g += self.movement_cost(node.pos, pos)
#         if self.node_cache.get(pos) is not None:
#             pass
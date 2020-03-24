import numpy as np

class AStarSearch(object):
    # higher elev_lambda => more weight on elevation change
    def __init__(self, elev_grid, pixel_size=100, elev_lambda=.5, heuristic_lambda=.5, fwd_only=False):
        # converts grid to x = row, y = col
        self.elev_grid = elev_grid.transpose(1,0)
        self.pixel_size = pixel_size
        self.e_lambda = elev_lambda
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

    def movement_cost(self, old_pos, new_pos):
        height_delta = np.abs(self.elev_grid[new_pos] - self.elev_grid[old_pos])
        height_cost = height_delta/self.pixel_size
        return (1-self.e_lambda)*1 + self.e_lambda*height_cost
    
    def heuristic_cost(self, pos, end_pos):
        # dist_cost = np.sqrt( (end_pos[0]-pos[0])**2 + (end_pos[1]-pos[1])**2 )
        dist_cost_simple = np.abs(end_pos[0]-pos[0]) + np.abs(end_pos[1]-pos[1])
        # avg_edge_ht = np.sum(self.elev_grid[-1])/self.elev_grid.shape[1]
        # height_delta = np.abs(avg_edge_ht - self.elev_grid[pos])
        height_delta = np.abs(self.elev_grid[end_pos] - self.elev_grid[pos])
        height_cost =  height_delta/self.pixel_size
        return (1-self.e_lambda)*dist_cost_simple + self.e_lambda*height_cost

    def search(self, start_y, end_y):
        start = (0,start_y)
        end = (self.elev_grid.shape[0]-1, end_y)
        open_nodes = { start: {'g':0, 'f':0, 'parent':None} } # {position: (cost, previous_position)}
        closed_nodes = {}
        current_pos = start
        iters = 0
        while current_pos != end: # while not at right edge
            current_pos = min(open_nodes.keys(), key=lambda n:open_nodes[n]['f'])
            if iters%1000 == 0 and iters>0:
                print('DEBUG', iters, len(closed_nodes), max(closed_nodes.keys(), key=lambda n:n[0]))
            current_cost = open_nodes[current_pos]['g']
            closed_nodes[current_pos] = open_nodes.pop(current_pos)
            # print(current_pos, open_nodes)
            for pos in self.get_neighbor_positions(current_pos):
                pos_cost = current_cost+self.movement_cost(current_pos, pos)
                if pos in open_nodes.keys() and pos_cost < open_nodes[pos]['g']:
                    open_nodes.pop(pos)
                # if pos in closed_nodes.keys() and pos_cost < closed_nodes[pos]['g']:
                #     closed_nodes.pop(pos)
                if pos not in open_nodes.keys() and pos not in closed_nodes.keys():
                    net_cost = (1-self.h_lambda)*pos_cost + self.h_lambda*self.heuristic_cost(pos, end)
                    open_nodes[pos] = {'g':pos_cost, 'f':net_cost, 'parent':current_pos}
            iters += 1
        # get coordinates to draw path
        path = []
        while current_pos is not None:
            path.append(current_pos[::-1]) # convert (x,y) to (y,x) for rendering
            current_pos = closed_nodes[current_pos]['parent']
        print([closed_nodes[pos[::-1]] for pos in path])

        return path, [pos[::-1] for pos in closed_nodes.keys()]

    if __name__ == '__main__':
        search(0,None)
class MinPriorityQueue(object):
    def __init__(self, inp=[]):
        self.q = inp
    
    def add(self, x):
        self.q.append(x)
        x_idx = len(self.q)-1
        while True:
            parent_idx = (x_idx-1)//2
            if x_idx == 0 or self.q[parent_idx] <= x:
                return
            else:
                self.q[x_idx] = self.q[parent_idx]
                self.q[parent_idx] = x
                x_idx = parent_idx
    
    def pop_root(self):
        to_pop = self.q[0]
        if len(self.q) < 2:
            self.q = []
            return to_pop
        self.q[0] = self.q.pop(-1)
        self.q = min_heapify(self.q, 0)
        return to_pop
    
    # def __iter__(self):
    #     self.iterct = 0
    #     pass

    # def __next__(self):
    #     self.iterct += 1
    #     if self.iterct > len(self.q):
    #         raise StopIteration
    #     return self.q[self.iterct-1]

    def __getitem__(self, i):
        return self.q[i]
    
    def __repr__(self):
        return str(self.q)


def min_heapify(arr, i):
    small = i
    while 2*i+1 < len(arr):
        l = 2*i+1
        r = 2*i+2
        if l < (len(arr)-1) and (small == -1 or arr[l] < arr[small]):
            small = l
        if r < (len(arr)-1) and arr[r] < arr[small]:
            small = r
            
        if small != i:
            arr[i], arr[small] = arr[small], arr[i]
            i = small
        else:
            break
    return arr
        
# Heap
from math import log, ceil, floor
from collections import defaultdict, deque
from operator import le, lt, ge, gt
from copy import deepcopy as dcopy, copy

class Heap:

    NAME = 'Heap'

    def __init__(self, *things):
        self.data = deque()
        self.dict = {}
        self += things

    def __iadd__(self, iterable):
        for x in iterable:
            self.append(x)
        return self

    def copy(self):
        return copy(self)

    def deepcopy(self):
        return dcopy(self)

    def __len__(self):
        return self.data.__len__()

    def __iter__(self):
        new = self.copy()
        while new:
            yield new.popleft()
        return None

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        def wraping(a_deque):
            a_deque.appendleft(f"{self.NAME} {'v'*12}")
            a_deque.append('^'*20)
            return a_deque

        data = self.data
        n_layers = ceil( log( len(data)+1, 2 ) )

        if n_layers < 1:
            return '\n'.join(wraping(deque()))
        elif n_layers == 1:
            stack = deque([str(data[0])])
            stack = wraping(stack)
            return '\n'.join(stack)
        else:
            stack = deque()
            ele_per_layer = [2**x for x in range(n_layers)]
            # +2 account for space encompass each element
            span_per_ele = max(len(str(x)) for x in data) + 1
            # The real number of element for the last layer is in [-1]
            graph_width = ele_per_layer[-1] * (span_per_ele)
            start = 0
            for epl in ele_per_layer:
                end = start + epl
                ele_span = graph_width // epl
                # every element is center-aligned
                this_layer = ''.join( f"{str(x):^{ele_span}}" for x in list(data)[start:end] )
                stack.append(this_layer)
                start = end
            stack = wraping(stack)
            # right-align all layers
            return '\n'.join(f"{row:<{graph_width}}" for row in stack)

    def append(self, x):
        self.data.append(x)
        idx = self.data.__len__() - 1
        self.dict[x] = idx
        self.up(idx)

    def appendleft(self, x):
        self.data.appendleft(x)
        self.dict[x] = 0
        self.down([0])

    def pop(self):
        x = self.data.pop()
        del self.dict[x]
        # TODO: potential bug if there are multiple entries for the same key
        return x

    def popleft(self):
        # Must have at least one element to pop()
        data = self.data
        dic = self.dict
        if len(data) >= 1:
            data[0], data[-1] = data[-1], data[0]
            dic[data[0]] = 0
            del dic[data[-1]]
            out = data.pop()
            if len(data) > 0:
                self.down(0)
            return out
        else:
            raise IndexError("Heap is empty.")

    def up(self, child_i):
        data = self.data
        dic = self.dict
        compare = self.UP_COMPARE
        child_v = data[child_i]
        # If index i == 0, no need to do anthing.
        while child_i > 0:
            parent_i = (child_i-1)//2
            parent_v = data[parent_i]
            if compare(child_v, parent_v):
                data[child_i] = parent_v
                dic[parent_v] = child_i
                child_i = parent_i
            else:
                # Can no longer move upwards
                break
        data[child_i] = child_v
        dic[child_v] = child_i

    def down(self, parent_i):
        compare = self.DOWN_COMPARE
        data = self.data
        dic = self.dict
        end = len(data)-1
        if parent_i == end:
            return None
        if parent_i > end:
            raise IndexError(f"This index exceed the length: {parent_i}")
        parent_v = data[parent_i]
        child_i = parent_i
        while True:
            lchild_i = parent_i*2+1
            rchild_i = lchild_i+1
            # Child may not exist for one node, check before getting the value
            # If Right Exist, this means that Left exist.
            # Reversely if Left do not exist, there is no Right.
            if lchild_i <= end:
                lchild_v = data[lchild_i]
                # We do not sure to swap with L or R
                if rchild_i <= end:
                    rchild_v = data[rchild_i]
                    if compare(lchild_v, rchild_v) and compare(parent_v, rchild_v):
                        child_i = rchild_i
                        data[parent_i] = rchild_v
                        dic[rchild_v] = parent_i
                        parent_i = child_i
                        continue
                # Can only swap with L
                if compare(parent_v, lchild_v):
                    child_i = lchild_i
                    data[parent_i] = lchild_v
                    dic[lchild_v] = parent_i
                    parent_i = child_i
                    continue
            break
        data[child_i] = parent_v
        dic[parent_v] = child_i

    def update(self, old_v, new_v):
        dic = self.dict
        old_i = dic[old_v]
        del dic[old_v]
        dic[new_v] = old_i
        self.data[old_i] = new_v
        if self.DOWN_COMPARE(new_v, old_v):
            self.down(old_i)
        else:
            self.up(old_i)


class MaxHeap(Heap):

    NAME = "MaxHeap"
    UP_COMPARE = gt
    DOWN_COMPARE = lt


class MinHeap(Heap):
    
    NAME = "MinHeap"
    UP_COMPARE = lt
    DOWN_COMPARE = gt
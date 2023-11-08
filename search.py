from warnings import warn
from sys import stdin
from time import perf_counter as tt
from time import sleep 
from collections import deque, namedtuple, defaultdict
from bisect import bisect
from copy import copy
from math import sqrt
from itertools import repeat, chain

from heap import MinHeap, MaxHeap



class Search:
              
    def __init__(self, start_node, child_func, end_node, cost):
        self.start_node = start_node
        self.child_nodes = child_func
        self.end_node = end_node
        self.cost = cost
        
    def run(self):
        return True
    
    def timeit(self, start_node=None):
        """Benchmark the search performance. Wrap over self.run().
        """
        self.found = 0
        self.step = 0
        self.path = []
        t0 = tt()
        result = self.run(start_node)
        t1 = tt()
        self.timelog = timelog = t1-t0
        print(self)
        return result, timelog
        
    def __str__(self):
        return (f"Search time = {self.timelog} s\n"
              f"Step        = {self.step}\n"
              f"Found       = {self.found}\n"
              f"Path len    = {str(len(self.path)) + str(self.path) if len(self.path)<30 else len(self.path)}")


class IDSearch(Search):
        
    def timeit(self, start_node=None, limit=1_000_000):
        """Benchmark the search performance. Wrap over self.run().
        Iterative Depth Search
        """
        self.found = 0
        self.step = 0
        self.path = []
        cap = 0
        t0 = tt()
        while not self.found and cap < limit:
            result = self.run(start_node, cap=cap)
            cap += 4
        t1 = tt()
        self.timelog = timelog = t1-t0
        print(self)
        return result, timelog


class BreadthFirstSearch(Search):
   
    def run(self, start_node=None, depth=0, cap=1_000_000, step=0, found=0):
        memory = set()
        end_node = self.end_node
        childs = self.child_nodes
        path = [start_node or self.start_node]
        stack = set([start_node or self.start_node])
        t0 = tt()
        while not found and (-1<step<=cap or -1<depth<=cap) :
            if tt()-t0 > 3:
                print(f"Step, Depth  = {step}, {depth}")
                t0 = tt()
            diff = 0
            if end_node in stack:
                diff += 1
                stack = list(stack)
                path.append(end_node)
                found = 1
                break
            else:
                diff += len(stack)
                memory.update(stack)
                stack = set(chain.from_iterable(childs(x) for x in stack)) - memory
            if not diff:
                break
            step += diff
            depth += 1
        del stack
        self.step, self.path, self.found = self.step+step, path, bool(found)
        return True
    
class DepthFirstSearch(IDSearch):
    
    def run(self, start_node=None, depth=0, cap=1, step=0, found=0):
        childs = self.child_nodes
        memory = set()
        memorise = memory.add
        start_node, end_node = start_node or self.start_node, self.end_node
        path = [start_node or self.start_node]
        stack = defaultdict(deque)
        stack[depth] += [start_node or self.start_node]
        t0 = tt()
        while not found and (-1<depth<=cap):
            if tt()-t0 > 3:
                print(f"Step, Depth  = {step}, {depth}")
                t0 = tt()
            search_space = stack[depth]
            while len(search_space):
                if depth == cap:
                    if end_node in search_space:
                        step += len(search_space)
                        found = True
                        break
                    else:
                        path.pop()
                        depth -= 1
                        break
                node = search_space.popleft()
                if node in memory:
                    continue
                memorise(node)
                step += 1
                path.append(node)
                if node == end_node:
                    found = True
                    break
                depth += 1
                stack[depth] += list(set(childs(node)) - memory)
                break
            else:
                path.pop()
                depth -= 1
        del stack
        self.path, self.step, self.found = path, self.step+step, found
        return True
        
        
class BidirectionSearch(Search):
    
    def run(self, start_node=None, depth=0, cap=1_000_000, step=0, found=0):
        childs = self.child_nodes
        head, tail = start_node or self.start_node, self.end_node
        head_pathfinder, tail_pathfinder = {head:head}, {tail:tail}
        head_space, tail_space = set([head]), set([tail])
        head_memory, tail_memory = set([head]), set([tail])
        intersect = set()
        diff = 2
        t0 = tt()
        while not found and -1<step<=cap:
            if tt()-t0 > 3:
                print(f"Step, Depth  = {step}, {depth}")
                t0 = tt()
            intersect = head_space.intersection(tail_space)
            if intersect:
                found = True
                break
            heads = dict(chain.from_iterable(
                ((v,k) for v in childs(k) if v not in head_memory) 
                for k in head_space
                ))
            head_pathfinder.update(heads)
            heads = set(heads.keys())
            head_space.update(heads)
            
            tails = dict(chain.from_iterable(
                ((v,k) for v in childs(k) if v not in tail_memory) 
                for k in tail_space
                ))
            tail_pathfinder.update(tails)
            tails = set(tails.keys())
            tail_space.update(tails)
            
            head_memory.update(heads)
            tail_memory.update(tails)
            
            diff1, diff2 = len(tails), len(heads)
            step += diff1 + diff2
            if not diff1 or not diff2:
                break
            depth += 1
        step = len(head_space) + len(tail_space)
        del head_space
        del tail_space
        path = []
        if found:
            print(intersect)
            middle = list(intersect)[0]
            p1, p2 = middle, head_pathfinder[middle]
            path = deque([middle])
            while p1 != p2:
                path.appendleft(p2)
                p1, p2 = p2, head_pathfinder[p2]
            p1, p2 = middle, tail_pathfinder[middle]
            while p1 != p2:
                path.append(p2)
                p1, p2 = p2, tail_pathfinder[p2]
            path = list(path)
        self.path, self.step, self.found = path, self.step+step, bool(found)
        return True


class HeuristicSearch(Search):
    
    def run(self, start_node=None, depth=0, cap=1_000_000, step=0, found=0):
        memory = set()
        memorise = memory.add
        end_node = self.end_node
        childs = self.child_nodes
        path = [start_node or self.start_node]
        stack = MinHeap(start_node or self.start_node)
        t0 = tt()
        while not found and len(stack) and (-1<step<=cap or -1<depth<=cap) :
            if tt()-t0 > 3:
                print(f"Step, Depth  = {step}, {depth}")
                t0 = tt()
            step += 1
            node = stack.popleft()
            memorise(node)
            if node == end_node:
                found = 1
                path.append(node)
                break
            stack += set(childs(node)) - memory
            depth += 1
        del stack
        self.step, self.path, self.found = self.step+step, path, bool(found)
        return True

class ForeseeSearch(Search):
    
    def run(self, start_node=None, depth=0, cap=1_000_000, step=0, found=0, foresee_step=16):
        memory = set()
        memorise = memory.add
        end_node = self.end_node
        childs = self.child_nodes
        self.path = path = [start_node or self.start_node]
        stack = MinHeap(start_node or self.start_node)
        t0 = tt()
        while not found and len(stack) and (-1<step<=cap or -1<depth<=cap) :
            if tt()-t0 > 3:
                print(f"Step, Depth  = {step}, {depth}")
                t0 = tt()
            step += 1
            try:
                parent = stack.popleft()
            except IndexError:
                found = 0
                break
            memorise(parent)
            if parent == end_node:
                found = 1
                path.append(parent)
                break
            new_nodes = set(childs(parent)) - memory
            backtrack = {child:parent for child in new_nodes}
            working_memory = memory | new_nodes
            count = 1
            foreseeing = MinHeap()
            foreseeing += new_nodes
            while (not found) and new_nodes and count < foresee_step:
                all_new_childs = set()
                for new_node in new_nodes:
                    new_childs = set(childs(new_node)) - working_memory
                    working_memory |= new_childs
                    all_new_childs |= new_childs
                    backtrack.update({child:new_node for child in new_childs})
                    if end_node in new_childs:
                        found = 1
                        break
                new_nodes = all_new_childs
                foreseeing += all_new_childs
                count += 1
            # Walk to the node of lowest cost
            if len(foreseeing):
                destination = foreseeing.popleft()
                if destination <= parent: 
                    backpath = [destination]
                    _from, _to = backtrack[destination], destination
                    while _from != parent:
                        # Add nodes in the middle back into the stack
                        # Since we don't know if they will lead us to the end,
                        # we haven't explore them yet.
                        stack.append(_to)
                        backpath.append(_from)
                        _from, _to = backtrack[_from], _from
                    backpath = backpath[::-1]   
                    step += len(backpath)
                    path += backpath
            if found:
                break
            depth += count
            
        del stack
        self.step, self.path, self.found = self.step+step, path, bool(found)
        return True
        
class IDASearch(IDSearch):   
    
    def run(self, start_node=None, depth=0, cap=1_000_000, step=0, found=0):
        childs = self.child_nodes
        memory = set()
        memorise = memory.add
        start_node, end_node = start_node or self.start_node, self.end_node
        path = [start_node or self.start_node]
        stack = defaultdict(MinHeap)
        stack[depth] += [start_node or self.start_node]
        t0 = tt()
        while not found and (-1<depth<=cap):
            if tt()-t0 > 3:
                print(f"Step, Depth  = {step}, {depth}")
                t0 = tt()
            search_space = stack[depth]
            while len(search_space):
                if depth == cap:
                    if end_node in search_space:
                        step += len(search_space)
                        found = True
                        break
                    else:
                        path.pop()
                        depth -= 1
                        break
                node = search_space.popleft()
                if node in memory:
                    continue
                memorise(node)
                step += 1
                path.append(node)
                if node == end_node:
                    found = True
                    break
                depth += 1
                stack[depth] += set(childs(node)) - memory
                break
            else:
                path.pop()
                depth -= 1
        del stack
        self.path, self.step, self.found = path, self.step+step, found
        return True
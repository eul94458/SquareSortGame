from warnings import warn
from sys import stdin
from time import perf_counter as tt
from time import sleep 
from collections import deque, namedtuple, defaultdict
from bisect import bisect
from copy import copy
from math import sqrt
from itertools import repeat


from search import *



class State(tuple):
    
    def __new__(self, value, state):
        return tuple.__new__(State, (value, state))
        
    @property
    def value(self):
        return self[0]
    
    @property
    def state(self):
        return self[1]
    
    def __len__(self):
        return len(self[1])
    
    def __hash__(self):
        return hash(tuple(self[1]))
    
    def __eq__(self, x):
        return self[0] == x[0]
    
    def __repr__(self):
        return "State {a}: {b}".format(a=self[0], b=str(self[1]))
    
    def __str__(self):
        return "{a}: {b}".format(a=self[0], b=self[1])
    
    def __lt__(self, x):
        return self[0] <  x[0]

    def __le__(self, x):
        return self[0] <= x[0]

    def __gt__(self, x):
        return self[0] >  x[0]

    def __ge__(self, x):
        return self[0] >= x[0]
    
    def __format__(self, f):
        return f"{str(self):{f}}"

    def index(self, x):
        return self[1].split(' ').index(x)
        
    def split(self, x):
        return self[1].split(x)



class SquareSortGame:
    
    search_methods = {
        'bfs': BreadthFirstSearch, 
        'dfs': DepthFirstSearch, 
        'bid': BidirectionSearch, 
        'ham': HeuristicSearch,
        'man': HeuristicSearch,
        'hyb': HeuristicSearch,
        'idaham': IDASearch,
        'idaman': IDASearch,
        'idahyb': IDASearch,
        'fore': ForeseeSearch,
        }

    def __init__(self, width):
        self.width = width
    
    def new(self, heuristic_name: str=None):
        self.stack = []
        self.chosen_heuristic = self.heuristic_methods.get(heuristic_name, None)
        self.end_state = self.get_end_state()
        self.current_state = self.rand_state()
    
    @property
    def current_state(self):
        return self.stack[-1]
    
    @current_state.setter
    def current_state(self, state):
        self.stack.append(state)
        self.checked_next = False
        self.what_next()
    
    def make_state(self, state):
        transform = self.chosen_heuristic
        return ( state if transform is None and isinstance(state, str) else 
                 state.state if transform is None and isinstance(state, State) else
                 State(transform(state.state),  state.state) if isinstance(state, State) else
                 State(transform(state), state)
                )
    
    def get_end_state(self, width=None):
        w = width or self.width
        s = ' '.join(map( str, tuple(range(1,w**2))+(0,) ))
        if self.chosen_heuristic is None:
            return s
        else:
            return State(0, s)
       
    def rand_state(self):
        from random import shuffle
        is_solvable = self.is_solvable
        result = ( self.end_state.state.split(' ') 
                   if isinstance(self.end_state, State) else 
                   self.end_state.split(' ')  )
        shuffle(result)
        while not is_solvable(result):
            shuffle(result)
        s = ' '.join(map(str,result))
        return self.make_state(s)

    @staticmethod
    def TimBabych(state):
        if isinstance(state, str):
            state = state.split(' ')
        s = [x for x in map(int, state) if x]
        parity = 0
        sorted_left = []
        for i, u in enumerate(s):
            # i is also the length of sorted_left
            insert_pt = bisect(sorted_left, u)
            parity += i - insert_pt
            sorted_left.insert(insert_pt, u)
        return parity
    
    @staticmethod
    def is_solvable(state):
        s = [x for x in map(int, state) if x]
        parity = 0
        sorted_left = []
        for i, u in enumerate(s):
            # i is also the length of sorted_left
            insert_pt = bisect(sorted_left, u)
            parity ^= (i - insert_pt)%2
            sorted_left.insert(insert_pt, u)
        return parity==0
    
    @staticmethod
    def draw(state):
        s = state.state if isinstance(state, State) else state
        s = s.split(' ')
        w = int(sqrt(len(s)))
        maxlen = len(str(len(s)))
        board = ( (x if x else ' ' for x in y) 
                  for y in zip(* repeat(iter(s), w) ) )
        board = ( ' '.join(f"{' '*(maxlen-len(x))}{x}" for x in row)
                  for row in board )
        result = '\n'.join(board)
        return result
    
    @staticmethod
    def print(state):
        print(SquareSortGame.draw(state), end='\n\n')
    
    def now(self):
        print("*** now ***")
        self.print(self.current_state)
    
    def is_result(self, state) -> bool:
        return state == self.end_state
    
    def check(self) -> bool:
        return self.current_state == self.end_state
    
    def hamming_distance(self, state, end_state=None):
        me, emeny = state, end_state or self.end_state
        me, emeny = map(int, me.split(' ')), map(int, emeny.split(' '))
        return sum( x!=y for x,y in zip(me, emeny) )
    
    def manhattan_distance(self, state, end_state=None):
        me, emeny = state, end_state or self.end_state
        me, emeny = map(int, me.split(' ')), map(int, emeny.split(' '))
        w = self.width
        return sum( abs(a//w-b//w)+abs(a%w-b%w) for a,b in zip(me, emeny) )
    
    def hybrid_distance(self, state, end_state=None):
        return ( self.hamming_distance(state, end_state) 
                +self.manhattan_distance(state, end_state)
                )
    
    @property
    def heuristic_methods(self):
        return {
            'ham': self.hamming_distance,
            'man': self.manhattan_distance,
            'hyb': self.hybrid_distance,
            'idaham': self.hamming_distance,
            'idaman': self.manhattan_distance,
            'idahyb': self.hybrid_distance,
            'fore': self.hybrid_distance,
            }
        
    def possible_move(self, state=None, debug=0):
        state = state or self.current_state
        state = state if self.chosen_heuristic is None else state.state
        state = state.split(' ')
        w = self.width
        zero_at = state.index('0')
        if debug: print(f"zero_at = {zero_at}")
        
        def _where(width, linear_position):
            lp = linear_position
            i, j = lp//width, lp%width
            if debug: print(f"i,j = {i}, {j}")
            return i, j
            
        i, j = _where(w, zero_at)
        
        def _neighbors(state, width, linear_position, i, j):
            from operator import itemgetter
            
            lp = linear_position
            targets = set()
            push = targets.add
            if i != 0: push('up')
            if i != width-1: push('down')
            if j != 0: push('left')
            if j != width-1: push('right')
            if debug: print(f"targets = {targets}")
            
            doit = {'up'   : lambda k: k-width,
                    'down' : lambda k: k+width,
                    'left' : lambda k: k-1,
                    'right': lambda k: k+1, 
                    }
            
            linear_positions = [ doit[t](lp) for t in targets ]
            if debug: print(f"_neighbors at {linear_positions}")
            return linear_positions
        
        if debug: print("possible moves")
        new_states = dict()
        for x in _neighbors(state, w, zero_at, i, j):
            temp = list(state[:])
            temp[zero_at], temp[x] = state[x], '0'
            new_states[state[x]] = self.make_state(' '.join(temp))
        return new_states
    
    def show(self):
        # show possible move
        result = ["=== possible moves ==="]
        push = result.append
        stack = self.next
        n_tol = len(stack)
        n_indent = len(str(n_tol)) + 3
        indent = ' ' * n_indent
        for i, x in stack.items():
            i =  "{i})".format(i=i)
            x = self.draw(x)
            temp = x.split('\n')
            temp[0] = i + ' '*(n_indent-len(i)) + temp[0]
            temp[1:] = [indent+row for row in temp[1:]]
            result += temp
            push('----------------------')
        print('\n'.join(result))
    
    def what_next(self):
        if not self.checked_next:
            self.checked_next = True
            self.next = self.possible_move()
        return self.next
        
    def move(self, i):
        if i in self.next:
            self.current_state = self.next[i]
            return True
        else:
            print("Invalid move.")
            return False
    
    def back(self):
        self.stack.pop()
        print("back to previous move")
        
    def restart(self):
        self.stack = self.stack[:1]
        self.end_state = self.make_state(self.end_state)
        self.stack = [self.make_state(x) for x in self.stack]
        print("### restarted the game ###")

    def play(self):
        _L1_option = {  'show': self.show ,
                        'back': self.back ,
                        'restart': self.restart ,
                        'new': self.new ,
                        'auto': self.auto_play }
        _L1_nextround = { 'back', 'restart', 'new' }
        _L1_end = {'auto'}
        keep_playing = True
        while keep_playing:
            # initiate game
            self.new()
            # make sure player do not win at the first place
            while self.check():
                self.new()
            bye = False
            # continously get input and move
            while not self.check() and not bye:
                self.now()
                nextround = False
                while not nextround:
                    inp = input("move(interger) | show | back | restart | new | auto | exit -> ")
                    inp = inp.strip().lower()
                    if inp == 'exit':
                        bye = True
                        break
                    elif inp in _L1_option:
                        _L1_option[inp]()
                        if inp in _L1_nextround:
                            nextround = True
                        if inp in _L1_end:
                            break
                    elif ( (not inp.isdigit()) or 
                         (inp.isdight() and int(inp) not in self.next) 
                         ):
                        print("Please input *interger* that match any of the possible moves.")
                        nextround = 1
                    if nextround:
                        continue
                    nextround = self.move(int(inp))
                    
            keep_playing = False
            if bye:
                return True
            print("+++ You win! Play again? +++")
            if input('(Y/n)').strip().lower() in ('y', 'yes'):
                keep_playing = True
                continue
            return True
            
    def auto_play(self, n=None):
        search_methods = self.search_methods.keys()
        query = ' | '.join(search_methods)

        while True:
            inp = input(f"{query} | new | exit -> ").strip().lower()
            if inp == 'exit':
                return None
            if inp == 'new':
                self.new()
                continue
            if inp not in search_methods:
                print(f"Please input *correct* method name.")
                continue
            
            search_method = self.search_methods[inp]
            self.chosen_heuristic = self.heuristic_methods.get(inp, None)
            if search_method is None:
                print(f"search_method for {inp} do not exist. Try again.")
                continue
            
            self.restart()
            solving = search_method(
                start_node=self.current_state,
                end_node= self.end_state,
                child_func=lambda state: self.possible_move(state).values(),
                cost=lambda x: 1
                )                
            solving.timeit()
            
        return None
    
    @staticmethod
    def auto(n):
        game = SquareSortGame(n)
        game.new(n)
        game.auto_play()
        

if __name__ == '__main__':
    title = """
    ######################################################################
      #####        ###       ##     ##      ###      #######     #########
    ##     ##    ##   ##     ##     ##    ##   ##    ##     ##   ##       
    ##          ##     ##    ##     ##   ##     ##   ##     ##   ##       
      #####     ## ##  ##    ##     ##   #########   #######     #########
           ##   ##  ## ##    ##     ##   ##     ##   ##    ##    ##       
    ##     ##    ##   ##     ##     ##   ##     ##   ##     ##   ##       
      #####        ###  ##     #####     ##     ##   ##     ##   #########
    ######################################################################
    """
    print(title)
    done = False
    while not done:
        instruc = input("What size? (integer) | Nauto | exit -> ").strip().lower()
        if instruc == 'exit':
            break
        if instruc.endswith('auto'):
            n = instruc[:-4]
            if n.isdigit():
                n = int(n)
                if n > 1:
                    SquareSortGame.auto(n)
                    continue
        if not instruc.isdigit():
            print("Please input *interger*.")
            continue
        size = int(instruc)
        if size <= 1:
            print("Please input interger *larger than 1*.")
            continue
        the_game = SquareSortGame(size)
        bye = the_game.play()
        if bye:
            break

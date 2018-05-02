import os, os.path
import logging

logger = logging.getLogger(os.path.basename(__name__))

def logging_level():
    return logger.getEffectiveLevel()

# location history is a list of pairs:
# [(c1,w1),(c2,w2),...,(cn,wn)]
#   ci = location
#   wi = # of days in that location

# residence history is a list of pairs:
# [(c1,i1),(c2,i2),...]
#   cj = location
#   ij = index in the location history when the person made that place their residence


valid_context_attrs = ['memA','i0','H','W','L']

# X is context: memA,i0,H,W,L,rho
class Context:
    pass

####
# Setup functions
def load_history(fname):
    fh = open(fname)

    H = []
    for line in fh:
        if line.startswith('#') or len(line.strip()) == 0:
            continue

        loc,duration = line.strip().split('\t')

        H.append((loc,int(duration)))

    return H

def compute_W(H):
    W = []
    acc = 0
    for i in range(len(H)):
        acc += H[i][1]
        W.append(acc)

    return W

def compute_i0(W,rho):
    for i in range(len(W)):
        if sum(W[:(i+1)]) >= rho:
            return i

    # couldn't find an i0!  The history isn't long enough
    return None

def compute_iF(W,i0,rho):
    for i in range(i0+1,len(W)):
        if sum(W[i:]) < rho:
            return i

    return i0

def compute_L(H):
    return set([h[0] for h in H])

####
# Core algorithm

def initialize_A(H,L,i0):
    memA = {}

    for l in L:
        away_time = 0
        for i in range(0,i0+1):
            if H[i][0] != l:
                away_time += H[i][1]

        memA[(i0,l)] = (away_time,[])

    return memA

def compute_Q(i,X):

    for j in range(X.i0+1,i):
        if sum(X.W[j:i]) <= X.rho:
            return j-1

    # if we couldn't find an index, start with i0
    return X.i0

def compute_A(i,l,X):

    if (i,l) in X.memA:
        return X.memA[(i,l)][0]
     
    Q = compute_Q(i,X)
    
    min_prev_rezes = None
    min_total_cost = float('infinity')
    for j in range(X.i0,Q+1):
        for loc in X.L:
            # compute time prior to living here
            prev_rez_cost = compute_A(j,loc,X)

            # compute time away while living here
            cur_rez_cost = 0
            for k in range(j+1,i+1):
                if X.H[k][0] != l:
                    cur_rez_cost += X.H[k][1]

            total_cost = prev_rez_cost + cur_rez_cost

            # keep track of the min total cost
            if total_cost < min_total_cost:
                min_total_cost = total_cost
                min_prev_rezes = [(j,loc)]
            elif total_cost == min_total_cost:
                min_prev_rezes.append((j,loc))

    # memoize
    X.memA[(i,l)] = (min_total_cost,min_prev_rezes)

    return min_total_cost

def compute_trailing_cost(l,X):
    """
    How much time did we spent away from l between iF and n?
    """
    acc = 0
    for i in range(X.iF+1,X.n+1):
        if X.H[i][0] != l:
            acc += X.H[i][1]

    return acc

def reconstruct_residence_history(X):

    ####
    # run the algorithm
    min_loc = None
    min_away = float('infinity')

    # compute the cost of residence ending at location l at time iF
    if logging_level() == logging.DEBUG: print('Total away costs:')
    for l in X.L:
        away_time = compute_A(X.iF,l,X) + compute_trailing_cost(l,X)
    
        if logging_level() == logging.DEBUG: print('\t%s = %d' % (l,away_time))
        if away_time < min_away:
            min_loc = l
            min_away = away_time

    if logging_level() == logging.DEBUG: print()

    ####
    # build the move history
    move_history = [(X.iF,min_loc)]
    while True:
        cur_rez = move_history[0]

        if cur_rez[0] == X.i0:
            break
        else:
            cur_rez = X.memA[cur_rez][1][0]
            move_history.insert(0,cur_rez)

    ####
    # make the residence history from the move history
    R = []
    next_move_idx = 0 
    while len(move_history) > 0:
        move = move_history.pop(0)

        R.append((move[1],next_move_idx))
        next_move_idx = move[0]+1

    return R

####
# For visualization & debugging
def print_move_history_tree(i,l,X,indent=0):

    cur_rez = (i,l)
    print('\t'*indent,cur_rez)

    if i == X.i0:
        return 
    else:
        for last_rez in X.memA[cur_rez][1]:
            print_move_history_tree(*last_rez,X,indent+1)

####
# Main for testing this
def main():
    import sys

    logging.basicConfig(level=logging.DEBUG)

    # read in residence min time
    rho = int(sys.argv[1])

    # load history
    H = load_history(sys.argv[2]) 

    ####
    # build context
    L = compute_L(H)
    W = compute_W(H)
    i0 = compute_i0(W,rho)
    iF = compute_iF(W,i0,rho)

    if logging_level() == logging.DEBUG:
        print('Context info:')
        print('\tW =',W)
        print('\ti0 =',i0)
    
    memA = initialize_A(H,L,i0)

    if logging_level() == logging.DEBUG: print()

    X = Context()
    X.memA = memA
    X.i0 = i0
    X.iF = iF
    X.H = H
    X.W = W
    X.L = L
    X.rho = rho
    X.n = len(H)-1

    rez_history = reconstruct_residence_history(X)

    print('Residence history:',rez_history)

if __name__ == '__main__':
    main()

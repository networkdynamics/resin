import os, os.path
import logging
import numpy as np
import csv
"""
@author = Derek Ruths <derek@derekruths.com>

Overview
--------
This implements the rezin algorithm for residence history inference.

In this library, location history is a list of pairs:
 [(c1,w1),(c2,w2),...,(cn,wn)]
   ci = location
   wi = # of days in that location

Residence history is a list of pairs:
 [(c1,i1),(c2,i2),...]
   cj = location
   ij = index in the location history when the person made that place their residence

To use the library
------------------

*Running the algorithm.* Call the rezin(...) function to run the full algorithm
from start to finish and get the residence history for a specific location history.

*Loading a location history.* Call the load_location_history(...) function.  It
assumes a tab-separated file in which first column is location name, second
column is the time of move (must be an integer).  See the .rez files in the
tests/ directory.

To run this file
----------------

You can run the algorithm by using this script directly:

    python rezin.py <min_rez_time> <location_history_file>

It will print a bunch of stuff with the residence history at the bottom.
"""

logger = logging.getLogger(os.path.basename(__name__))

def logging_level():
    return logger.getEffectiveLevel()


VALID_CONTEXT_ATTRS = ['memA','i0','H','W','L']
class Context:
    pass

####
# Setup functions
def load_location_history(fname):
    fh = open(fname)

    H = []
    for line in fh:
        if line.startswith('#') or len(line.strip()) == 0:
            continue

        loc,duration = line.strip().split('\t')

        H.append((loc,int(duration)))

    return H

def compute_W(H):
    return [h[1] for h in H]

def compute_i0(W,rho):
    for i in range(len(W)):
        if sum(W[:(i+1)]) >= rho:
            return i

    # couldn't find an i0!  The history isn't long enough
    return None

def compute_iS(W,rho):
    for i in range(1,len(W)):
        if sum(W[1:(i+1)]) >= rho:
            return i

    # couldn't find an iS!  The history isn't long enough - just use the base case.
    return None

def compute_iF(W,i0,rho):
    for i in range(i0+1,len(W)):
        if sum(W[i:]) < rho:
            return i-1

    return len(W)-1

def compute_L(H):
    return set([h[0] for h in H])

####
# Core algorithm

def initialize_A(H,L,i0,iS):
    memA = {}

    # initialize the A(i0,l)
    for l in L:
        away_time = 0
        for i in range(0,i0+1):
            if H[i][0] != l:
                away_time += H[i][1]

        memA[(i0,l)] = (away_time,[])

    # initialize the A(i,l) for i0 < i < iS
    for l in L:
        for i in range(i0+1,iS):
            away_time = memA[(i0,l)][0]
            for j in range(i0+1,i+1):
                if H[j][0] != l:
                    away_time += H[j][1]

            memA[(i,l)] = (away_time,[])

    # done
    return memA

def compute_Q(i,X):

    Q = []
    for j in range(X.i0,i):
        if sum(X.W[(j+1):(i+1)]) >= X.rho:
            Q.append(j)

    # if we couldn't find an index, there's an error
    if len(Q) == 0:
        raise Exception('No valid Q(%d) could be computed - i_s was incorrect or not checked?' % i)
    else:
        return Q

def compute_A(i,l,X,debug=False):

    if (i,l) in X.memA:
        return X.memA[(i,l)][0]
     
    if debug: print('\tA(%d,%s)' % (i,l))

    Q = compute_Q(i,X)
    
    if debug: print('\tQ=%s' % Q)
    
    min_prev_rezes = None
    min_total_cost = float('infinity')
    for j in Q: #range(X.i0,Q+1):
        for loc in X.L:

            # compute time prior to living here
            prev_rez_cost = compute_A(j,loc,X)

            # compute time away while living here
            cur_rez_cost = 0
            for k in range(j+1,i+1):
                if debug: print('k = %d' % k)
                if X.H[k][0] != l:
                    if debug: print('%s != %s' % (X.H[k][0],l))
                    cur_rez_cost += X.H[k][1]

            total_cost = prev_rez_cost + cur_rez_cost

            if debug: print('\t(%d,%s) mid (%d,%s): total_cost %d = %d + %d' % (i,l,j,loc,total_cost,prev_rez_cost,cur_rez_cost))

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

def reconstruct_residence_history(X,debug=False):

    ####
    # run the algorithm
    min_loc = None
    min_away = float('infinity')

    # compute the cost of residence ending at location l at time iF
    if debug:
        logging.debug('Total away time:')

    for l in X.L:
        dyn_away_time = compute_A(X.iF,l,X,debug=debug)
        trailing_away_time = compute_trailing_cost(l,X)
        away_time = dyn_away_time + trailing_away_time 
        if debug:
            print('\t%s\t%d=%d+%d' % (l,away_time,dyn_away_time,trailing_away_time))
    
        if debug:
            print_move_history_tree(X.iF,l,X,2)

        if away_time < min_away:
            min_loc = l
            min_away = away_time

    ####
    # build the move history
    move_history = [(X.iF,min_loc)]
    while True:
        #print('Move history part - ',move_history[0])
        cur_rez = move_history[0]

        if cur_rez[0] < X.iS:
            break
        else:
            cur_rez = X.memA[cur_rez][1][0]
            move_history.insert(0,cur_rez)

    #print('Move history: ',move_history)

    ####
    # make the residence history from the move history
    R = []
    next_move_idx = 0 
    while len(move_history) > 0:
        move = move_history.pop(0)

        if len(R) > 0 and R[-1][0] == move[1]:
            # detect staying in the same place ... that's not moving!
            pass
        else:
            R.append((move[1],next_move_idx))

        next_move_idx = move[0]+1

    return R

def rezin(H,rho,debug=False):
    """
    Function for running the entire algorithm on
        H the location history
        rho the minimum residence time

    Returns a residence history (as described above).
    """
    ####
    # build context
    L = compute_L(H)
    W = compute_W(H)
    i0 = compute_i0(W,rho)
    iS = compute_iS(W,rho)
    iF = compute_iF(W,i0,rho)

    X = Context()
    X.memA = initialize_A(H,L,i0,iS) # for memoizing the results of the dynamic algorithm
    X.i0 = i0 # the index in the location history where we initialize the A's
    X.iS = iS # the index where we begin computing the first A's (beyond those initialized)
    X.iF = iF # the last index in the location history our dynamic algorithm should evaluate
    X.H = H # the location history
    X.W = W # the stay times for each position in the location history - for efficiency
    X.L = L # the set of locations in the location history
    X.rho = rho # the min residence time interval
    X.n = len(H)-1 # the last index in the location history

    logger.debug('Context')
    logger.debug('\ti0 ' + str(i0))
    logger.debug('\tiS ' + str(iS))
    logger.debug('\tiF ' + str(iF))
    logger.debug('\tH ' + str(H))
    logger.debug('\tW ' + str(W))
    logger.debug('\tL ' + str(L))
    logger.debug('')

    # run the algorithm
    rez_history = reconstruct_residence_history(X,debug=debug)

    return rez_history

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

def print_interleaved_histories(R,H):
    localH = list(H)
    localR = list(R)

    idx = 0
    while len(localH) > 0:
        if len(localR) > 0 and localR[0][1] <= idx:
            print(' => %s' % localR.pop(0)[0])
        else:
            print('%s\t%d' % localH.pop(0))
            idx += 1

    return

def caitrin_read_history(inFile):
    user_dict = {}

    with open(inFile, "rb") as csvFile:
        reader = csv.reader(csvFile)
        for line in reader:
            if line[0]:
                user = line[0]
            else:
                continue
            if user not in user_dict:
                user_dict[user] = list()
            country = line[3]
            if len(country) == 3 and country.isalpha(): #funky file
                datetime_object = np.datetime64(line[2])
                user_dict[user].append((datetime_object, country)) #(time, country)
    print "finished reading file"

    user_histories = []
    for user in sorted(user_dict):
        user_dates = sorted(user_dict[user], key=lambda x: x[0])
        time_warped_history = []

        prev_loc = user_dates[0][1]
        prev_date = user_dates[0][0]
        cur_sum = 0

        for d in user_dates[1:]:
            if d[1] == prev_loc:
                cur_sum += (d[0] - prev_date).item().days
                prev_date = d[0]
            else:
                if cur_sum == 0:
                    cur_sum = 1
                lh = (prev_loc, cur_sum)
                time_warped_history.append(lh)
                prev_loc = d[1]
                cur_sum = 0
        if cur_sum != 0:
            lh = (prev_loc, cur_sum)
            time_warped_history.append(lh)

        user_histories.append(time_warped_history)
    return user_histories
####
# Main for testing this
def main():
    """
    For running the algorithm on a location history.  Cmd-line arguments:
        - minimum residence history
        - location history file
    """
    import sys

    logging.basicConfig(level=logging.DEBUG)

    # read in residence min time

    #rho = int(sys.argv[1])

    # load history
    #H = load_location_history(sys.argv[2])

    inFile = sys.argv[1]
    user_hists = caitrin_read_history(inFile)
    rho = 120

    for H in user_hists:

        if len(H) > 100: #quick check for bots
            continue
        print "-----------------------"
        for e in H:
            print "%s\t%d" %(e[0], e[1])
        try:
            rez_history = rezin(H,rho)
        except Exception as e:
            print e.message, e.args
            continue

        print('Residence history:',rez_history)

        print('\nInterleaved histories:')
        print_interleaved_histories(rez_history,H)


    #return

if __name__ == '__main__':
    main()

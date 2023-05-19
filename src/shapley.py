from itertools import compress, product

def combinations(items):
    return ( set(compress(items,mask)) for mask in product(*[[0,1]]*len(items)) )

import math
def coalition_wt(N,S):
    num=math.factorial(S)*math.factorial(N-S-1)
    den=math.factorial(N)
    return num/den
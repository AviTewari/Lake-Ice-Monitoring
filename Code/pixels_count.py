from modules import *
from masks import *
from mask_mapping import *

def bincount_2d(arr, max_int):
    counts_full = [0 for n in range(max_int)]
    for row in arr:
        counts = np.bincount(row).tolist()#get the counts for the row
        pad = [0 for n in range(max_int-len(counts))]
        counts = counts + pad #add extra zeroes to account for colors above the max in the row
        counts_full = [counts_full[n] + counts[n] for n in range(max_int)]
    return(counts_full)
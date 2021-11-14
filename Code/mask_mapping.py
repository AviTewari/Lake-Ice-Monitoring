from modules import *
from masks import *

def map_mask(mask, lib):
    new_mask = mask.copy()
    for key, val in lib.items():    #map the elements of the array to their new values according to the library
        new_mask[mask==key]=val
    return new_mask
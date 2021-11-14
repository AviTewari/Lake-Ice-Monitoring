from modules import *
from mask_preprocessing import *

mask_df['conc_minor']=mask_df[['conc_1', #lists the total concentration of under-represented ice classes
                               'conc_2', 
                               'conc_3', 
                               'conc_4', 
                              ]].sum(axis=1)

n_pixels = mask_df.iloc[0, 6:].sum(axis=0)#total number of pixels in each image
over_sample_names = mask_df[mask_df['conc_minor']/n_pixels>0.3] #we will over-sample these images of the under-represented classes
over_sample_names = over_sample_names['name'].values.tolist()
from modules import *
from mask_mapping import *
from pixels_count import *


mask_dir = 'Data/Masks'
mask_ext = '-mask.png'
new_mask_ext = '-mask-mod.png'
dat = [] #list that will hold information on the masks

for img_name in img_names:
    name = mask_dir + img_name
    # importing the image  
    mask = Image.open(name + mask_ext)

    # converting mask
    mask = np.array(mask)#convert to numpy
    new_mask = map_mask(mask, mask_lib)#map values
    
    #update dataframe
    name = img_name.split('-')  
    d = [img_name, 
         name[0][1:],  #patch id
         name[1][0:4], #year
         name[1][4:6], #month
         name[1][6:8], #day
         name[1][8:10]]#hour
    
    counts = bincount_2d(new_mask, n_colors) #values counts of the class of ice over all pixels in the image
    d.extend(counts)
    dat.append(d)
    
    # exporting the image 
    new_mask = Image.fromarray(new_mask)#convert back to image
    new_mask.save('./' + img_name + new_mask_ext, 'PNG') 

mask_dir = './'#update mask directory and extension
mask_ext= new_mask_ext

#create dataframe of mask information
mask_df = pd.DataFrame(dat, columns = ['name', 'patch_id', 'year', 'month', 'day', 'hour', 
                            'conc_0', 'conc_1', 'conc_2', 'conc_3', 'conc_4', 'conc_5', 'conc_6',  
                            'conc_land'])

""" counts = mask_df.iloc[:,6:].sum()
norm = counts.sum()
probs = counts/norm*100

plt.figure(figsize=(8,5))
probs.plot(kind='bar')
plt.ylabel('Fraction of Pixel Values (%)')
plt.grid() """
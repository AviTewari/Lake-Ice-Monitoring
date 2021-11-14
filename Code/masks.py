from  modules import *

mask_lib = {55:0, #ice free
            1:0, #<1/10 (open water)
            2:0, #bergy water
            10:1, #1/10
            12:1, #1/10-2/10
            13:1, #1/10-3/10
            20:1, #2/20
            23:1, #2/20-3/10
            24:2, #2/20-4/10
            30:2, #...
            34:2,
            35:2,
            40:2,
            45:2,
            46:3,
            50:3,
            56:3,
            57:3,
            60:3,
            67:3,
            68:4, #...
            70:4, #7/10
            78:4, #7/10-8/10
            79:4, #7/10-9/10
            80:4, #8/10
            89:4, #8/10-9/10
            81:5, #8/10-10/10
            90:5, #9/10
            91:5, #9/10-10/10
            92:6, #10/10 - fast ice
            100:7, #land
            99:7, #unknown - there is nothing in this class for this dataset
           }

#define a colormap for the mask
n_colors=8
ice_colors = n_colors-1
jet = plt.get_cmap('jet', ice_colors)
newcolors = jet(np.linspace(0, 1, ice_colors))
black = np.array([[0, 0, 0, 1]])
white = np.array([[1, 1, 1, 1]])
newcolors = np.concatenate((newcolors, black), axis=0) #land will be black
cmap = ListedColormap(newcolors)
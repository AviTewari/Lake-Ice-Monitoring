# Neural Net Mapping of Hudson Bay Sea Ice

The Canadian Ice Service produces weekly regional sea ice charts for ship safety and environmental monitoring. In this project I use a convolutional neural network to automatically generate ice charts from satellite imagery. With increasing availability of satellite data, this network may be able to produce similar ice charts globally at higher detail.
<p float="left">
  <img src="/Images/pred1.png" width="800" /> 
</p>

-  Collected 3392 satellite images of Hudson Bay sea ice in the Canadian Arctic from 2016-1-1 to 2018-7-31
-  Generated sea ice concentrations masks for each image using Canadian Regional Ice Chart shapefiles
-  Trained a Convolutional Neural Network (U-Net) to generate sea ice charts from satellite images based on eight different classes (seven levels of ice concentration and land)
    -  Model Accuracy: 83%
    -  Model Mean IoU (intersection over union) score: 0.44
- Found a strong class imbalance favoring thick solid ice due to complete freezing in the winter months. Future work should focus on collecting more data during the spring months when ice is thawing and there is a greater variety in ice concentration.
- Future work could also take advantage of additional satellite wavelength collection bands beyond the visible spectrum.
- The dataset is available here: https://www.kaggle.com/alexandersylvester/arctic-sea-ice-image-masking
 

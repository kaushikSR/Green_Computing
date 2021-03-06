# Automatic Detection of PV arrays and extracting its features from Aerial Satellite Imagery

## Dataset Columns (few columns not used):
polygon_id -> Each PV array corresponds to a polygon <br>
centroid_latitude/centroid_longitude -> latitude and longitude of the centroid of the polygon <br>
centroid_latitude_pixels/centroid_longitude_pixels -> pixel co-ordinates of the centroid of the polygon <br>
city -> location of the PV array <br>
area_pixels -> Area given by Google Earth when you draw a polygon around the PV array <br>
image_name -> image containing the PV array <br>
nw_corner_of_image_latitude/nw_corner_of_image_longitude/se_corner_of_image_latitude/se_corner_of_image_longitude -> latitudes and longitudes of the corners of the image. <br>
resolution -> meters per pixel <br>

1) Supervised Learning <br>
The implementation can be broadly divided into the
following steps: <br>
i) Data Labelling <br>
ii) Data Munging <br>
iii) Image Segmentation <br>
iv) Feature Extraction <br>
v) Classification (Logistic Regression and Random Forest) <br>
vi) Regression <br>

2) Weakly supervised learning using CNN. <br>

For more details, look at Final_Report.pdf <br>





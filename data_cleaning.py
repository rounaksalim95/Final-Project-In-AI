"""
2017.4
Process for cleaning the data.
"""

from image_process import *

# Step 1 - Download images from google

# Note that the google_images_scraper might fail in some images, 
# we will have to make sure every image for a certain fruit in
# the folder is available.

# Foe the first step, you have to execute the file:
# google_images_scraper.py


# Step 2 - Convert images files to Gray Scale

# At the second step, we will have to convert the downloaded image
# to gray scale version, and save separately.

fruits = ['Watermelon']
start_index = 1
end_index = 100

# for fruit in fruits:
#     process_gray_scale('./pic/' + fruit + '/', start_index, end_index)
# print "Finished converting to gray scale..."


# Step 3 - Rescale images to 256*256

# At this step we will make further modifications to gray scale
# images by rescaling them to 256*256.

for fruit in fruits:
	process_rescale_s('./pic/' + fruit + '/', fruit, start_index, end_index)
print "Finished rescaling..."


# Step 4 - Save res

# At this step we will save pixel info into a csv file with labels.

# for fruit in fruits:
#     collect_pixel_info_clr('./Pictures/' + fruit + '/', start_index, end_index, fruit)
# print "Finished saving pixel info into the csv..."


# Step 5 - Read in the csv file

# training_data = read_list("pixel_info_" + str(fruits[0]) + ".csv")
# print training_data[0]
# print len(training_data)

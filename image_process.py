"""
2017.2
Functions for gathering features from images.
"""

from PIL import Image
import math
import csv
import os

# Colors
SRED = (247,82,88)
DRED = (176,21,42)
SGREEN = (140,170,50)
DGREEN = (70,100,18)
YELLOW = (230,180,20)
ORANGE = (255,150,40)
SPURPLE = (170,50,60)
DPURPLE = (150,60,90)
BLACK = (0,0,0)
WHITE = (255,255,255)

colors = []
colors.append(SRED)
colors.append(DRED)
colors.append(SGREEN)
colors.append(DGREEN)
colors.append(YELLOW)
colors.append(ORANGE)
colors.append(SPURPLE)
colors.append(DPURPLE)
colors.append(BLACK)
colors.append(WHITE)


def color_diff(rgb1, rgb2):
    """get the difference between a color with one of ten base colors"""
    diff = math.sqrt((rgb1[0]-rgb2[0])**2 + (rgb1[1]-rgb2[1])**2 + (rgb1[2]-rgb2[2])**2)
    return diff


def color_distribution(img_file, dist_collection, num_features):
    """get the color distribution of a single image and save in a list of list"""
    color_list = [0] * num_features
    img = Image.open(img_file)
    pixels = img.load()

    for i in range(img.size[0]):
        for j in range(img.size[1]):
            diff_list = list(map(lambda x: color_diff(x,pixels[i,j]), colors))
            min_index = diff_list.index(min(diff_list))
            color_list[min_index] += 1

    # convert to percentage
    sum_pixel = sum(color_list)
    color_list = map(lambda x: round(x*1.0/sum_pixel, 5), color_list)

    dist_collection.append(color_list)
    return dist_collection


def convert_gray_scale(img_file):
    """convert an image to the gray scale counterpart"""
    img = Image.open(img_file)
    pixels = img.load()

    for i in range(img.size[0]):
        for j in range(img.size[1]):
            gray_scale = int(0.299*pixels[i,j][0] + 0.587*pixels[i,j][1] + 0.114*pixels[i,j][2])
            pixels[i,j] = (gray_scale, gray_scale, gray_scale)

    outfile = img_file[:-4] + "g.jpg"
    img.save(outfile)


def resize_small(img_file):
    """rescale an image so that its longer side is smaller than 600 pixel"""
    img = Image.open(img_file)
    pixels = img.load()

    width = img.size[0]
    height = img.size[1]

    base = 256

    if width > base:
        scale_factor = base / float(width)
        new_height = int(float(scale_factor) * float(height))
        img = img.resize((base,new_height), Image.ANTIALIAS)
    elif height > base:
        scale_factor = base / float(width)
        new_width = int(float(scale_factor) * float(width))
        img = img.resize((new_width,base), Image.ANTIALIAS)
    else:
        pass
    
    outfile = img_file[:-4] + ".jpg"
    img.save(outfile)


def resize_small_256(img_file, name):
    """rescale an image to 256*256, disregarding its wh ratio"""
    img = Image.open(img_file)
    pixels = img.load()

    base = 256
    img = img.resize((base,base), Image.ANTIALIAS)
    
    outfile = img_file[:-4] + '_' + str(name.lower()) + ".jpg"
    img.save(outfile)


def resize_small_base(img_file, base, name):
    """rescale an image to 256*256, disregarding its wh ratio"""
    img = Image.open(img_file)
    pixels = img.load()

    img = img.resize((base,base), Image.ANTIALIAS)

    i = 1
    while img_file[-i] != '/':
        i += 1
    i = i-1

    j = 1
    while img_file[-j] != '.':
        j += 1
    
    outfile = img_file[:-i] + str(name.lower()) + str(img_file[-i:-j]) + "-s.jpg"
    img.save(outfile)


def gather_gs_info(img_file):
    """gather the gray scale image info by storing every first value of rgb values"""
    img = Image.open(img_file)
    pixels = img.load()
    gs_info = []

    for i in range(img.size[0]):
        for j in range(img.size[1]):
            gs_info.append(pixels[i,j][0])

    return gs_info


def gather_clr_info(img_file):
    """gather the gray scale image info by storing every first value of rgb values"""
    img = Image.open(img_file)
    pixels = img.load()
    gs_info = []

    for i in range(img.size[0]):
        for j in range(img.size[1]):
            gs_info.append(pixels[i,j][0])
            gs_info.append(pixels[i,j][1])
            gs_info.append(pixels[i,j][2])

    return gs_info


def write_list(csv_file,data_list):
    """write out a list of list into a csv file"""
    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.writer(csvfile, dialect='excel', quoting=csv.QUOTE_NONNUMERIC)
            for data in data_list:
                writer.writerow(data)
    except IOError as (errno, strerror):
            print("I/O error({0}): {1}".format(errno, strerror))    
    return


def read_list(csv_file):
    """read in a list of list from a csv file"""
    try:
        with open(csv_file) as csvfile:
            reader = csv.reader(csvfile, dialect='excel', quoting=csv.QUOTE_NONNUMERIC)
            datalist = []
            datalist = list(reader)
            return datalist
    except IOError as (errno, strerror):
            print("I/O error({0}): {1}".format(errno, strerror))    
    return


def collect_color_distribution(img_direct, file_start, file_end, label, num_features):
    """get the color distribution of all images in a directory and save the lol as csv"""
    dist_collection = []
    for x in xrange(file_start,file_end+1):
        dist_collection = color_distribution(img_direct+str(x)+".jpg", dist_collection, num_features)
        print "finished " + str(x)
    map(lambda x: x.append(label), dist_collection)
    write_list("color_distribution.csv",dist_collection)


def collect_pixel_info(img_direct, file_start, file_end, label):
    """get the list of pixels of all images in a directory and save the lol as csv"""
    pixel_info = []
    for x in xrange(file_start,file_end+1):
        pixel_info.append(gather_gs_info(img_direct+str(x)+"-256.jpg"))
        print "finished " + str(x)
    map(lambda x: x.append(label), pixel_info)
    write_list("pixel_info_" + str(label) + ".csv", pixel_info)


def collect_pixel_info_clr(img_direct, file_start, file_end, label):
    """get the list of pixels of all images in a directory and save the lol as csv"""
    pixel_info = []
    for x in xrange(file_start,file_end+1):
        pixel_info.append(gather_clr_info(img_direct+str(x)+"-256.jpg"))
        print "finished " + str(x)
    map(lambda x: x.append(label), pixel_info)
    write_list("pixel_info_clr_" + str(label) + ".csv", pixel_info)


def process_gray_scale(img_direct, name, file_start, file_end):
    """convert images to gray scale in a directory in batch"""
    for x in xrange(file_start,file_end+1):
        convert_gray_scale(img_direct+name+str(x)+"-s.jpg")
        # print "gray scale: " + str(x) + " finished..."


def process_rescale(img_direct, file_start, file_end):
    for x in xrange(file_start,file_end+1):
        resize_small(img_direct+str(x)+".jpg")


def process_rescale_256(img_direct, name, file_start, file_end):
    for x in xrange(file_start,file_end+1):
        resize_small_256(img_direct+str(x)+".jpg", name)


def process_rescale_s(img_direct, name, file_start, file_end):
    for x in xrange(file_start,file_end+1):
        resize_small_base(img_direct+str(x)+".jpg", 100, name)


# Utility Tools
def get_min_color(img_file):
    """get the mean color rgb values from a piece"""
    img = Image.open(img_file)
    pixels = img.load()
    r = 0
    g = 0
    b = 0

    for i in range(img.size[0]):
        for j in range(img.size[1]):
            r += pixels[i,j][0]
            g += pixels[i,j][1]
            b += pixels[i,j][2]

    r = r*1.0 / (img.size[0]*img.size[1])
    g = g*1.0 / (img.size[0]*img.size[1])
    b = b*1.0 / (img.size[0]*img.size[1])
    return r, g, b


# get mean RGB values for different color pieces
# print get_min_color('./img/01.jpg')

# Collect color distribution and add labels
# collect_color_distribution('./img/', 1, 20, "banana", 10)

# Read in instances in a list from a certain folder
# color_dist = read_list("color_distribution.csv")
# print color_dist

# convert images to gray scale in a directory in batch
# process_gray_scale('./img/', 1, 6)

# resize_small('./img/80.jpg')
# Created by Kevin Haslett in March of 2017

import os
from sklearn.cluster import KMeans
from scipy import spatial
from skimage import io, color, img_as_float
import numpy as np

#  Map of bayer filters for various sizes
thresholdMap = {
    2: [[0, 3],
        [2, 1]],
    3: [[0, 6, 4],
        [7, 5, 1],
        [3, 2, 8]],
    4: [[ 0, 12,  3, 15],
        [ 8,  4, 11,  7],
        [ 2, 14,  1, 13],
        [10,  6,  9,  5]],
    8: [[ 0, 48, 12, 60,  3, 51, 15, 63],
        [32, 16, 44, 28, 35, 19, 47, 31],
        [ 8, 56,  4, 52, 11, 59,  7, 55],
        [40, 24, 36, 20, 43, 27, 39, 23],
        [ 2, 50, 14, 62,  1, 49, 13, 61],
        [34, 18, 46, 30, 33, 17, 45, 29],
        [10, 58,  6, 54,  9, 57,  5, 53],
        [42, 26, 38, 22, 41, 25, 37, 21]],
}

# Finds the closest colour in the palette to the provided colour
# Uses a kd-tree query for O(log n) queries
def nearest(palette, colour):
    dist, i = palette.query(colour)
    return palette.data[i]

# Finds the two closest colours in the palette to the provided colour
# Also return the euclidiean distnaces from each colour
# Uses a kd-tree query for O(log n) queries
def nearestTwo(palette, colour):
    dist, i = palette.query(colour, 2)
    return palette.data[i[0]], dist[0], palette.data[i[1]], dist[1]

# Make a kd-tree palette from the provided list of colours
def makePalette(colours):
    print(colours)
    return spatial.KDTree(colours)

# Dynamically calculates and N-colour palette for the given image
# Uses the KMeans clustering algorithm to determine the best colours
# Returns a kd-tree palette with those colours
def findPalette(image, nColours):
     # fit all the coloyrs in
    data = image.ravel().reshape(-1, 3)
    kmeans = KMeans(n_clusters=nColours, random_state=0, tol=.01).fit(data)
    colours = kmeans.cluster_centers_

    return makePalette(colours)

# Save a palette into a file
# Produces an image with a row of 25x25 pixel soloid squares for each colour in the palette
def savePalette(colours, fname):
    panelSize = 25
    nColours = len(colours)
    width = panelSize * nColours
    height = panelSize
    paletteImage = np.zeros((height, width, 3), dtype=np.double)
    for index, colour in enumerate(colours):
        for i in range(panelSize):
            for j in range(panelSize):
                x = index * panelSize + i
                y = j
                paletteImage[y][x] = colour

    fname += '-palette.png'
    io.imsave(fname, paletteImage)
    print(fname + " saved!")

# Produce a black and white ordered dither of the image with the give Bayer Filter size
def bwOrderedDither(image, fname, size):
    threshold = thresholdMap[size] # Get the right threshold map for our size

    width, height, *rest = image.shape
    for x in range(width):
        for y in range(height):
            colour = image[x][y]
            bracketed = colour * (size**2) # Map the colour into the appropriate range

            # Finally, check if that bracketed colour should be white or black based on the
            # threshhold for the current pixel
            image[x][y]= 1 if bracketed > threshold[x % size][y % size] + 0.5 else 0

    # Append  a descriptive marking to the end of the file and return it
    return fname + '-bw-' + str(size) + 'x' + str(size) + '.png'

# Produces a black and white Floyd Steinberg dither of the provided image
def bwFloydSteinbergDither(image, fname):
    width, height = image.shape
    for y in range(height):
        for x in range(width):
            colour = image[x][y]
            new = 1 if colour > 0.5 else 0 # Deterime if the colour should be black or white
            image[x][y] = new # # Set that new colour for the current pixel

            # Now we calcualte the error from what the exact colour should be
            error = colour - new
            # And we distribute that error to the neighbouring pixels (if they exist) with specific weights
            # When we're at the edge of the image the pixels at these positions might not exist,
            # thus we have the try/except blocks
            # We don't howerever concern ourselves with this as it's not not very noticable
            try:
                image[x + 1][y] = image[x + 1][y] + error * 7 / 16
            except IndexError:
                pass
            try:
                image[x - 1][y + 1] = image[x - 1][y + 1] + error * 3 / 16
            except IndexError:
                pass
            try:
                image[x][y + 1] = image[x][y + 1] + error * 5 / 16
            except IndexError:
                pass
            try:
                image[x + 1][y + 1] = image[x + 1][y + 1] + error * 1 / 16
            except IndexError:
                pass

    # Append  a descriptive marking to the end of the file and return it
    return fname + '-bw-FS.png'

# Porudces an image that is reduced to only the colours in the provided palette
# No dithering is done
def colourReduce(image, palette, fname):
    width, height, *rest = image.shape
    for x in range(0, width):
        for y in range(0, height):
            colour = image[x][y]
            # Simply find the nearest colour in the palette and save it
            image[x][y] = nearest(palette, colour)

    # Append  a descriptive marking to the end of the file and return it
    return fname + '-r.png'

# Produce an ordered dither of the image with the give Bayer Filter size
# using only the colours in the provided palette
def colourOrderedDither(image, palette, fname, size):
    threshold = thresholdMap[size]

    width, height, *rest = image.shape
    for x in range(width):
        for y in range(height):
            colour = image[x][y]

            # Find the nearest two colours and their distances from the targeted colour
            colour1, dist1, colour2, dist2 = nearestTwo(palette, colour)
            # Calculate the blend amount by dividing the distance of the farther colour by the
            # total distance from both colours
            # This will produce a higher number when closer to colour 1 and lower when closer to colour 2
            percentage = dist2 / (dist1 + dist2)
            bracket = round(percentage * size**2) # Map this percentage to the appropriate range

            # Finally, check if that bracketed colour should be colour1 or colour2 based on the
            # threshhold for the current pixel
            image[x][y] = colour1 if bracket > threshold[x % size][y % size] + 0.5 else colour2

    # Append  a descriptive marking to the end of the file and return it
    return fname + '-' + str(size) + 'x' + str(size) + '.png'

def colourFloydSteinbergDither(image, palette, fname):
    width, height, *rest = image.shape
    for y in range(height):
        for x in range(width):
            colour = image[x][y]
            new = nearest(palette, colour) # Determine the new colour for the current pixel
            # Compute the error between the new colour and the existing one
            error = [colour[0] - new[0], colour[1] - new[1], colour[2] - new[2]]
            # Set the new pixel's colour
            image[x][y] = new

            # Now we distribute that error to the neighbouring pixels (if they exist) with specific weights
            # When we're at the edge of the image the pixels at these positions might not exist,
            # thus we have the try/except blocks
            # We don't howerever concern ourselves with this as it's not very noticable
            try:
                image[x + 1][y][0] += error[0] * 7 / 16
                image[x + 1][y][1] += error[1] * 7 / 16
                image[x + 1][y][2] += error[2] * 7 / 16
            except IndexError:
                pass
            try:
                image[x - 1][y + 1][0] += error[0] * 3 / 16
                image[x - 1][y + 1][1] += error[1] * 3 / 16
                image[x - 1][y + 1][2] += error[2] * 3 / 16
            except IndexError:
                pass
            try:
                image[x][y + 1][0] += error[0] * 5 / 16
                image[x][y + 1][1] += error[1] * 5 / 16
                image[x][y + 1][2] += error[2] * 5 / 16
            except IndexError:
                pass
            try:
                image[x + 1][y + 1][0] += error[0] * 1 / 16
                image[x + 1][y + 1][1] += error[1] * 1 / 16
                image[x + 1][y + 1][2] += error[2] * 1 / 16
            except IndexError:
                pass

    # Append  a descriptive marking to the end of the file and return it
    return fname + '-FS.png'

# This changes a list of hex string colours into a list of RGB colours
# This is used to map the manual palettes into a data we can work with
def hexToRGB(hexes):
    return [[c for c in bytes.fromhex(hex)] for hex in hexes]

# List of predfined palettes oh hex strings
# Add to this map if you want to define a new custom palette
palettes = {
    'chrono': ['080000', '201A0B', '432817', '492910',
               '234309', '5D4F1E', '9C6B20', 'A9220F',
               '2B347C', '2B7409', 'D0CA40', 'E8A077',
               '6A94AB', 'D5C4B3', 'FCE76E', 'FCFAE2'],

    'eight': ['000000', '0000FF', '00FF00', '00FFFF', 'FF0000', 'FF00FF', 'FFFF00', 'FFFFFF'],

    'blueish': ['000000', '008000', '00FF00',
                '0000FF', '0080FF', '00FFFF',
                '800000', '808000', '80FF00',
                '8000FF', '8080FF', '80FFFF',
                'FF0000', 'FF8000', 'FFFF00',
                'FF00FF', 'FF80FF', 'FFFFFF'],

    'ega16': ['000000', '0000aa', '00aa00', '00aaaa',
              'aa0000', 'aa00aa', 'aa5500', 'aaaaaa',
              '555555', '5555ff', '55ff55', '55ffff',
              'ff5555', 'ff55ff', 'ffff55', 'ffffff'],

    'seurat': ['606f41', '49665e', '3b5f98', '25467e', '6f5e82', 'a52e24', 'c67a29', 'ceaa40', 'dcd4cc', '120e0a'],

    'gypsy': ['aa0e05', '5b9fa6', '151126', 'f6a90d', 'ede5bd', '904e52'],

    'millet': ['bbaf9c', '4b4033', '957b50', 'f3decc', '27211a', 'ba9b61', '705b41', 'dcc5af'],

    'obama': ['01253d', 'e7000a', '5092a0', 'f0dfb3'],
}

# This is the main function the runs when runnging `python dither.py`
def main():
    #######################################
    # CONFIGURATION ZONE
    #
    # This is where you can change any of the variables to alter the settings when running dither.py
    ###########################################
    nColours = 3 # The number colours to use when generating a dynamic palette
                 # This is ignored when manualPalette is specified or when
                 # using a black and white mode
    size = 8 # Size of the Bayer filter pattern to use when producing and ordered dithered image
             # Can be any value from: 2, 3, 4, 8
    infile = 'originals/lenna.png' # The path to the target input image file
                                   # This is relative to the directory you are running `python dither.py` from
                                   # Ideally you should be running it from within the dither folder
    lab = False # Enable Lab colour mode (does not work with Floyd Steinberg dithering modes)
    manualPalette = '' # Specify the name of a palette in the `palettes` map above to use that palette
                       # This will override the nColours setting and not generate a dynmaic palette when set
                       # A value of '' will not set a manual palette and instead generate a dynamic one
                       # e.g. manualPalette = 'ega16'
                       # e.g. manualPalette = 'obama'
    mode = 'colourOrd' # Sets the mode for which type of image to generate
                       # This can be set to one of:
                       #
                       # 'bwOrd': black and white ordered dither
                       # 'bwFS': black and bhite Floyd Steinberg dither
                       # 'reduce': colour reduction without dithering
                       #  'colourOrd': a colour ordered dither
                       #  'colourFS': a colour Floyd Steinberg dither
                       #
                       # e.g. mode = 'colourOrd'
    #####################################
    # END OF CONFIGURATION ZONE
    #
    # Tamper at your own risk
    ###################################

    # Split the filepath into filename and extension
    basename = os.path.basename(infile)
    fname, ext = os.path.splitext(basename)

    # Read in the image file
    image = io.imread(infile)

    # Strip the alpha channel if it exists
    if 'colour' in mode:
        image = image[:,:,:3]

    # Convert the image from 8bits per channel to floats in each channel for precision
    # The added precision is needed to make the error calculations in Floyd Steinberg
    # dithering work properly
    if mode == 'colourFS':
        image = img_as_float(image)

    # Start by converting the image to greyscale to
    if 'bw' in mode:
        image = color.rgb2gray(image)

    # Convert the image the Lab colour space if enabled (and not using Floyd Steinberg)
    if lab and mode != 'colourFS':
        image = color.rgb2lab(image)

    # For colour modes only
    if 'bw' not in mode:
        if manualPalette:
            # Convert our chosen manualPalette to rgb values
            colours = hexToRGB(palettes[manualPalette])
            # Save the palette to a file
            fname += '-' + manualPalette
            savePalette(img_as_float(np.array([colours], dtype=np.uint8))[0], 'palettes/' + manualPalette)

            # Convert our palette to floats for FLoyd Steinberg dithering for the precision
            if mode == 'colourFS':
                colours = img_as_float(np.array([colours], dtype=np.uint8))[0]
            # Convert out palette to Lab colour space if it's enabled
            elif lab:
                colours = color.rgb2lab(np.array([colours], dtype=np.uint8))[0]
            # Make a formal kd-tree palette from these colour
            palette = makePalette(colours)
        else:
            # Dynamically generate an N colour palette for the given image
            palette = findPalette(image, nColours)
            colours = palette.data
            # Add a the number of colours to the output filename
            fname += '-' + str(len(colours)) + 'c'
            # Convert Lab colour space colours to RGB for saving the palette file
            if lab and mode != 'colourFS':
                fname += '-L' # Add an indicator to the filename that this was a Lab colourspace palette
                colours = color.lab2rgb([colours])[0]
            # Convert our palette colours to a consisten range for saving to a file
            colours = img_as_float([colours.astype(np.ubyte)])[0]
            savePalette(colours, 'palettes/' + fname) # Save the palette file

    # Use the right function for the specified mode
    if mode == 'bwOrd':
        fname = bwOrderedDither(image, fname, size)
    if mode == 'bwFS':
        fname = bwFloydSteinbergDither(image, fname)
    if mode == 'reduce':
        fname = colourReduce(image, palette, fname)
    elif mode == 'colourOrd':
        fname = colourOrderedDither(image, palette, fname, size)
    elif mode == 'colourFS':
        fname = colourFloydSteinbergDither(image, palette, fname)

    # Convert Lab colour space images back to RGB for saving the sile
    if lab:
        image = color.lab2rgb(image)

    # Save the file to the `output` folder
    # This is inside the dither folder
    # Note: You should be running `python dither.py` from the `dither` folder
    io.imsave('output/' + fname, image)

    # Indicate the file is saved and we're done
    print(fname + " saved!")

# Python junk
if __name__ == "__main__":
    main()

import os
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
from scipy import spatial
from skimage import io, color
import numpy as np

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

def bwDither(image, outname, size=4):
    threshold = thresholdMap[size]

    width, height = image.shape
    for x in range(0, width, size):
        for y in range(0, height, size):

            count = 0
            total = 0
            for i in range(0, size):
                for j in range(0, size):
                    try:
                        r, g, b = image.getpixel((x + i, y + j))
                        total += (r + g + b) / 3
                        count += 1  # count is needed in case we go past the edge of the picture
                    except IndexError:
                        pass

            grey = total / count
            bracketed = round(grey / (256 / (size * size)))

            for i in range(0, size):
                for j in range(0, size):
                    try:
                        image.putpixel((x + i, y + j), ((bracketed > threshold[i][j] + 0.5 and 255,) * 3))
                    except IndexError:
                        pass

    image.save(outname)

def nearestColour(colour, palette):
    pal = palette.cluster_centers_
    closestColour = pal[0]
    closestDist = colourDist(colour, toCol(pal[0]))
    for c in pal[1:]:
        dist = colourDist(colour, toCol(c))
        if (dist < closestDist):
            closestColour = c
            closestDist = dist

    return toCol(closestColour)

def nearestTwo(palette, colour):
    dist, i = palette.query(colour, 2)
    return toCol(palette.data[i[0]]), dist[0], toCol(palette.data[i[1]]), dist[1]

def colourDist(colour1, colour2):
    r1, g1, b1 = colour1
    r2, g2, b2 = colour2
    return ((r2-r1)**2 + (g2-g1)**2 + (b2-b1)**2)

def makePalette(colours):
    print(colours)
    return spatial.KDTree(colours)

def findPalette(image, nColours):
    data = image.ravel().reshape(-1, 3)
    colours = KMeans(n_clusters=nColours, random_state=0, tol=.01).fit(data)

    return makePalette(colours.cluster_centers_)

def toCol(list):
    return [int(x) for x in list]

def savePalette(colours, fname):
    panelSize = 25
    nColours = len(colours)
    paletteImage = Image.new('RGB', (panelSize * nColours, panelSize))
    draw = ImageDraw.Draw(paletteImage)
    for i, colour in enumerate(colours):
        draw.rectangle([panelSize * i, 0, panelSize * (i + 1), panelSize], tuple(toCol(colour)))
    paletteImage.save(fname + '-palette.png')

def colourReduce(image, palette, fname):
    width, height, channels = image.shape
    for x in range(0, width):
        for y in range(0, height):
            colour = image[x][y]
            nearest = nearestColour(colour, palette)

            image[x][y] = nearest

    return fname + '-r.png'

def colourDither(image, palette, fname, size=4):
    threshold = thresholdMap[size]

    width, height, channels = image.shape
    for x in range(width):
        for y in range(height):
            colour = image[x][y]

            colour1, dist1, colour2, dist2 = nearestTwo(palette, colour)
            percentage = dist2 / (dist1 + dist2)
            bracket = round(percentage * size**2)

            image[x][y] = colour1 if bracket > threshold[x % size][y % size] + 0.5 else colour2

    return fname + '-' + str(size) + 'x' + str(size) + '.png'

def hexToRGB(hexes):
    return [[c for c in bytes.fromhex(hex)] for hex in hexes]

def main():
    nColours = 8
    size = 8
    infile = 'originals/lenna.png'
    basename = os.path.basename(infile)
    fname, ext = os.path.splitext(basename)
    lab = False
    manualPalette = False

    image = io.imread(infile)
    if lab:
        image = color.rgb2lab(image)

    colours = None
    if manualPalette:
        chrono = ['080000', '201A0B', '432817', '492910',
                  '234309', '5D4F1E', '9C6B20', 'A9220F',
                  '2B347C', '2B7409', 'D0CA40', 'E8A077',
                  '6A94AB', 'D5C4B3', 'FCE76E', 'FCFAE2']
        colours = hexToRGB(chrono)
        if (lab):
            labColours = color.rgb2lab(np.array([colours], dtype=np.uint8))[0]
            palette = makePalette(labColours)
        else:
            palette = makePalette(colours)
    else:
        palette = findPalette(image, nColours)
        colours = palette.data

    fname += '-' + str(len(colours)) + 'c'
    if lab:
        fname += '-L'

    savePalette(colours, 'palettes/' + fname)

    # fname = colourReduce(image, fname)
    fname = colourDither(image, palette, fname, size)

    if lab:
        image = color.lab2rgb(image)

    io.imsave('dithered/' + fname, image)

    print(fname + " saved!")

if __name__ == "__main__":
    main()

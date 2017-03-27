import os
import operator
import random
import numpy as np
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
from scipy import spatial
from skimage import io, color

thresholdMap = {
    2: [[0, 3], [2, 1]],
    3: [[0, 6, 4], [7, 5, 1], [3, 2, 8]],
    4: [[0, 12, 3, 15], [8, 4, 11, 7], [2, 14, 1, 13], [10, 6, 9, 5]]
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

paletteKDTree = None

def nearestTwo(colour):
    dist, i = paletteKDTree.query(colour, 2)
    return toCol(paletteKDTree.data[i[0]]), dist[0], toCol(paletteKDTree.data[i[1]]), dist[1]

def colourDist(colour1, colour2):
    r1, g1, b1 = colour1
    r2, g2, b2 = colour2
    return ((r2-r1)**2 + (g2-g1)**2 + (b2-b1)**2)

def findPalette(image, nColours):

    # def kMeans(clusters, data, iters):
    #     for it in range(iters):
    #         assignments = [[] for i in range(len(clusters))]
    #         for point in data:
    #             distances = [(colourDist(point, cluster), i) for i, cluster in enumerate(clusters)]
    #             cluster, i = min(distances)
    #             assignments[i].append(point)
    #
    #         print([len(x) for x in assignments])
    #         for i, assignment in enumerate(assignments):
    #             if len(assignment) > 0:
    #                 clusters[i] = tuple([sum(x) / len(assignment) for x in zip(*assignment)]) # average of the points
    #         print(clusters)

    # start with nColours randomly generated colours
    # palette = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for i in range(nColours)]
    # kMeans(palette, list(image.getdata()), 10)

    # data = np.array(image.getdata())
    data = image.ravel().reshape(-1, 3)
    palette = KMeans(n_clusters=nColours, random_state=0, tol=.01).fit(data)
    print(palette.cluster_centers_)

    global paletteKDTree
    paletteKDTree = spatial.KDTree(palette.cluster_centers_)

    # p = [(180.03311678424606, 79.65831345480383, 91.0300588742831), (103.88152738838883, 30.588313413014607, 70.02926565343564), (119, 252, 20), (213.95614663217222, 128.09845929963964, 118.37952739536173), (229.1862956229898, 188.24502866731925, 166.0561040413928)]
    # return [(int(r), int(g), int(b)) for r, g, b in p]
    return palette

def toCol(list):
    return [int(x) for x in list]

def savePalette(palette, outname):
    panelSize = 25
    nColours = len(palette.cluster_centers_)
    paletteImage = Image.new('RGB', (panelSize * nColours, panelSize))
    draw = ImageDraw.Draw(paletteImage)
    for i, colour in enumerate(palette.cluster_centers_):
        draw.rectangle([panelSize * i, 0, panelSize * (i + 1), panelSize], toCol(colour))
    paletteImage.save(outname + '-' + str(nColours) + 'c-palette.png')

def colourReduce(image, fname, nColours=8):
    palette = findPalette(image, nColours)
    # savePalette(palette)

    width, height, channels = image.shape
    for x in range(0, width):
        for y in range(0, height):
            colour = image[x][y]
            nearest = nearestColour(colour, palette)

            image[x][y] = nearest

    return fname + '-' + str(nColours) + 'c-r.png'

def colourDither(image, fname, nColours=8, size=4):
    threshold = thresholdMap[size]

    palette = findPalette(image, nColours)
    # savePalette(palette, outname)

    width, height, channels = image.shape
    # width, height = image.size
    for x in range(width):
        for y in range(height):

            # count = 0
            # total = (0, 0, 0)
            # for i in range(0, size):
            #     for j in range(0, size):
            #         try:
            #             colour = image.getpixel((x + i, y + j))
            #             total = tuple(map(operator.add, total, colour))
            #             count += 1  # count is needed in case we go past the edge of the picture
            #         except IndexError:
            #             pass

            colour = image[x][y]
            # colour = image.getpixel((x, y))

            # colour = tuple(x / count for x in total)
            colour1, dist1, colour2, dist2 = nearestTwo(colour)
            percentage = dist2 / (dist1 + dist2)
            # bracket = round((percentage / (1 + percentage**2))**(0.5) * size**2)
            bracket = round(percentage * size**2)

            image[x][y] = colour1 if bracket > threshold[x % size][y % size] + 0.5 else colour2
            # image.putpixel((x, y), colour1 if bracket > threshold[x % size][y % size] + 0.5 else colour2)

            # for i in range(0, size):
            #     for j in range(0, size):
            #         try:
            #             image.putpixel((x + i, y + j), colour1 if bracket > threshold[i][j] + 0.5 else colour2)
            #         except IndexError:
            #             pass
    return fname + '-' + str(nColours) + 'c' + '-' + str(size) + 'x' + str(size) + '.png'

def main():
    nColours = 16
    size = 4
    infile = 'chrono-cross.png'
    file, ext = os.path.splitext(infile)
    lab = True

    # image = Image.open('originals/' + infile).convert("RGB")
    image = io.imread('originals/' + infile)
    if lab:
        image = color.rgb2lab(image)

    # colourDither(image, nColours, size)
    fname = 'dithered/' + file
    if (lab):
        fname += '-l'

    # fname = colourReduce(image, fname, nColours)
    fname = colourDither(image, fname, nColours, size)

    if lab:
        image = color.lab2rgb(image)

    io.imsave(fname, image)

    print(fname + " saved!")
    # image.save(filename)

if __name__ == "__main__":
    main()

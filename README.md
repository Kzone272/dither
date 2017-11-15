Provided
==

All the code can be found in the dither.py file.

I've provided the original images I used in the originals folder.
The outputs from various runs in the outputs folder.
The palettes from various runs are in the palettes folder.
There is more there than I included in the PowerPoint so you can have look at some of my experiments if you're curious.

Due to technical limitations and time restrictions, I was not able to make a distributable build of the code for you.
So instead I've provided these instruction for how to run this program yourself.

Installation Instructions
==

You will need python 3.4 installed on your computer.

If you have a different version installed, an easy way to switch versions:
    Install Miniconda https://conda.io/miniconda.html
    Once installed run `conda install python=3.4` from command line
    You can easily switch back again by running `conda install python=3.5` (or your preferred version)

Once you have python 3.4 installed, you will need to install the dependencies.

From command line, run:

    pip install numpy
    pip install Pillow
    pip install scipy
    pip install scikit-image
    pip install scikit-learn

Numpy is a common package that does lots of multidimensional array procedures.
Pillow is the common image library for simple image manipulation.
scipy is a library containing a number of mathematical operations and functions.  I use it for the KD-Tree implementation.
scikit-image is library that adds more image manipulation capabilities.  I use it for Lab colour space conversion.
scikit-learn is library filled with lots of machine learning tools.  I use it for the the K-Means clustering algorithm.

Aside:
    I actually implemented my own kmeans clustering algorithm but it was way too slow because it was written in python.
    scikit-learn is written in C with Python bindings and runns many times faster

Once you've got those installed you should be good to go.

Open a terminal and navigate to the `dither` folder on your computer (the folder containing this README)

Once there simply run the following from the command line:

    python dither.py

After a few seconds you will have your output image and palette saved to their respective folders.
A message should appear telling you what files were saved.

Note: The running time for the 512 by 512 pixel Lenna image (originals/lenna.png) takes about 30 seconds to
      generate a coloured dither image.  Expect this time to scale about linearly with the number of pixels
      in the image.

Configuration
==

At the very bottom of dither.py you'll find the main() function.  At the top of the main function you'll find
the Configuration Zone.  There you can modify the input variables to change the settings used when running the program.
Further instructions on those details can be found in the comments in that section.

I wanted to make a nice GUI to make it much easier for you to use, but unfortunately I didn't have the time.
Hopefully this isn't too cumbersome.

If You Have Any Problems
==

If for some reason it still doesn't work, the PowerPoint should give a pretty good idea what this program can do.

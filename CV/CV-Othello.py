import math

import numpy as np
import scipy as sp
from scipy import misc
import skimage
from scipy import ndimage as ni
from skimage.transform import hough_circle, hough_circle_peaks, probabilistic_hough_line
from skimage.feature import canny
from skimage.draw import line_aa
import matplotlib.pyplot as plt

im = misc.imresize(sp.misc.imread("Othello_orig.jpg")[:,16:-16], 1/10) / 256

EDGE_THRESHOLD = 0.4
CIRCLE_THRESHOLD = 0.55

MAX_ROT = 10 # Maximum image rotation, degrees

piece_size = 0.05 * (im.shape[0] + im.shape[1]) / 2
print("Piece size:", piece_size)



# Extract the background board
print("Green edge detection... ", end="", flush=True)
greens = im[:,:,1] - im[:,:,(0,2)].sum(axis=2) / 2
plt.subplot(211); plt.imshow(im)
plt.subplot(212); plt.imshow(greens)
plt.show()

# Edge detection
sx = ni.sobel(greens, axis=0, mode="constant")
sy = ni.sobel(greens, axis=1, mode="constant")
sobel = np.hypot(sx, sy)
edges = sobel > EDGE_THRESHOLD

print("Done")


# Find circles (othello pieces)
print("Circle detection... ", end="", flush=True)
radii_lim = np.arange(piece_size * 0.8, piece_size * 1.2)
hough_res = hough_circle(edges, radii_lim)
prob, cx, cy, radii = hough_circle_peaks(
        hough_res,
        radii_lim,
        min_xdistance=int(piece_size),
        min_ydistance=int(piece_size),
        threshold=CIRCLE_THRESHOLD
        )

circles = list(zip(prob, cx, cy, radii))
print("Done")


# Find lines
print("Line detection... ", end="", flush=True)


angles = np.linspace(-math.radians(MAX_ROT), math.radians(MAX_ROT), 100)

angles = np.concatenate((angles, angles + np.pi / 2))

lines = probabilistic_hough_line(
        edges,
        line_length=im.shape[0]/2,
        theta=angles
        )
print("Done")



markers = np.zeros((*sobel.shape, 3))
for (prob, x, y, rad) in circles:
    print("    circle x={:.2f}, y={:.3f}, r={:.2f}".format(x, y, rad))
    rad = int(rad)
    for x_ in range(-rad, rad):
        for y_ in range(-rad, rad):
            aa = rad - (x_ ** 2 + y_ ** 2) ** 0.5
            aa_clip = np.clip(aa, 0, 1)
            markers[y + y_, x + x_, 0] += prob * aa_clip

for ((x0, y0), (x1, y1)) in lines:
    print("    line x₀={:.2f}, y₀={:.2f}, x₁={:.2f}, y₁={:.2f}".format(x0, y0, x1, y1))
    xs, ys, intensities = line_aa(x0, y0, x1, y1)
    markers[ys, xs, 2] += intensities

sobel_green = np.array([np.zeros_like(sobel), sobel, np.zeros_like(sobel)]).transpose((1, 2, 0))

plt.subplot(221); plt.imshow(sobel)
plt.subplot(222); plt.imshow(edges)
plt.subplot(223); plt.imshow(markers + sobel_green)
plt.subplot(224); plt.imshow(markers)
plt.show()

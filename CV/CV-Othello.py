import math
import random

import numpy as np
import scipy as sp
from scipy import misc
import skimage
from scipy import ndimage as ni
from skimage.transform import hough_circle, hough_circle_peaks, probabilistic_hough_line
from skimage.feature import canny
from skimage.draw import line_aa
import matplotlib.pyplot as plt

im = misc.imresize(sp.misc.imread("img0.jpg"), 1/10) / 256

EDGE_THRESHOLD = 0.4
CIRCLE_THRESHOLD = 0.45
CIRCLE_GREEN_THRESHOLD = 0.1

MAX_ROT = 10 # Maximum image rotation, degrees
AMOUNT_OF_ANGLES = 100 # How many angles to test


piece_size = 0.05 * (im.shape[0] + im.shape[1]) / 2
print("Piece size:", piece_size)


# Extract the background board
print("Green edge detection... ", end="", flush=True)
greens = im[:,:,1] - im[:,:,(0,2)].sum(axis=2) / 2

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

circles_not_intersecting = []
for i, circle in enumerate(circles):
    prob, x, y, rad = circle
    intersecting = False
    for (_, x_, y_, rad_) in circles[:i]:
        dist_sqrd = (x - x_) ** 2 + (y - y_) ** 2
        if dist_sqrd < ((rad + rad_) / 2) ** 2:
            intersecting = True
            break

    if not intersecting:
        circles_not_intersecting.append(circle)

circles_with_colors = []
for prob, x, y, rad in circles_not_intersecting:
    rad = int(rad)

    brightness_in_rad = []
    green_in_rad = []

    for x_ in range(-rad, rad):
        for y_ in range(-rad, rad):
            if (x_ ** 2 + y_ ** 2) < (rad - 3) ** 2:
                brightness_in_rad.append(im[y + y_, x + x_].sum() / 3)

                green_in_rad.append(greens[y + y_, x + x_])

    brightness_in_rad = np.array(brightness_in_rad)
    avg_green = sum(green_in_rad) / len(green_in_rad)


    blacks = (brightness_in_rad < 0.6).sum()
    whites = (brightness_in_rad > 0.6).sum()

    if avg_green < CIRCLE_GREEN_THRESHOLD:
        circles_with_colors.append((prob, x, y, rad, blacks > whites / 2))

circles = circles_with_colors

print("Done")


# Find lines (upper and lower edges)
print("Line and corner detection... ", end="", flush=True)

angles = np.linspace(-math.radians(MAX_ROT), math.radians(MAX_ROT), AMOUNT_OF_ANGLES)
angles = np.concatenate((angles, angles + np.pi / 2))

lines = probabilistic_hough_line(
        edges,
        line_length=im.shape[0]/2,
        theta=angles,
        threshold=0
        )


# Remove close duplicates
lines_not_intersecting = []
for i, line in enumerate(lines):
    (x0, y0), (x1, y1) = line
    intersecting = False

    for other in lines[:i]:
        (X0, Y0), (X1, Y1) = other

        # Move the opposite line to the same orientation as the current line
        if (x0 - X0) ** 2 + (y0 - Y0) ** 2 > (x0 - X1) ** 2 + (y0 - Y1):
            (X1, Y1), (X0, Y0) = other


        theta = math.atan2(y0 - y1, x0 - x1)
        theta_1 = math.atan2(y0 - Y1, x0 - X1)
        theta_2 = math.atan2(Y0 - y1, X0 - x1)

        diff = (theta - theta_1) ** 2 + (theta - theta_2) ** 2

        if diff < 0.01:
            intersecting = True

    if not intersecting:
        lines_not_intersecting.append(line)
    else:
        print("Intersecting")

lines = lines_not_intersecting

# Find corners
corners = []
for line in lines:
    (x0, y0), (x1, y1) = line
    for other in lines:
        if other == line: continue
        (X0, Y0), (X1, Y1) = other
        if X0 == X1: continue

        # Find intersection
        if x0 == x1:
            K = (Y0 - Y1) / (X0 - X1)
            # y=K(x-X0)+Y0 where x=X0
            corners.append((X0, K * (x0 - X0) + Y0))
            continue

        k = (y0 - y1) / (x0 - x1)
        K = (Y0 - Y1) / (X0 - X1)

        # y=k(x-x0)+y0, y=K(x-X0)+Y0
        # kx-kx0+y0 = Kx-KX0+Y0
        x = (y0 - Y0 + K*X0 - k*x0) / (K-k)
        y = k * (x - x0) + y0

        if not any(map(lambda pos: math.isclose(x, pos[0]) and math.isclose(y, pos[1]), corners)):
            corners.append((x, y))

corners = list(filter(lambda p: 0 < p[0] < im.shape[1] and 0 < p[1] < im.shape[0], corners))
print("Done")

markers = np.zeros((*sobel.shape, 3))
for prob, x, y, rad, is_black in circles:
    print("    circle x={:.2f}, y={:.3f}, r={:.2f}, is_black={}".format(x, y, rad, is_black))
    rad = int(rad)
    for x_ in range(-rad, rad):
        for y_ in range(-rad, rad):
            aa = rad - (x_ ** 2 + y_ ** 2) ** 0.5
            aa_clip = np.clip(aa, 0, 1)
            if is_black:
                markers[y + y_, x + x_, 2] += prob * aa_clip
            else:
                markers[y + y_, x + x_] += prob * aa_clip

for (x0, y0), (x1, y1) in lines:
    print("    line x₀={:.2f}, y₀={:.2f}, x₁={:.2f}, y₁={:.2f}".format(x0, y0, x1, y1))
    for x_ in [-1, 0, 1]:
        for y_ in [-1, 0, 1]:
            xs, ys, intensities = line_aa(x0, y0, x1, y1)
            intensities_colors = np.array([intensities * random.random() for _ in range(3)]).transpose((1, 0))
            try: markers[ys + y_, xs + x_] += intensities_colors / 3
            except: pass

for x, y in corners:
    print("    corner x={:.2f}, y={:.2f}".format(x, y))
    rad = 10
    for x_ in range(-rad, rad):
        for y_ in range(-rad, rad):
            aa = rad - (x_ ** 2 + y_ ** 2) ** 0.5
            aa_clip = np.clip(aa, 0, 1)
            x__ = int(x) + x_
            y__ = int(y) + y_
            if 0 <= y__ < markers.shape[0] and 0 <= x__ < markers.shape[1]:
                markers[y__, x__, 1] += aa_clip



sobel_green = np.array([np.zeros_like(sobel), sobel, np.zeros_like(sobel)]).transpose((1, 2, 0))
mask = 1 - markers.sum(axis=2)

plt.subplot(231); plt.imshow(np.clip(im, 0, 1))
plt.subplot(232); plt.imshow(np.clip(sobel, 0, 1))
plt.subplot(233); plt.imshow(np.clip(edges, 0, 1))
plt.subplot(234); plt.imshow(np.clip(markers, 0, 1))
plt.subplot(235); plt.imshow(np.clip(markers + sobel_green, 0, 1))
plt.subplot(236); plt.imshow(np.clip(markers + im * np.array([mask, mask, mask]).transpose((1, 2, 0)), 0, 1))
plt.show()

import math
import random
import sys

import numpy as np
import scipy as sp
from scipy import misc
import skimage
from scipy import ndimage as ni
from skimage.transform import hough_circle, hough_circle_peaks, probabilistic_hough_line
from skimage.feature import canny
from skimage.draw import line_aa
import matplotlib.pyplot as plt

from dependencies import get_value, set_value, dynamic, placeholder

set_value("CIRCLE_EDGE_THRESHOLD", 0.3)   # Sharp edge minimum
set_value("CIRCLE_THRESHOLD", 0.45)       # For hough transform
set_value("CIRCLE_GREEN_THRESHOLD", 0.1)  # How much green is allowed in a circle (average of green - not green)
set_value("LINE_EDGE_THRESHOLD", 0.3)     # Sharp edge minimum. This is usually what you want to change when things aren't working
set_value("MAX_ROT", 10)                  # Maximum line rotation, degrees
set_value("AMOUNT_OF_ANGLES", 10)         # How many angles to test
set_value("SIMILARITY_ANGLE", 5)          # How similar two lines angles can be to be considered the same

placeholder("path")

@dynamic("im")
def load_im(path):
    return misc.imresize(sp.misc.imread(path), 1/10) / 256


@dynamic("piece_size")
def get_piece_size(im):
    return 0.05 * (im.shape[0] + im.shape[1]) / 2


@dynamic("greens")
def amount_of_green(im):
    return im[:, :, 1] - im[:, :, (0, 2)].sum(axis=2) / 2


@dynamic("sobel")
def generate_sobel(greens):
    sx = ni.sobel(greens, axis=0, mode="constant")
    sy = ni.sobel(greens, axis=1, mode="constant")
    return np.hypot(sx, sy)


@dynamic("circle_edges")
def get_circle_edges(sobel, CIRCLE_EDGE_THRESHOLD):
    return sobel > CIRCLE_EDGE_THRESHOLD


@dynamic("line_edges")
def get_circle_edges(sobel, LINE_EDGE_THRESHOLD):
    return sobel > LINE_EDGE_THRESHOLD

@dynamic("circles")
def generate_circles(circle_edges, greens, im, piece_size, CIRCLE_THRESHOLD, CIRCLE_GREEN_THRESHOLD):
    # Find circles (othello pieces)
    print("Circle detection... ", end="", flush=True)
    radii_lim = np.arange(piece_size * 0.8, piece_size * 1.2)
    hough_res = hough_circle(circle_edges, radii_lim)
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

    return circles

@dynamic("lines")
def generate_lines(line_edges, MAX_ROT, AMOUNT_OF_ANGLES, SIMILARITY_ANGLE):
    print("Line detection... ", end="", flush=True)

    angles = np.linspace(
            -math.radians(MAX_ROT),
            math.radians(MAX_ROT),
            AMOUNT_OF_ANGLES)
    angles = np.concatenate((angles, angles + np.pi / 2))

    lines = probabilistic_hough_line(
            line_edges,
            line_length=line_edges.shape[0]/3,
            theta=angles,
            threshold=0
            )


    # Returns the lowest absolute value of x % y
    def moddown(x, y):
        x = x % y
        if x > y / 2:
            return x - y
        else:
            return x

    # Remove close duplicates
    lines_not_intersecting = []
    for i, line in enumerate(lines):
        (x0, y0), (x1, y1) = line
        intersecting = False

        for other in lines[:i]:
            (X0, Y0), (X1, Y1) = other

            for _ in range(2):
                # Move the opposite line to the same orientation as the current line
                (X1, Y1), (X0, Y0) = (X0, Y0), (X1, Y1)

                # Compare the angle of line to the angle if you would exchange one of the ends with the other lines
                # corresponding end

                theta   = math.atan2(y0 - y1, x0 - x1)
                theta_1 = math.atan2(y0 - Y1, x0 - X1)
                theta_2 = math.atan2(Y0 - y1, X0 - x1)

                diff = max(abs(moddown(theta - theta_1, np.pi)), abs(moddown(theta - theta_2, np.pi)))

                # print("    ", other, diff)
                # print("        ", theta, theta_1, theta_2)

                if diff < math.radians(SIMILARITY_ANGLE):
                    intersecting = True
                    break

            if intersecting:
                break
        if not intersecting:
            lines_not_intersecting.append(line)

    lines = lines_not_intersecting
    print("Done")

    return lines

@dynamic("corners")
def generate_corners(lines, im):
    print("Corner detection... ", end="", flush=True)

    corners = []
    for line in lines:
        (x0, y0), (x1, y1) = line
        for other in lines:
            (X0, Y0), (X1, Y1) = other

            if other == line:
                continue
            if X0 == X1:
                continue

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

    corners = list(filter(
        lambda p:
            im.shape[1] * -0.1 < p[0] < im.shape[1] * 1.1 and
            im.shape[0] * -0.1 < p[1] < im.shape[0] * 1.1, corners))

    print("Done")

    return corners

@dynamic("othello_grid")
def generate_othello_grid(corners, im, circles):
    # Generate the grid from the pieces
    if len(corners) == 4:
        def corner_sort(corner):
            is_lower = corner[1] > im.shape[0] / 2
            is_right = corner[0] > im.shape[1] / 2
            return is_lower * 2 + is_right

        corners.sort(key=corner_sort)
        upper_left, upper_right, lower_left, lower_right = corners

        othello_grid = np.zeros((8, 8))

        grid_width = upper_right[0] - upper_left[0]
        grid_height = lower_left[1] - upper_left[1]

        midpoint_x = (upper_right[0] + upper_left[0] + lower_right[0] + lower_left[0]) / 4
        midpoint_y = (upper_right[1] + upper_left[1] + lower_right[1] + lower_left[1]) / 4

        grid_rot_upper = math.atan2(upper_right[1] - upper_left[1], upper_right[0] - upper_left[0])
        grid_rot_lower = math.atan2(lower_right[1] - lower_left[1], lower_right[0] - lower_left[0])
        grid_rot = (grid_rot_upper + grid_rot_lower) / 2

        print("Grid = {:.2f}x{:.2f} ↺ {:.2f}°".format(grid_width, grid_height, math.degrees(grid_rot)))

        for circle in circles:
            _, x, y, _, is_black = circle

            dx = (x - midpoint_x) / grid_width
            dy = (y - midpoint_y) / grid_height

            # Rotate the points
            x_res = dx * math.cos(grid_rot) - dy * math.sin(grid_rot)
            y_res = dx * math.sin(grid_rot) + dy * math.cos(grid_rot)

            x_othello = int((x_res + 0.5) * 8)
            y_othello = int((y_res + 0.5) * 8)

            othello_grid[y_othello, x_othello] = is_black + 1
    else:
        print("\033[38;5;1m\033[1mWRONG NUMBERS OF CORNERS!!!\033[0m")
        othello_grid = np.zeros((8, 8))

    return othello_grid

def draw():
    im = get_value("im")
    greens = get_value("greens")
    circle_edges = get_value("circle_edges")
    line_edges = get_value("line_edges")
    sobel = get_value("sobel")
    othello_grid = get_value("othello_grid")

    print("Drawing... ", end="", flush=True)
    markers = np.zeros((*sobel.shape, 3))
    for prob, x, y, rad, is_black in get_value("circles"):
        # print("    circle x={:.2f}, y={:.3f}, r={:.2f}, is_black={}".format(x, y, rad, is_black))
        rad = int(rad)
        for x_ in range(-rad, rad):
            for y_ in range(-rad, rad):
                aa = rad - (x_ ** 2 + y_ ** 2) ** 0.5
                aa_clip = np.clip(aa, 0, 1)
                if is_black:
                    markers[y + y_, x + x_, 2] += prob * aa_clip
                else:
                    markers[y + y_, x + x_] += prob * aa_clip

    for (x0, y0), (x1, y1) in get_value("lines"):
        # print("    line x₀={:.2f}, y₀={:.2f}, x₁={:.2f}, y₁={:.2f}".format(x0, y0, x1, y1))
        for x_ in [-1, 0, 1]:
            for y_ in [-1, 0, 1]:
                xs, ys, intensities = line_aa(x0, y0, x1, y1)
                try:
                    markers[ys + y_, xs + x_, 0] += intensities / 3
                except:
                    pass

    for x, y in get_value("corners"):
        # print("    corner x={:.2f}, y={:.2f}".format(x, y))
        rad = 10
        for x_ in range(-rad, rad):
            for y_ in range(-rad, rad):
                aa = rad - (x_ ** 2 + y_ ** 2) ** 0.5
                aa_clip = np.clip(aa, 0, 1)
                x__ = int(x) + x_
                y__ = int(y) + y_
                if 0 <= y__ < markers.shape[0] and 0 <= x__ < markers.shape[1]:
                    markers[y__, x__, 1] += aa_clip

    print("Done")

    sobel_green = np.array([np.zeros_like(sobel), sobel, np.zeros_like(sobel)]).transpose((1, 2, 0))
    mask = 1 - markers.sum(axis=2) / 3

    othello_show = np.zeros_like(othello_grid)
    othello_show[othello_grid == 0] = 0
    othello_show[othello_grid == 1] = 1
    othello_show[othello_grid == 2] = 0.3

    plt.subplot(251); plt.imshow(np.clip(im, 0, 1))
    plt.subplot(252); plt.imshow(np.clip(greens, 0, 1))
    plt.subplot(253); plt.imshow(np.clip(sobel, 0, 1))
    plt.subplot(254); plt.imshow(np.clip(circle_edges / 2 + ni.gaussian_filter(circle_edges * 1., 1) / 2, 0, 1))
    plt.subplot(255); plt.imshow(np.clip(line_edges / 2   + ni.gaussian_filter(line_edges * 1.,   1) / 2, 0, 1))
    plt.subplot(245); plt.imshow(np.clip(markers, 0, 1))
    plt.subplot(246); plt.imshow(np.clip(markers + sobel_green, 0, 1))
    plt.subplot(247); plt.imshow(np.clip(markers + im * np.array([mask, mask, mask]).transpose((1, 2, 0)), 0, 1))
    plt.subplot(248); plt.imshow(othello_show, cmap="ocean", vmin=0, vmax=1)
    plt.show()

import os
set_value("path", sys.argv[1] if len(sys.argv) == 2 else "img0.jpg")
draw()
set_value("path", os.path.expanduser("~/Othello_big/img4.jpg"))
draw()
set_value("LINE_EDGE_THRESHOLD", 0.4)
draw()

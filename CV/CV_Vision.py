import math
import random
import sys
from itertools import product

import numpy as np
import scipy as sp
from scipy import misc
import skimage
from scipy import ndimage as ni
from skimage.transform import hough_circle, hough_circle_peaks, probabilistic_hough_line
from skimage.segmentation import watershed
from skimage.feature import canny
from skimage.filters import gaussian
from skimage.draw import line_aa, circle
import matplotlib.pyplot as plt

from dependencies import get_value, set_value, dynamic, placeholder, draw_dependencies

set_value("CIRCLE_EDGE_THRESHOLD", 0.3)   # Sharp edge minimum
set_value("CIRCLE_THRESHOLD", 0.45)       # For hough transform
set_value("CIRCLE_GREEN_THRESHOLD", 0.1)  # How much green is allowed in a circle (average of green - not green)
set_value("LINE_EDGE_THRESHOLD", 0.2)     # Sharp edge minimum. This is usually what you want to change when things aren't working
set_value("LINE_MIN_LENGTH_RATIO", 0.25)  # How long a line has to be in relation to the input image
set_value("PIECE_IMAGE_RATIO", 0.04)      # How large a circle is compared to the whole image
set_value("MAX_ROT", 10)                  # Maximum line rotation, degrees
set_value("AMOUNT_OF_ANGLES", 10)         # How many angles to test
set_value("SIMILARITY_ANGLE", 15)         # How similar two lines angles can be to be considered the same
set_value("LINE_TRIES", 1)                # How many passes to do to find lines

set_value("corners_to_remove", set())
set_value("corners_to_add", set())
placeholder("path")

@dynamic("im")
def load_im(path):
    try:
        im = sp.misc.imread(path)
    except Exception as e:
        print("Could not read image!")
        print(e)
        return np.zeros((1, 1, 3))
    return misc.imresize(im, 2 * 300 / sum(im.shape[:2])) / 256


@dynamic("piece_size")
def get_piece_size(im, PIECE_IMAGE_RATIO):
    return PIECE_IMAGE_RATIO * (im.shape[0] + im.shape[1]) / 2


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


@dynamic("circles")
def generate_circles(im, circle_edges, greens, piece_size, CIRCLE_THRESHOLD, CIRCLE_GREEN_THRESHOLD):
    if im.shape[0] == im.shape[1] == 1:
        return []
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
                    if 0 <= y + y_ < im.shape[0] and \
                       0 <= x + x_ < im.shape[1]:
                        brightness_in_rad.append(im[y + y_, x + x_].sum() / 3)

                        green_in_rad.append(greens[y + y_, x + x_])

        brightness_in_rad = np.array(brightness_in_rad)
        avg_green = sum(green_in_rad) / (len(green_in_rad) + 1)

        blacks = (brightness_in_rad < 0.6).sum()
        whites = (brightness_in_rad > 0.6).sum()

        if avg_green < CIRCLE_GREEN_THRESHOLD:
            circles_with_colors.append((prob, x, y, rad, blacks > whites / 2))

    circles = circles_with_colors

    print("Done")

    return circles

@dynamic("line_edges_all")
def get_circle_edges(sobel, circles, LINE_EDGE_THRESHOLD):
    sobel_without_circles = np.array(sobel)

    for _, x, y, rad, _ in circles:
        xs, ys = circle(x, y, rad * 1.2)
        for x_, y_ in zip(xs, ys):
            if 0 <= y_ < sobel.shape[0] and \
               0 <= x_ < sobel.shape[1]:
                sobel_without_circles[y_, x_] = 0

    sobel_without_circles = gaussian(sobel_without_circles, sigma=2)
    return np.array(sobel_without_circles > LINE_EDGE_THRESHOLD, dtype="float32")


@dynamic("line_edges")
def remove_other(line_edges_all):
    # Remove all but the biggest object (second biggest to not include the background)
    labels, _ = ni.label(line_edges_all)
    sizes = np.bincount(labels.ravel())

    if len(sizes) >= 2:
        second_best = sizes.argsort()[-2]
        return np.array(labels == second_best, dtype="float32")
    else:
        return line_edges_all

@dynamic("lines_intersecting")
def generate_lines(line_edges, MAX_ROT, AMOUNT_OF_ANGLES, LINE_TRIES, LINE_MIN_LENGTH_RATIO):
    print("Line detection... ", end="", flush=True)

    angles = np.linspace(
            -math.radians(MAX_ROT),
            math.radians(MAX_ROT),
            AMOUNT_OF_ANGLES)
    angles = np.concatenate((angles, angles + np.pi / 2))

    lines = []
    for i in range(LINE_TRIES):
        new_lines = probabilistic_hough_line(
                line_edges + np.random.random(size=line_edges.shape) > 0.95,
                line_length=line_edges.shape[0] * LINE_MIN_LENGTH_RATIO,
                theta=angles,
                threshold=0,
                line_gap=5,
                )

        lines += new_lines

    return lines


@dynamic("lines")
def generate_lines(lines_intersecting, line_edges, SIMILARITY_ANGLE):

    # Returns the lowest absolute value of x % y
    def moddown(x, y):
        x = x % y
        if x > y / 2:
            return x - y
        else:
            return x


    # Remove close duplicates
    lines_not_intersecting = []

    for i, line in enumerate(lines_intersecting):
        (x0, y0), (x1, y1) = line
        intersecting = False

        for other in lines_intersecting[:i]:
            (X0, Y0), (X1, Y1) = other

            for _ in range(2):
                # Move the opposite line to the same orientation as the current line
                (X1, Y1), (X0, Y0) = (X0, Y0), (X1, Y1)

                # Compare the angle of line to the angle if you would exchange one of the ends with the other lines
                # corresponding end

                theta   = math.atan2(y0 - y1, x0 - x1)
                theta_1 = math.atan2(y0 - Y1, x0 - X1)
                theta_2 = math.atan2(Y0 - y1, X0 - x1)

                diff_theta = max(abs(moddown(theta - theta_1, np.pi)), abs(moddown(theta - theta_2, np.pi)))

                if diff_theta < math.radians(SIMILARITY_ANGLE):
                    intersecting = True
                    break

            if intersecting:
                break

        if not intersecting:
            lines_not_intersecting.append(line)

    print("Done")


    return lines_not_intersecting

@dynamic("intersections_all")
def generate_corners(lines, im, corners_to_add):
    print("Corner detection... ", end="", flush=True)

    corners = []
    for line in lines:
        (x0, y0), (x1, y1) = line
        for other in lines:
            (X0, Y0), (X1, Y1) = other

            if other == line or X0 == X1:
                continue

            # Find intersection
            if x0 == x1:
                K = (Y0 - Y1) / (X0 - X1)
                # y=K(x-X0)+Y0 where x=X0
                corners.append((X0, K * (x0 - X0) + Y0))
                continue

            k = (y0 - y1) / (x0 - x1)
            K = (Y0 - Y1) / (X0 - X1)

            if K - k == 0:
                continue

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

    return corners + list(corners_to_add)

@dynamic("intersections")
def get_intersections(intersections_all, corners_to_remove):
    return list(filter(lambda corner: corner not in corners_to_remove, intersections_all))

@dynamic("corners")
def get_corners(intersections, im):
    corners = [None, None, None, None]
    for x in [0, 1]:
        for y in [0, 1]:
            corner_idx = y * 2 + x
            best = None

            for inter in intersections:
                x_, y_ = x * im.shape[1], y * im.shape[0]
                score = ( (inter[0] - x_) ** 2 + (inter[1] - y_) ** 2 ) ** 0.5

                if best == None or best > score:
                    corners[corner_idx] = inter
                    best = score
            if corners[corner_idx] == None:
                corners[corner_idx] = (0, 0)

    return corners

@dynamic("othello_grid")
def generate_othello_grid(corners, im, circles):
    # Generate the grid from the pieces
    corners_amount = len(set(corners))
    if corners_amount == 4:
        upper_left, upper_right, lower_left, lower_right = corners

        othello_grid = np.zeros((8, 8), dtype="int")

        grid_width = upper_right[0] - upper_left[0]
        grid_height = lower_left[1] - upper_left[1]

        if grid_width == 0 or grid_height == 0:
            return np.zeros((8, 8), dtype="int")

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

            for (x_round, y_round) in product([0.5, 0.2, 0.8], [0.5, 0.2, 0.8]):
                x_othello = int((x_res + x_round) * 8)
                y_othello = int((y_res + y_round) * 8)

                if 0 <= x_othello < othello_grid.shape[1] and \
                   0 <= y_othello < othello_grid.shape[0]:
                   if othello_grid[y_othello, x_othello] == 0:
                       othello_grid[y_othello, x_othello] = is_black + 1
                       break
    else:
        print("\033[38;5;1m\033[1mWRONG NUMBERS OF CORNERS: EXPECTED 4, GOT {}!!!\033[0m".format(corners_amount))
        othello_grid = np.zeros((8, 8), dtype="int")

    return othello_grid

@dynamic("drawn")
def generate_drawn(im, greens, circle_edges, line_edges_all, line_edges, sobel, othello_grid):

    print("Drawing... ", end="", flush=True)
    markers = np.zeros((*sobel.shape, 3))
    for prob, x, y, rad, is_black in get_value("circles"):
        # print("    circle x={:.2f}, y={:.3f}, r={:.2f}, is_black={}".format(x, y, rad, is_black))
        rad = int(rad)
        for x_ in range(-rad, rad):
            for y_ in range(-rad, rad):
                if 0 <= y + y_ < sobel.shape[0] and \
                   0 <= x + x_ < sobel.shape[1]:
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

    for (x0, y0), (x1, y1) in get_value("lines_intersecting"):
        # print("    line x₀={:.2f}, y₀={:.2f}, x₁={:.2f}, y₁={:.2f}".format(x0, y0, x1, y1))
        xs, ys, intensities = line_aa(x0, y0, x1, y1)
        try:
            markers[ys + y_, xs + x_, 2] += intensities / 3
        except:
            pass

    for inter in get_value("intersections_all"):
        x, y = inter
        col = 0 if inter in get_value("corners") else 1

        # print("    corner x={:.2f}, y={:.2f}".format(x, y))
        rad = 10
        for x_ in range(-rad, rad):
            for y_ in range(-rad, rad):
                aa = rad - (x_ ** 2 + y_ ** 2) ** 0.5
                aa_clip = np.clip(aa, 0, 1)
                x__ = int(x) + x_
                y__ = int(y) + y_
                if 0 <= y__ < markers.shape[0] and 0 <= x__ < markers.shape[1]:
                    markers[y__, x__, col] += aa_clip

    print("Done")

    sobel_green = np.array([np.zeros_like(sobel), sobel, np.zeros_like(sobel)]).transpose((1, 2, 0))
    mask = 1 - markers.sum(axis=2) / 3

    othello_show = np.zeros_like(othello_grid, dtype="float")
    othello_show[othello_grid == 0] = 0
    othello_show[othello_grid == 1] = 1
    othello_show[othello_grid == 2] = 0.5

    return \
            (im,
            greens,
            sobel,
            circle_edges / 2 + ni.gaussian_filter(circle_edges * 1., 1) / 2,
            line_edges_all,
            line_edges,
            markers,
            markers + sobel_green,
            markers + im * np.array([mask, mask, mask]).transpose((1, 2, 0)),
            othello_show
            )

def draw():
    for i, image in enumerate(get_value("drawn")):
        plt.subplot(2, 6, i + 1); plt.imshow(image, vmin=0, vmax=1)
    plt.show()


if __name__ == "__main__":
    draw_dependencies()

    set_value("path", sys.argv[1] if len(sys.argv) == 2 else "img0.jpg")
    draw()
    # set_value("path", "img10.jpg")
    # draw()
    # set_value("LINE_EDGE_THRESHOLD", 0.4)
    # draw()

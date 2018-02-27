import os
import argparse
# Parse arguments
parser = argparse.ArgumentParser(description="Process images of Othello boards")
parser.add_argument(
        "--input", "-i",
        type=str, default="img{}.jpg",
        help="The path to read images from. Uses `.format(n)` where `n` is the index of the image to load. Example: `img{}.jpg` reads img0.jpg, img1.jpg, ...")
parser.add_argument(
        "--output-images", "-o",
        type=str, default=os.path.join("res", "img_small{}.jpg"),
        help="Where to save each image. Also uses the `.format(n)`")
parser.add_argument(
        "--output-moves", "-O",
        type=str, default=os.path.join("res", "moves.txt"),
        help="Where to store the move data")
parser.add_argument(
        "--img-nr", "-n",
        type=int, default=0,
        help="What image to start on")

args = parser.parse_args()

import sys
from ast import literal_eval

import pygame
import scipy as sp
import numpy as np
from scipy.misc import imresize, imsave, imread

from dependencies import get_value, set_value, dynamic, placeholder, recalc
import CV_Vision


pygame.init()

width, height = size = 1080, 800

MENU_HEIGHT = 60
MENU_PIC_WIDTH = 40
MENU_ITEM_WIDTH = width / 12
IMG_WIDTH = width / 2
OPTION_HEIGHT = 24
PIECE_RADIUS = 24

BUTTON_SIZE = 60

OPTION_FONT = pygame.font.Font(pygame.font.match_font("helveticaneue"), OPTION_HEIGHT)
BUTTON_FONT = pygame.font.Font(pygame.font.match_font("helveticaneue"), BUTTON_SIZE)

PARAMETERS_IMG = sp.misc.imread(os.path.join(os.path.split(sys.argv[0])[0], "parameters.png"))[:,:,[0,1,2]]

set_value("imgnr", args.img_nr)

@dynamic("path")
def get_path(imgnr):
    print("Loading image {}".format(imgnr))
    return args.input.format(imgnr)


show_last = False
last_grid = None
tabn = 0


PARAMETERS = [
    "CIRCLE_EDGE_THRESHOLD",
    "CIRCLE_THRESHOLD",
    "CIRCLE_GREEN_THRESHOLD",
    "LINE_EDGE_THRESHOLD",
    "PIECE_IMAGE_RATIO",
    "MAX_ROT",
    "AMOUNT_OF_ANGLES",
    "SIMILARITY_ANGLE",
    "LINE_TRIES",
    ]

EDITING = []

screen = pygame.display.set_mode(size, pygame.RESIZABLE)



def array_to_surface(arr):
    arr = np.clip(arr, 0, 255)
    if len(arr.shape) == 2:
        arr = np.array([np.zeros_like(arr), arr, np.zeros_like(arr)])
        arr = arr.transpose((1, 2, 0))
    arr = arr.transpose((1, 0, 2))


    surf = pygame.pixelcopy.make_surface(arr)
    return surf

s_save = BUTTON_FONT.render("Save, next", True, (0, ) * 3)
s_skip = BUTTON_FONT.render("Skip", True, (0, ) * 3)
s_prev = BUTTON_FONT.render("Previous", True, (0, ) * 3)
s_redo = BUTTON_FONT.render("Redo intersection finding", True, (0, ) * 3)
s_ferr = BUTTON_FONT.render("File not found", True, (255, 60, 60))

save_rect = pygame.Rect(0, 0, 0, 0)
skip_rect = pygame.Rect(0, 0, 0, 0)
prev_rect = pygame.Rect(0, 0, 0, 0)

def next_image():
    set_value("imgnr", get_value("imgnr") + 1)
    set_value("corners_to_remove", set())
    set_value("corners_to_add", set())

def save():
    board = get_value("othello_grid")

    if os.path.isfile(args.output_moves):
        f = open(args.output_moves, "r")
        content = literal_eval(f.read())
        f.close()
    else:
        content = {}

    content[get_value("imgnr")] = board.tolist()

    OUTPUT_FILE = open(args.output_moves, "w")

    OUTPUT_FILE.write(repr(content))
    OUTPUT_FILE.close()


    im = imread(get_value("path"))
    output_image = np.zeros((128, 128, 3))

    if im.shape[0] > im.shape[1]:
        new_size = int(im.shape[1] / im.shape[0] * 128)
        im_resized = imresize(im, (128, new_size))
    else:
        new_size = int(im.shape[0] / im.shape[1] * 128)
        im_resized = imresize(im, (new_size, 128))

    dx = output_image.shape[0] - im_resized.shape[0]
    dy = output_image.shape[1] - im_resized.shape[1]

    for x in range(im_resized.shape[0]):
        for y in range(im_resized.shape[1]):
            output_image[x + dx // 2, y + dy // 2] = im_resized[x, y]

    imsave(args.output_images.format(get_value("imgnr")), output_image)

def draw_controls(click):
    global EDITING
    if click is not None:
        i = (click[1] - MENU_HEIGHT) // OPTION_HEIGHT
        if click[1] > MENU_HEIGHT and i < len(PARAMETERS):
            EDITING = [i, ""]

    rendered_texts = []
    max_width = 0
    for i, text in enumerate(PARAMETERS):
        s_text = OPTION_FONT.render(text + " = ", True, (0, ) * 3)
        rendered_texts.append(s_text)
        if s_text.get_width() > max_width:
            max_width = s_text.get_width()

    for i, text in enumerate(PARAMETERS):
        s_text = rendered_texts[i]

        if len(EDITING) == 2 and EDITING[0] == i:
            s_value = OPTION_FONT.render(str(EDITING[1]) + "|", True, (0, ) * 3)
        else:
            s_value = OPTION_FONT.render(str(get_value(text)), True, (0, ) * 3)

        y = MENU_HEIGHT + i * OPTION_HEIGHT
        screen.blit(s_text, (max_width - s_text.get_width() + 10, y))
        screen.blit(s_value, (max_width + 10, y))
        y += OPTION_HEIGHT

def draw_main(click):
    global last_grid

    im = get_value("im")
    grid = get_value("othello_grid") if not show_last else last_grid

    if im.shape == (1, 1, 3):
        screen.blit(s_ferr, ((width - s_ferr.get_width()) / 2, MENU_HEIGHT))
    else:
        im_height = im.shape[1] / im.shape[0] * IMG_WIDTH
        im = imresize(im, (int(IMG_WIDTH), int(im_height)), interp="nearest")
        surf = array_to_surface(im)

        screen.blit(surf, (width / 2 - surf.get_width(), MENU_HEIGHT))

        board_size = PIECE_RADIUS * grid.shape[0] * 2
        board_rect = pygame.Rect(width / 2, MENU_HEIGHT, board_size, board_size)
        pygame.draw.rect(screen, (0, 255, 0), board_rect)

        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                value = int(grid[y, x])
                col = [(0, 200, 0), (255, 255, 255), (0, 0, 0)][value]
                px, py = (x + 0.5) * PIECE_RADIUS * 2, (y + 0.5) * PIECE_RADIUS * 2
                pygame.draw.circle(screen, col, (int(px + width / 2), int(py + MENU_HEIGHT)), PIECE_RADIUS)

    prev_rect = pygame.Rect(0, height - s_prev.get_height(), s_prev.get_width(), s_prev.get_height())
    save_rect = pygame.Rect(
            width - s_save.get_width(),
            height - s_save.get_height(),
            s_save.get_width(),
            s_save.get_height())
    skip_rect = pygame.Rect(
            width - s_skip.get_width(),
            height - s_skip.get_height() - save_rect.height,
            s_skip.get_width(),
            s_skip.get_height())

    pygame.draw.rect(screen, (255, 0, 0), prev_rect)
    pygame.draw.rect(screen, (0, 0, 255), skip_rect)
    pygame.draw.rect(screen, (0, 255, 0) if get_value("othello_grid").sum() > 0 else (125, 175, 125), save_rect)

    screen.blit(s_prev, (0, height - s_prev.get_height()))
    screen.blit(s_skip, (width - s_skip.get_width(), height - s_skip.get_height() - save_rect.height))
    screen.blit(s_save, (width - s_save.get_width(), height - s_save.get_height()))

    if click is not None:
        clickpos = click[:2]
        if prev_rect.collidepoint(clickpos):
            set_value("imgnr", get_value("imgnr") - 1)
            set_value("corners_to_remove", set())

        elif save_rect.collidepoint(clickpos) and get_value("othello_grid").sum() > 0:
            last_grid = get_value("othello_grid")

            save()
            next_image()

        elif skip_rect.collidepoint(clickpos):
            last_grid = get_value("othello_grid")

            next_image()

def draw_corners(click):
    im = get_value("im")
    intersections_all = get_value("intersections_all")
    intersections = get_value("intersections")
    corners = get_value("corners")

    im_width = im.shape[0]
    im_height = im.shape[1]
    im_new_height = im.shape[1] / im_width * IMG_WIDTH
    im = imresize(im, (int(IMG_WIDTH), int(im_new_height)), interp="nearest")
    surf = array_to_surface(im)
    screen.blit(surf, (0, MENU_HEIGHT))

    selecting_intersection = click is not None and click[0] < im.shape[1] and click[1] < im.shape[0] and click[2] == 1
    closest_to_click = None

    for intersection in intersections_all:
        is_removed = intersection not in intersections
        is_corner = intersection in corners
        col = (255, 255, 255)
        if is_removed:
            col = (128, 128, 128)
        elif is_corner:
            col = (255, 0, 0)
            if intersection in get_value("corners_to_add"):
                col = (255, 128, 255)
        elif intersection in get_value("corners_to_add"):
            col = (128, 128, 255)

        x, y = intersection
        x = x * IMG_WIDTH / im_width
        y = y * im_new_height / im_height + MENU_HEIGHT
        pygame.draw.circle(screen, col, (int(x), int(y)), 15)

        if selecting_intersection:
            dist = (click[0] - x) ** 2 + (click[1] - y) ** 2
            if closest_to_click == None or dist < closest_to_click[1]:
                closest_to_click = (intersection, dist)

    if selecting_intersection:
        print("Closest", closest_to_click[0], get_value("corners_to_add"))
        if closest_to_click[0] in get_value("corners_to_add"):
            print("removing")
            set_value("corners_to_add", get_value("corners_to_add") - { closest_to_click[0] })
        else:
            corners_to_remove = get_value("corners_to_remove")
            set_value("corners_to_remove", corners_to_remove ^ { closest_to_click[0] })

    redo_rect = pygame.Rect(0, height - s_redo.get_height(), s_redo.get_width(), s_redo.get_height())
    pygame.draw.rect(screen, (255, 0, 0), redo_rect)
    screen.blit(s_redo, (0, height - s_redo.get_height()))

    if click is not None:
        clickpos = click[:2]
        if redo_rect.collidepoint(clickpos):
            recalc("corners")

        if click[2] == 3: # Right click, add corner
            x = click[0] / IMG_WIDTH * im_width
            y = (click[1] - MENU_HEIGHT) / im_new_height * im_height

            set_value("corners_to_add", get_value("corners_to_add") | { (x, y) })


TABS = [(draw_main, "im"),
        (draw_controls, PARAMETERS_IMG),
        (draw_corners, "corners"),
        ]

while True:
    pygame.display.set_caption(os.path.basename(get_value("path")))

    click_position = None
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()
        if event.type == pygame.KEYDOWN:
            if event.scancode == 124: # Left
                tabn += 1
            elif event.scancode == 123: # Right
                tabn -= 1
            elif event.key == ord("l") and last_grid is not None: # Show last
                show_last = True
            elif len(EDITING) == 2:
                if  event.key >= ord("0") and \
                    event.key <= ord("9") or \
                    event.key == ord("-") or \
                    event.key == ord("."):
                    EDITING[1] += chr(event.key)
                if event.key == 8:
                    EDITING[1] = EDITING[1][:-1]
                if event.key == 13:
                    try:
                        value = literal_eval(EDITING[1])
                        print(PARAMETERS[EDITING[0]])
                        set_value(PARAMETERS[EDITING[0]], value)
                        EDITING = []
                    except:
                        pass
                if event.key == 27:
                    EDITING = []
            elif event.key >= ord("0") and \
                 event.key <= ord("9"):
                tabn = event.key - ord("0")
        if event.type == pygame.KEYUP:
            if event.key == ord("l"): # Hide last
                show_last = False

        if event.type == pygame.VIDEORESIZE:
            width, height = size = (event.w, event.h)
            surface = pygame.display.set_mode(size, pygame.RESIZABLE)


        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.pos[1] < MENU_HEIGHT:
                tabn = int(event.pos[0] / MENU_ITEM_WIDTH)
            else:
                click_position = (*event.pos, event.button)

    screen.fill((255, 255, 255))

    images = list(get_value("drawn"))
    tabn %= len(TABS) + len(images)

    MENU_ITEM_WIDTH = width / (len(TABS) + len(images))

    for i in range(len(TABS) + len(images)):
        if i == tabn:
            col = (200, ) * 3
        else:
            col = (100, ) * 3

        pygame.draw.rect(screen, col, pygame.Rect(i * MENU_ITEM_WIDTH, 0, MENU_ITEM_WIDTH, MENU_HEIGHT))
        pygame.draw.line(screen, (64, ) * 3, (i * MENU_ITEM_WIDTH, 0), (i * MENU_ITEM_WIDTH, MENU_HEIGHT - 1), 3)
        icon = None

        if i < len(TABS):
            f, im = TABS[i]
            if type(im) == str:
                im = get_value(im)
            try:
                if len(im) < MENU_PIC_WIDTH:
                    icon = imresize(im, (MENU_PIC_WIDTH, MENU_PIC_WIDTH), interp="nearest")
                else:
                    icon = imresize(im, (MENU_PIC_WIDTH, MENU_PIC_WIDTH))
            except Exception as e:
                print(e)
                exit()
            if i == tabn:
                f(click_position)
        else:
            im = images[i - len(TABS)]
            if len(im) < MENU_PIC_WIDTH:
                icon = imresize(im, (MENU_PIC_WIDTH, MENU_PIC_WIDTH), interp="nearest")
            else:
                icon = imresize(im, (MENU_PIC_WIDTH, MENU_PIC_WIDTH))

            if i == tabn:
                im = np.array(images[tabn - len(TABS)] * 256, dtype="uint64")
                im_height = im.shape[1] / im.shape[0] * IMG_WIDTH
                im = imresize(im, (int(IMG_WIDTH), int(im_height)), interp="nearest")
                surf = array_to_surface(im)

                screen.blit(surf, (0, MENU_HEIGHT))


        screen.blit(array_to_surface(icon),
                    (i * MENU_ITEM_WIDTH + (MENU_ITEM_WIDTH - MENU_PIC_WIDTH) / 2,
                    (MENU_HEIGHT - MENU_PIC_WIDTH) / 2)
                )


    pygame.display.flip()


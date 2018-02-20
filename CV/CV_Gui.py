import sys
import os
from ast import literal_eval

import pygame
import scipy as sp
import numpy as np
from scipy.misc import imresize

from dependencies import get_value, set_value, dynamic, placeholder
import CV_Vision

pygame.init()

width, height = size = 1080, 800

MENU_HEIGHT = 60
MENU_PIC_WIDTH = 40
MENU_ITEM_WIDTH = width / 12
IMG_WIDTH = width / 3
OPTION_HEIGHT = 24
PIECE_RADIUS = 24

BUTTON_SIZE = 60

OPTION_FONT = pygame.font.Font(pygame.font.match_font("helveticaneue"), OPTION_HEIGHT)
BUTTON_FONT = pygame.font.Font(pygame.font.match_font("helveticaneue"), BUTTON_SIZE)


IMG_IN_PATH = sys.argv[1] if len(sys.argv) == 2 else os.path.expanduser("~/Othello_bigger/img_{}.jpg")

set_value("imgnr", 0)

@dynamic("path")
def get_path(imgnr):
    return IMG_IN_PATH.format(imgnr)


tabn = 0

PARAMETERS_IMG = sp.misc.imread("parameters.png")[:,:,[0,1,2]]

PARAMETERS = [
    "CIRCLE_EDGE_THRESHOLD",
    "CIRCLE_THRESHOLD",
    "CIRCLE_GREEN_THRESHOLD",
    "LINE_EDGE_THRESHOLD",
    "MAX_ROT",
    "AMOUNT_OF_ANGLES",
    "SIMILARITY_ANGLE",
    ]

EDITING = []

screen = pygame.display.set_mode(size, pygame.RESIZABLE)



def array_to_surface(arr):
    if len(arr.shape) == 2:
        arr = np.array([np.zeros_like(arr), arr, np.zeros_like(arr)])
        arr = arr.transpose((1, 2, 0))
    arr = arr.transpose((1, 0, 2))

    arr = np.clip(arr, 0, 255)

    surf = pygame.pixelcopy.make_surface(arr)
    return surf

s_save = BUTTON_FONT.render("Save, next", True, (0, ) * 3)
s_skip = BUTTON_FONT.render("Skip", True, (0, ) * 3)
s_prev = BUTTON_FONT.render("Previous", True, (0, ) * 3)

save_rect = pygame.Rect(0, 0, 0, 0)
skip_rect = pygame.Rect(0, 0, 0, 0)
prev_rect = pygame.Rect(0, 0, 0, 0)

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
        screen.blit(s_text, (max_width - s_text.get_width(), y))
        screen.blit(s_value, (max_width, y))
        y += OPTION_HEIGHT

def draw_main(_):
    im = get_value("im")
    grid = get_value("othello_grid")


    im_height = im.shape[1] / im.shape[0] * IMG_WIDTH
    im = imresize(im, (int(IMG_WIDTH), int(im_height)), interp="nearest")
    surf = array_to_surface(im)

    screen.blit(surf, (width / 2 - surf.get_width(), MENU_HEIGHT))

    board_size = PIECE_RADIUS * grid.shape[0] * 2
    board_rect = pygame.Rect(width / 2, MENU_HEIGHT, board_size, board_size)
    pygame.draw.rect(screen, (0, 255, 0), board_rect)

    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            value = grid[y][x]
            if value == 0:
                continue
            col = (0, 0, 0) if value == 2 else (255, 255, 255)
            px, py = (x + 0.5) * PIECE_RADIUS * 2, (y + 0.5) * PIECE_RADIUS * 2
            pygame.draw.circle(screen, col, (int(px + width / 2), int(py + MENU_HEIGHT)), PIECE_RADIUS)

def draw_corners(click):
    im = get_value("im")
    corners = get_value("corners")
    corners_all = get_value("corners_all")

    im_width = im.shape[0]
    im_height = im.shape[1]
    im_new_height = im.shape[1] / im_width * IMG_WIDTH
    im = imresize(im, (int(IMG_WIDTH), int(im_new_height)), interp="nearest")
    surf = array_to_surface(im)
    screen.blit(surf, (0, MENU_HEIGHT))

    closest_to_click = None

    for corner in corners_all:
        is_removed = corner in corners
        col = (255, 255, 255) if is_removed else (128, 128, 128)

        x, y = corner
        x = x * IMG_WIDTH / im_width
        y = y * im_new_height / im_height
        pygame.draw.circle(screen, col, (int(x), int(y) + MENU_HEIGHT), 20)
        if click != None:
            dist = (click[0] - x) ** 2 + (click[1] - y) ** 2
            if closest_to_click == None or dist < closest_to_click[1]:
                closest_to_click = (corner, dist)

    if closest_to_click != None:
        # set_value("corners_to_remove", [(int(closest_to_click[0]), int(closest_to_click[1]))])

        corners_to_remove = get_value("corners_to_remove")
        set_value("corners_to_remove", corners_to_remove ^ set(closest_to_click))

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
            if len(EDITING) == 2:
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
        if event.type == pygame.VIDEORESIZE:
            width, height = size = (event.w, event.h)
            surface = pygame.display.set_mode(size,
                                              pygame.RESIZABLE)


        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.pos[1] < MENU_HEIGHT:
                tabn = int(event.pos[0] / MENU_ITEM_WIDTH)
            elif prev_rect.collidepoint(event.pos):
                set_value("imgnr", get_value("imgnr") - 1)
                set_value("corners_to_remove", set())
            elif save_rect.collidepoint(event.pos) and get_value("othello_grid").sum() > 0:
                print("Save??")
                set_value("imgnr", get_value("imgnr") + 1)
                set_value("corners_to_remove", set())
            elif skip_rect.collidepoint(event.pos):
                set_value("imgnr", get_value("imgnr") + 1)
                set_value("corners_to_remove", set())
            else:
                click_position = event.pos

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
            icon = imresize(im, (MENU_PIC_WIDTH, MENU_PIC_WIDTH), interp="nearest")
            if i == tabn:
                f(click_position)
        else:
            im = images[i - len(TABS)]
            icon = imresize(im, (MENU_PIC_WIDTH, MENU_PIC_WIDTH), interp="nearest")

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


    pygame.display.flip()


# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

from random import shuffle
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import os
import math

version = 5.0

save_dir = "/Users/liupeng/Documents/dl2tcc/experiment/process_dataset/version_{}".format(version)

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

characters_file = "../charset/final_common_3500.txt"
char_3000_file = save_dir + "/common_3000_{}.txt".format(version)
char_500_file = save_dir + "/common_500_{}.txt".format(version)

font_file = "../fontset/simkai.TTF"

charset = None


char_3000 = None
char_500 = None

image_size = 64
column_num = 10
row_num = 20
total_num = column_num * row_num

font_tempt = ImageFont.truetype(font_file, size=image_size)

with open(characters_file, "r") as f:
    charset = f.readlines()
    charset = [char.strip() for char in charset]
    print(len(charset))
    # print(charset)
    shuffle(charset)
    shuffle(charset)
    shuffle(charset)
    shuffle(charset)
    shuffle(charset)
    print("-----")
    print("-----")
    # print(charset)

    char_3000 = charset[0:3000]
    char_500 = charset[3000:len(charset)]

    # print(char_500)

print(len(char_3000))
print(len(char_500))


def draw_single_char(ch, font, canvas_size, x_offset, y_offset):
    # img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    img = Image.new("L", (canvas_size, canvas_size), 255)
    draw = ImageDraw.Draw(img)
    draw.text((x_offset, y_offset), ch, 0, font=font)
    return img

# def draw_img(c, font, canvas_size, x_offset, y_offset):
#     for

with open(char_3000_file, "w") as out:
    for c in char_3000:
        out.write(c + "\n")

with open(char_500_file, "w") as out:
    for c in char_500:
        out.write(c + "\n")

bg = None
for index, c in enumerate(char_500):
    print(str(index) + " :" + c)

    if index % total_num == 0:
        if index != 0:
            bg.save(os.path.join(save_dir, "test_samples_{}_{}.png".format(math.ceil(index / total_num), version)))
        bg = Image.new("L", (image_size * column_num, image_size * row_num), 255)

    y_offset_index = int(((index) % total_num) / column_num)
    x_offset_index = ((index) % total_num) % column_num

    font_img = draw_single_char(c, font_tempt, image_size, 0, 0)
    bg.paste(font_img, (x_offset_index * 64, y_offset_index * 64))

    if index == len(char_500) - 1:
        bg.save(os.path.join(save_dir, "test_samples_{}_{}.png".format(math.ceil(index / total_num), version)))

    print("({}, {})".format(x_offset_index, y_offset_index))




for c, index in enumerate(char_500):
    pass
# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

from imp import reload
import argparse
import sys
import numpy as np
import os
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import json
import collections


def load_charset(char_dir):
    with open(char_dir, 'r') as f:
        charset = f.readlines()
        charset = [char.strip() for char in charset]
        return charset


def draw_example(ch, font, canvas_size, x_offset, y_offset):
    img = Image.new("L", (canvas_size, canvas_size), 255)
    draw = ImageDraw.Draw(img)
    draw.text((x_offset, y_offset), ch, 0, font=font)
    return img



def font2imglabel(font, charset, char_size, canvas_size, x_offset, y_offset, sample_dir, label):
    char_font = ImageFont.truetype(font, size=char_size)

    count = 0
    for c in charset:
        img = draw_example(c, char_font, canvas_size, x_offset, y_offset)
        if img:
            img.save(os.path.join(sample_dir, "%d_%04d.jpg" % (label, count)))
            count += 1
            if count % 100 == 0:
                print("processed %d chars" % count)


# load_global_charset()
parser = argparse.ArgumentParser(description='Convert font to images and labels')
parser.add_argument('--char_dir', dest='char_dir', required=True, help='path of the characters')
parser.add_argument('--font', dest='font', required=True, help='path of the font')
parser.add_argument('--label', dest='label', type=int, default=0, help='label as the font')
parser.add_argument('--char_size', dest='char_size', type=int, default=256, help='character size')
parser.add_argument('--canvas_size', dest='canvas_size', type=int, default=256, help='canvas size')
parser.add_argument('--x_offset', dest='x_offset', type=int, default=0, help='x offset')
parser.add_argument('--y_offset', dest='y_offset', type=int, default=0, help='y_offset')
parser.add_argument('--sample_dir', dest='sample_dir', help='directory to save examples')
parser.add_argument('--shuffle', dest='shuffle', type=int, default=0, help='shuffle a charset before processings')

#
#
#
# parser.add_argument('--dst_font', dest='dst_font', required=True, help='path of the target font')
# parser.add_argument('--filter', dest='filter', type=int, default=0, help='filter recurring characters')
# parser.add_argument('--charset', dest='charset', type=str, default='CN',
#                     help='charset, can be either: CN, JP, KR or a one line file')
# parser.add_argument('--shuffle', dest='shuffle', type=int, default=0, help='shuffle a charset before processings')
# parser.add_argument('--char_size', dest='char_size', type=int, default=256, help='character size')
# parser.add_argument('--canvas_size', dest='canvas_size', type=int, default=256, help='canvas size')
# parser.add_argument('--x_offset', dest='x_offset', type=int, default=0, help='x offset')
# parser.add_argument('--y_offset', dest='y_offset', type=int, default=0, help='y_offset')
# parser.add_argument('--sample_dir', dest='sample_dir', help='directory to save examples')
# parser.add_argument('--label', dest='label', type=int, default=0, help='label as the prefix of examples')

args = parser.parse_args()

if __name__ == "__main__":

    if not os.path.exists(args.sample_dir):
        os.mkdir(args.sample_dir)

    charset = load_charset(args.char_dir)

    if args.shuffle:
        np.random.shuffle(charset)
    font2imglabel(args.font, charset, args.char_size,
             args.canvas_size, args.x_offset, args.y_offset,
             args.sample_dir, args.label)

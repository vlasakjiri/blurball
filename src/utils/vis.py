import os
import os.path as osp
from typing import Tuple
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import Center


def draw_trail(
    img,
    points,
    head_color=(255, 220, 80),
    tail_color=(255, 80, 20),
    max_thickness=5,
):
    if len(points) < 2:
        return img

    overlay = img.copy()
    num_segments = len(points) - 1
    for idx in range(num_segments):
        start = points[idx]
        end = points[idx + 1]
        if start is None or end is None:
            continue

        progress = (idx + 1) / max(num_segments, 1)
        color = tuple(
            int(tail_color[channel] * (1.0 - progress) + head_color[channel] * progress)
            for channel in range(3)
        )
        thickness = max(1, int(max_thickness * progress))

        cv2.line(overlay, start, end, color, thickness + 4, cv2.LINE_AA)
        cv2.line(img, start, end, color, thickness, cv2.LINE_AA)

    img = cv2.addWeighted(overlay, 0.18, img, 0.82, 0)
    head = points[-1]
    if head is not None:
        cv2.circle(img, head, max(4, max_thickness + 2), head_color, 2, cv2.LINE_AA)

    return img


def draw_frame(
    img_or_path,
    center: Center,
    color: Tuple,
    radius: int = 5,
    thickness: int = -1,
    angle=None,
    l=None,
):
    if osp.isfile(img_or_path):
        img = cv2.imread(img_or_path)
    else:
        img = img_or_path

    xy = center.xy
    visi = center.is_visible
    if visi:
        x, y = xy
        x, y = int(x), int(y)
        img = cv2.circle(img, (x, y), radius, color, thickness=thickness)
        if angle is not None:
            if l != 0:
                angle_rad = np.deg2rad(angle)
                x1 = int(x + l * np.cos(angle_rad))
                y1 = int(y + l * np.sin(angle_rad))
                x2 = int(x - l * np.cos(angle_rad))
                y2 = int(y - l * np.sin(angle_rad))
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return img


def gen_video(video_path, vis_dir, resize=1.0, fps=30.0, fourcc="mp4v"):
    fnames = os.listdir(vis_dir)
    fnames.sort()
    h, w, _ = cv2.imread(osp.join(vis_dir, fnames[0])).shape
    im_size = (int(w * resize), int(h * resize))
    fourcc = cv2.VideoWriter_fourcc(*fourcc)
    out = cv2.VideoWriter(video_path, fourcc, fps, im_size)

    for fname in tqdm(fnames):
        im_path = osp.join(vis_dir, fname)
        im = cv2.imread(im_path)
        if im is not None:
            im = cv2.resize(im, None, fx=resize, fy=resize)
            out.write(im)
        else:
            print("COuldn't read image")
            print(fname)

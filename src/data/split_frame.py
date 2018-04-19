
# coding: utf-8

import sys
import cv2
import numpy as np
from argparse import ArgumentParser
from subprocess import Popen
from pathlib import Path
from functools import cmp_to_key
from tqdm import tqdm


def parser():
    usage = f'Usage: python {__file__} [src_pdf] [series_dir] [-v volume] [--help]'
    argparser = ArgumentParser(usage=usage)
    argparser.add_argument('src_pdf', type=str,
                           help='source pdf file')
    argparser.add_argument('-v', '--volume', type=int,
                           help='the number of volume for source pdf')
    argparser.add_argument('series_dir', type=str,
                           help='directory of the series files')
    args = argparser.parse_args()
    return args.src_pdf, args.series_dir, args.volume


def pdf2jpegs(pdf_path: Path, series_path: Path, volume_num: int):
    if not pdf_path.is_file():
        raise Exception('The pdf file is not exist')
    dir = Path(f'{series_path}/original/volume{volume_num}')
    if not dir.is_dir():
        dir.mkdir(parents=True)
    command = f'pdfimages -j {pdf_path} {dir}/{volume_num}'
    popen = Popen(command, shell=True)
    popen.wait()


def apply_adaptive_threshold(image, radius=15, C=5):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 2 * radius + 1, C)


def find_external_contours(thresh):
    _, contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    external_num = hierarchy.shape[1] if hierarchy is not None else 0
    return contours[0:external_num]


def extract_rects_from_controus(contours, min_perimeter, max_perimeter):
    frames = []

    for contour in contours:
        frame = cv2.minAreaRect(contour)
        center, size, angle = frame
        # 縦・横が逆になっている場合、90度回転させる
        if angle < -45:
            size = tuple(reversed(size))
            angle = angle + 90
        w, h = size
        perimeter = 2 * (w + h)
        if min_perimeter < perimeter < max_perimeter and abs(angle) < 3.0 and 0.1 <= min(w, h) / max(w, h) <= 1.0:
            frames.append((center, (w + 2, h + 2), angle))  # パディングを加える
    return frames


def cmp_frame(tolerance):
    def _cmp(lhs, rhs):
        return (lhs > rhs) - (lhs < rhs)

    def _cmp_frame(lhs, rhs):
        if lhs[0] == rhs[0]:
            return 0
        x1, y1 = lhs[0]
        x2, y2 = rhs[0]
        if abs(x1 - x2) < tolerance:
            return _cmp(y1, y2)
        else:
            return _cmp(x2, x1)

    return _cmp_frame


def cut_frame(image, rect):
    center, size, angle = rect
    size = int(np.round(size[0])), int(np.round(size[1]))
    box = cv2.boxPoints(rect)
    M = cv2.getAffineTransform(np.float32(box[1:4]),  np.float32(
        [[0, 0], [size[0], 0], [size[0], size[1]]]))
    return cv2.warpAffine(image, M, size)


def cut_frames(image):
    height, width, ch = image.shape

    # 二値化
    thresh = apply_adaptive_threshold(image)

    # 一番外側の輪郭wだけを抽出
    contours = find_external_contours(thresh)

    # 抽出した輪郭からコマの四角形だけを取り出す
    min_perimeter, max_perimeter = (
        width + height) * 0.25,  (width + height) * 1.5
    rects = extract_rects_from_controus(contours, min_perimeter, max_perimeter)

    # 抽出した四角形をソートする
    tolerance = width / 3 if width < height else width / 6
    rects = sorted(rects, key=cmp_to_key(cmp_frame(tolerance)))
    # コマの部分の画像を切り出す
    frames = []
    for rect in rects:
        frame = cut_frame(image, rect)
        frames.append(frame)
    return frames


def split_page_to_frame(original_path: Path, series_path: Path):
    for jpg in tqdm(list(original_path.glob('*.jpg'))):

        volume_num, page_num = jpg.stem.split('-')
        page_path = series_path / f'frame/volume{volume_num}/{page_num}'

        if not page_path.is_dir():
            page_path.mkdir(parents=True)

        image = cv2.imread(str(jpg))
        frames = cut_frames(image)
        for i, frame in enumerate(frames):
            # dst_path = os.path.join(page_dir, str(i + 1) + ext)
            output_frame_path = str(page_path / f'{i}.jpg')
            cv2.imwrite(output_frame_path, frame)


if __name__ == '__main__':
    src_pdf, series_dir, volume_num = parser()

    def abs_path(path): return Path(path).resolve()
    pdf_path, series_path = abs_path(src_pdf), abs_path(series_dir)

    original_path = series_path / f'original/volume{volume_num}'
    pdf2jpegs(pdf_path, series_path, volume_num)

    split_page_to_frame(original_path, series_path)

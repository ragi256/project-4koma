
# coding: utf-8

import cv2
import numpy as np
from urllib.request import urlopen
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm

CASCADE_FILE_NAME = 'lbpcascade_animeface.xml'
CASCADE_URL = 'https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml'


def parser():
    usage = f'Usage: python {__file__} [series_dir] [-v volume | -a -all] [--help]'
    parser = ArgumentParser(usage=usage)
    parser.add_argument('series_dir', type=str,
                        help='source pdf file')
    parser.add_argument('-v', '--volume', type=int,
                        help='the number of volume to extract face')
    parser.add_argument('-a', '--all', action='store_true',
                        help='all volume to extract face')
    args = parser.parse_args()
    return args.series_dir, args.volume, args.all


def get_cascade_file(cascade_file_name=CASCADE_FILE_NAME, cascade_url=CASCADE_URL):
    cascade_file_path = 'models/' + cascade_file_name
    with urlopen(cascade_url) as response, open(cascade_file_path, 'w') as xml_file:
        xml = response.read().decode('utf-8')
        xml_file.write(xml)
        return cascade_file_path


def detect_faces(image, cascade):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor=1.1,
                                     minNeighbors=5,
                                     minSize=(24, 24))
    return faces


def extract(frame_path: Path, face_path: Path, cascade):
    face_id = 0
    jpgs = sorted(frame_path.glob('*/*.jpg'))  # ページ単位にディレクトリ切るべきかも？

    # initialize log
    image = cv2.imread(str(jpgs[0]), cv2.IMREAD_COLOR)
    detect_faces(image, cascade)

    for jpg_path in tqdm(jpgs):
        image = cv2.imread(str(jpg_path), cv2.IMREAD_COLOR)
        faces = detect_faces(image, cascade)
        for (x, y, w, h) in faces:
            face_image = image[y:y+h, x:x+w]
            output_path = face_path / f'{face_id}.jpg'
            cv2.imwrite(str(output_path), face_image)
            face_id += 1


if __name__ == '__main__':
    series_dir, volume_num, all_flag = parser()

    if volume_num and all_flag:
        raise "Don't use both of -v and -a. Use eather one."
    if not (volume_num or all_flag):
        raise 'Need option -v or -a.'

    cascade_file_path = get_cascade_file()
    cascade = cv2.CascadeClassifier(cascade_file_path)

    series_path = Path(series_dir).resolve()
    if all_flag:
        volumes = (series_path / 'frame').glob('volume*')
    else:
        volumes = [series_path / f'frame/volume{volume_num}']

    for volume in sorted(volumes):
        output_path = series_path / f'face/{volume.name}'
        output_path.mkdir(parents=True, exist_ok=True)

        print(volume)
        extract(volume, output_path, cascade)

#!/usr/bin/env python3
import requests

# Replace with your server's public IP address or domain name
url = "http://ec2-13-127-71-25.ap-south-1.compute.amazonaws.com/receive-data"
import logging as log
import sys
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter
import csv
from datetime import datetime
import cv2
import numpy as np
from openvino import Core, get_version
import json
import os

sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))
sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python/model_zoo'))
sys.path.append(str(Path(__file__).resolve().parents[2] / 'demos/python'))
sys.path.append(str(Path(__file__).resolve().parents[2] / 'model'))

from utils import crop
from landmarks_detector import LandmarksDetector
from face_detector import FaceDetector
from faces_database import FacesDatabase
from face_identifier import FaceIdentifier

import monitors
from helpers import resolution
from images_capture import open_images_capture


from model_api.models import OutputTransform
from model_api.performance_metrics import PerformanceMetrics
import time

import asyncio
import aiohttp
from datetime import datetime, timedelta
import time
from async_data_sender import get_token, add_people_in_lab, upload_lab_usage
from db_updater import check_and_download_new_profiles_once
log_file = "logs/app"+ str(datetime.now()) +".log"

# Create the directory if it doesn't exist
os.makedirs(os.path.dirname(log_file), exist_ok=True)

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin"
BASE_URL = "http://localhost:5000/api"
last_people_count_time = time.time()
session = None

ie = Core()

# Load Human Detection model
human_model_path = r"weights/person-detection-retail-0013.xml"

human_net = ie.read_model(model=human_model_path)
human_compiled_model = ie.compile_model(human_net, "CPU")
human_infer_request = human_compiled_model.create_infer_request()
human_input_layer = next(iter(human_net.inputs))
human_output_layer = next(iter(human_net.outputs))
token = None

if not os.path.exists(log_file):
    open(log_file, 'a').close()

log.basicConfig(
    format='%(asctime)s - [ %(levelname)s ] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=log.DEBUG,
    filename=log_file
)

DEVICE_KINDS = ['CPU', 'GPU', 'HETERO']

def build_argparser():
    parser = ArgumentParser()

    general = parser.add_argument_group('General')
    general.add_argument('-i', '--input',
                        # default="rtsp://admin:L281ACFE@172.18.6.109:554/cam/realmonitor?channel=1&subtype=0",
                        default=0,
                        help='Required. An input to process. The input must be a single image, '
                              'a folder of images, video file or camera id.')
    general.add_argument('--loop', default=False, action='store_true',
                         help='Optional. Enable reading the input in a loop.')
    general.add_argument('-o', '--output',
                         help='Optional. Name of the output file(s) to save.')
    general.add_argument('-limit', '--output_limit', default=1000, type=int,
                         help='Optional. Number of frames to store in output. '
                              'If 0 is set, all frames are stored.')
    general.add_argument('--output_resolution', default=None, type=resolution,
                         help='Optional. Specify the maximum output window resolution '
                              'in (width x height) format. Example: 1280x720. '
                              'Input frame size used by default.')
    general.add_argument('--no_show', action='store_true',
                         help="Optional. Don't show output.")
    general.add_argument('--crop_size', default=(0, 0), type=int, nargs=2,
                         help='Optional. Crop the input stream to this resolution.')
    general.add_argument('--match_algo', default='HUNGARIAN', choices=('HUNGARIAN', 'MIN_DIST'),
                         help='Optional. Algorithm for face matching. Default: HUNGARIAN.')
    general.add_argument('-u', '--utilization_monitors', default='', type=str,
                         help='Optional. List of monitors to show initially.')
    general.add_argument('--video_name', default='input_video', type=str,
                         help='Optional. Name of the input video file.')
    general.add_argument('--output_dir', default='output_dir', type=str,
                         help='Optional. Path to the output directory.')

    gallery = parser.add_argument_group('Faces database')
    gallery.add_argument('-fg', default="known_faces", help='Optional. Path to the face images directory.')
    gallery.add_argument('--run_detector', action='store_true',
                         help='Optional. Use Face Detection model to find faces '
                              'on the face images, otherwise use full images.')
    gallery.add_argument('--allow_grow', action='store_true',
                         help='Optional. Allow to grow faces gallery and to dump on disk. '
                              'Available only if --no_show option is off.')

    models = parser.add_argument_group('Models')
    models.add_argument('-m_fd',default="weights/face-detection-retail-0005.xml", type=Path,
                        help='Required. Path to an .xml file with Face Detection model.')
    models.add_argument('-m_lm', default= "weights/landmarks-regression-retail-0009.xml", type=Path, 
                        help='Required. Path to an .xml file with Facial Landmarks Detection model.')
    models.add_argument('-m_reid', default= "weights/facenet-20180408-102900.xml", type=Path,
                        help='Required. Path to an .xml file with Face Reidentification model.')
    models.add_argument('--fd_input_size', default=(0, 0), type=int, nargs=2,
                        help='Optional. Specify the input size of detection model for '
                             'reshaping. Example: 500 700.')

    infer = parser.add_argument_group('Inference options')
    infer.add_argument('-d_fd', default='CPU', choices=DEVICE_KINDS,
                       help='Optional. Target device for Face Detection model. '
                            'Default value is CPU.')
    infer.add_argument('-d_lm', default='CPU', choices=DEVICE_KINDS,
                       help='Optional. Target device for Facial Landmarks Detection '
                            'model. Default value is CPU.')
    infer.add_argument('-d_reid', default='CPU', choices=DEVICE_KINDS,
                       help='Optional. Target device for Face Reidentification '
                            'model. Default value is CPU.')
    infer.add_argument('-v', '--verbose', action='store_true',
                       help='Optional. Be more verbose.')
    infer.add_argument('-t_fd', metavar='[0..1]', type=float, default=0.6,
                       help='Optional. Probability threshold for face detections.')
    infer.add_argument('-t_id', metavar='[0..1]', type=float, default=0.2,
                       help='Optional. Cosine distance threshold between two vectors '
                            'for face identification.')
    infer.add_argument('-exp_r_fd', metavar='NUMBER', type=float, default=1.15,
                       help='Optional. Scaling ratio for bboxes passed to face recognition.')
    return parser

async def initialize_session():
    global session, token
    session = aiohttp.ClientSession()
    token = await get_token(session, ADMIN_USERNAME, ADMIN_PASSWORD)

async def send_lab_entry(face_name):
    
    global session, token
    if not session or not token:
        
        await initialize_session()
    
    current_time = datetime.now()
    await upload_lab_usage(session, token, face_name,int(face_name), "unknown", current_time, current_time + timedelta(minutes=1))
    
async def send_people_count(count):
    global session, token
    if not session or not token:
        await initialize_session()
    
    await add_people_in_lab(session, token, count)

def write_to_csv(face_identities, face_name):
    csv_path = "archi/output.csv"
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        for identity in face_identities:
            if identity.id != FaceIdentifier.UNKNOWN_ID:
                writer.writerow([current_time, identity.id, face_name])
                asyncio.create_task(send_lab_entry(face_name))

def detect_humans(frame):
    blob = cv2.resize(frame, (544, 320))
    blob = blob.transpose((2, 0, 1))
    blob = blob.reshape(1, 3, 320, 544)

    human_infer_request.infer(inputs={human_input_layer.any_name: blob})
    detections = human_infer_request.get_output_tensor(human_output_layer.index).data

    height, width = frame.shape[:2]
    humans = []
    for detection in detections[0][0]:
        confidence = detection[2]
        if confidence > 0.5:
            x_min = int(detection[3] * width)
            y_min = int(detection[4] * height)
            x_max = int(detection[5] * width)
            y_max = int(detection[6] * height)
            humans.append([x_min, y_min, x_max - x_min, y_max - y_min])
    # print(f"Detected Humans: {humans}")
    return humans

def send_data_to_server(data):
    # Convert NumPy int64 to Python int
    data = {k: int(v) if isinstance(v, np.int64) else v for k, v in data.items()}
    
    # converting dict to json
    json_data = json.dumps(data)
    response = requests.post(url, json=json_data)
    if response.status_code == 200:
        print("Data sent successfully")
    else:
        print(f"Failed to send data. Status code: {response.status_code}")

def get_present_individuals():
    present = set()
    today = datetime.now().strftime("%Y-%m-%d")
    
    with open("archi/output.csv", "r") as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0].startswith(today):
                present.add(row[1])
    
    return list(present)

class FrameProcessor:
    QUEUE_SIZE = 16

    def __init__(self, args):
        self.allow_grow = args.allow_grow and not args.no_show
        self.face_count = {} 

        log.info('OpenVINO Runtime')
        log.info('\tbuild: {}'.format(get_version()))
        core = Core()

        self.face_detector = FaceDetector(core, args.m_fd,
                                          args.fd_input_size,
                                          confidence_threshold=args.t_fd,
                                          roi_scale_factor=args.exp_r_fd)
        self.landmarks_detector = LandmarksDetector(core, args.m_lm)
        self.face_identifier = FaceIdentifier(core, args.m_reid,
                                              match_threshold=args.t_id,
                                              match_algo=args.match_algo)

        self.face_detector.deploy(args.d_fd)
        self.landmarks_detector.deploy(args.d_lm, self.QUEUE_SIZE)
        self.face_identifier.deploy(args.d_reid, self.QUEUE_SIZE)

        log.debug('Building faces database using images from {}'.format(args.fg))
        self.faces_database = FacesDatabase(args.fg, self.face_identifier,
                                            self.landmarks_detector,
                                            self.face_detector if args.run_detector else None, args.no_show)
        self.face_identifier.set_faces_database(self.faces_database)
        log.info('Database is built, registered {} identities'.format(len(self.faces_database)))
    
    def reBuildDatabase(self, args):
        self.faces_database = FacesDatabase(args.fg, self.face_identifier,
                                            self.landmarks_detector,
                                            self.face_detector if args.run_detector else None, args.no_show)
        self.face_identifier.set_faces_database(self.faces_database)
        log.info('Database is built, registered {} identities'.format(len(self.faces_database)))


    async def update_face_count(self, face_identities):
        for identity in face_identities:
            if identity.id != FaceIdentifier.UNKNOWN_ID:
                face_name = self.face_identifier.get_identity_label(identity.id)
                if face_name not in self.face_count:
                    self.face_count[face_name] = 1
                    await send_lab_entry(face_name)
                else:
                    self.face_count[face_name] += 1

                if self.face_count[face_name] == 50:
                    print(f"Face '{face_name}' has been recognized 50 times!")
                    write_to_csv(face_identities, face_name)
                    await send_lab_entry(face_name)
        # 

        
        
        



    async def process(self, frame):
        orig_image = frame.copy()

        rois = self.face_detector.infer((frame,))
        if self.QUEUE_SIZE < len(rois):
            log.warning('Too many faces for processing. Will be processed only {} of {}'
                        .format(self.QUEUE_SIZE, len(rois)))
            rois = rois[:self.QUEUE_SIZE]

        landmarks = self.landmarks_detector.infer((frame, rois))
        face_identities, unknowns = self.face_identifier.infer((frame, rois, landmarks))
        await self.update_face_count(face_identities)
        if self.allow_grow and len(unknowns) > 0:
            for i in unknowns:
                # This check is preventing asking to save half-images in the boundary of images
                if rois[i].position[0] == 0.0 or rois[i].position[1] == 0.0 or \
                    (rois[i].position[0] + rois[i].size[0] > orig_image.shape[1]) or \
                    (rois[i].position[1] + rois[i].size[1] > orig_image.shape[0]):
                    continue
                crop_image = crop(orig_image, rois[i])
                name = self.faces_database.ask_to_save(crop_image)
                if name:
                    id = self.faces_database.dump_faces(crop_image, face_identities[i].descriptor, name)
                    face_identities[i].id = id

        return [rois, landmarks, face_identities], face_identities

def center_crop(frame, crop_size):
    fh, fw, _ = frame.shape
    crop_size[0], crop_size[1] = min(fw, crop_size[0]), min(fh, crop_size[1])
    return frame[(fh - crop_size[1]) // 2 : (fh + crop_size[1]) // 2,
                 (fw - crop_size[0]) // 2 : (fw + crop_size[0]) // 2,
                 :]

async def main():
    args = build_argparser().parse_args()
    await initialize_session()
    time_between_sending = 300 # 300 seconds = 5 minutes

    cap = open_images_capture(args.input, args.loop)
    frame_processor = FrameProcessor(args)


    frame_num = 0
    metrics = PerformanceMetrics()
    presenter = None
    output_transform = None
    input_crop = None
    last_time = time.time()
    user = {} 
    global last_people_count_time
    if args.crop_size[0] > 0 and args.crop_size[1] > 0:
        input_crop = np.array(args.crop_size)
    elif not (args.crop_size[0] == 0 and args.crop_size[1] == 0):
        raise ValueError('Both crop height and width should be positive')
    video_writer = cv2.VideoWriter()

    while True:
        start_time = perf_counter()
        frame = cap.read()
        if frame is None:
            if frame_num == 0:
                raise ValueError("Can't read an image from the input")
            break

        

        humans = detect_humans(frame)
        # print(f"Detected Humans: {len(humans)}")

        # run this once every minute

        current_time = time.time()

        
        # if current_time - last_people_count_time >= time_between_sending:  # 300 seconds = 5 minutes
        #     await send_people_count(len(humans))
        #     last_people_count_time = current_time
        
        if current_time - last_people_count_time >= time_between_sending:  # 300 seconds = 5 minutes
            await send_people_count(len(humans))
            changes_detected = await check_and_download_new_profiles_once(args.fg)
            if changes_detected:
                frame_processor.reBuildDatabase(args)
            last_people_count_time = current_time

        


 

        if input_crop is not None:
            frame = center_crop(frame, input_crop)
        if frame_num == 0:
            output_transform = OutputTransform(frame.shape[:2], args.output_resolution)
            if args.output_resolution:
                output_resolution = output_transform.new_resolution
            else:
                output_resolution = (frame.shape[1], frame.shape[0])
            presenter = monitors.Presenter(args.utilization_monitors, 55,
                                           (round(output_resolution[0] / 4), round(output_resolution[1] / 8)))
            if args.output and not video_writer.open(args.output, cv2.VideoWriter_fourcc(*'MJPG'),
                                                     cap.fps(), output_resolution):
                raise RuntimeError("Can't open video writer")

        detections, face_identities = await frame_processor.process(frame)
        
        presenter.drawGraphs(frame)
        presenter.drawGraphs(frame)

        metrics.update(start_time, frame)


        frame_num += 1
        if video_writer.isOpened() and (args.output_limit <= 0 or frame_num <= args.output_limit):
            video_writer.write(frame)



    metrics.log_total()
    for rep in presenter.reportMeans():
        log.info(rep)

    await session.close()

if __name__ == '__main__':
    asyncio.run(main())
# #!/usr/bin/env python3
# import requests

# # Replace with your server's public IP address or domain name
# url = "http://ec2-13-127-71-25.ap-south-1.compute.amazonaws.com/receive-data"
# import logging as log
# import sys
# from argparse import ArgumentParser
# from pathlib import Path
# from time import perf_counter
# import csv
# from datetime import datetime
# import cv2
# import numpy as np
# from openvino import Core, get_version
# import json
# import os

# sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))
# sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python/model_zoo'))

# from utils import crop
# from landmarks_detector import LandmarksDetector
# from face_detector import FaceDetector
# from faces_database import FacesDatabase
# from face_identifier import FaceIdentifier

# import monitors
# from helpers import resolution
# from images_capture import open_images_capture

# from model_api.models import OutputTransform
# from model_api.performance_metrics import PerformanceMetrics

# from model_api.models import DetectionModel, OutputTransform
# from model_api.pipelines import get_user_config, AsyncPipeline
# from model_api.adapters import create_core, OpenvinoAdapter

# import monitors
# from helpers import resolution
# from images_capture import open_images_capture
# from drawing_utils import ColorPalette

# log_file = "/home/mridul/ERP_attendence_system/app/logs/app.log"

# # Create the directory if it doesn't exist
# os.makedirs(os.path.dirname(log_file), exist_ok=True)

# # If log_file does not exist, create it
# if not os.path.exists(log_file):
#     open(log_file, 'a').close()

# log.basicConfig(
#     format='%(asctime)s - [ %(levelname)s ] %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S',
#     level=log.DEBUG,
#     filename=log_file
# )


# DEVICE_KINDS = ['CPU', 'GPU', 'HETERO']




# def build_argparser():
#     parser = ArgumentParser()

#     general = parser.add_argument_group('General')
#     general.add_argument('-i', '--input',
#                         # default="rtsp://admin:cc/tv%409711@192.168.1.91:554/streaming/channels/502",
#                         default=0,
#                         help='Required. An input to process. The input must be a single image, '
#                               'a folder of images, video file or camera id.')
#     general.add_argument('--loop', default=False, action='store_true',
#                          help='Optional. Enable reading the input in a loop.')
#     general.add_argument('-o', '--output',
#                          help='Optional. Name of the output file(s) to save.')
#     general.add_argument('-limit', '--output_limit', default=1000, type=int,
#                          help='Optional. Number of frames to store in output. '
#                               'If 0 is set, all frames are stored.')
#     general.add_argument('--output_resolution', default=None, type=resolution,
#                          help='Optional. Specify the maximum output window resolution '
#                               'in (width x height) format. Example: 1280x720. '
#                               'Input frame size used by default.')
#     general.add_argument('--no_show', action='store_true',
#                          help="Optional. Don't show output.")
#     general.add_argument('--crop_size', default=(0, 0), type=int, nargs=2,
#                          help='Optional. Crop the input stream to this resolution.')
#     general.add_argument('--match_algo', default='HUNGARIAN', choices=('HUNGARIAN', 'MIN_DIST'),
#                          help='Optional. Algorithm for face matching. Default: HUNGARIAN.')
#     general.add_argument('-u', '--utilization_monitors', default='', type=str,
#                          help='Optional. List of monitors to show initially.')
#     general.add_argument('--video_name', default='input_video', type=str,
#                          help='Optional. Name of the input video file.')
#     general.add_argument('--output_dir', default='/home/mridul/ERP_attendence_system/output_dir', type=str,
#                          help='Optional. Path to the output directory.')

#     gallery = parser.add_argument_group('Faces database')
#     gallery.add_argument('-fg', default="/home/mridul/ERP_attendence_system/knownFaces", help='Optional. Path to the face images directory.')
#     gallery.add_argument('--run_detector', action='store_true',
#                          help='Optional. Use Face Detection model to find faces '
#                               'on the face images, otherwise use full images.')
#     gallery.add_argument('--allow_grow', action='store_true',
#                          help='Optional. Allow to grow faces gallery and to dump on disk. '
#                               'Available only if --no_show option is off.')

#     models = parser.add_argument_group('Models')
#     models.add_argument('-m_fd',default="/home/mridul/ERP_attendence_system/app/model/weights/face-detection-retail-0005.xml", type=Path,
#                         help='Required. Path to an .xml file with Face Detection model.')
#     models.add_argument('-m_lm', default= "/home/mridul/ERP_attendence_system/app/model/weights/landmarks-regression-retail-0009.xml", type=Path, 
#                         help='Required. Path to an .xml file with Facial Landmarks Detection model.')
#     models.add_argument('-m_reid', default= "/home/mridul/ERP_attendence_system/app/model/weights/facenet-20180408-102900.xml", type=Path,
#                         help='Required. Path to an .xml file with Face Reidentification model.')
    
#     models.add_argument('-m_od', default='/home/mridul/ERP_attendence_system/intel/person-detection-retail-0013/FP32/person-detection-retail-0013.xml', type=Path,
#                         help=' Path to an .xml file with Object Detection model.')
#     models.add_argument('--fd_input_size', default=(0, 0), type=int, nargs=2,
#                         help='Optional. Specify the input size of detection model for '
#                              'reshaping. Example: 500 700.')
    
#     models.add_argument('-at', '--architecture_type', help='Required. Specify model\'s architecture type.', default="ssd",
#                         type=str,  choices=('yolov3', 'yolov4', 'yolo-v2-tiny-vehicle-detection', 'mobilenet-ssd',
#                                                           'person-vehicle-bike-detection-crossroad-0078'))

#     infer = parser.add_argument_group('Inference options')
#     infer.add_argument('--layout', type=str, default=None,
#                                    help='Optional. Model inputs layouts. '
#                                         'Ex. NCHW or input0:NCHW,input1:NC in case of more than one input.')
#     infer.add_argument('-nireq', '--num_infer_requests', help='Optional. Number of infer requests',
#                             default=0, type=int)
#     infer.add_argument('-d_fd', default='CPU', choices=DEVICE_KINDS,
#                        help='Optional. Target device for Face Detection model. '
#                             'Default value is CPU.')
#     infer.add_argument('-d_lm', default='CPU', choices=DEVICE_KINDS,
#                        help='Optional. Target device for Facial Landmarks Detection '
#                             'model. Default value is CPU.')
#     infer.add_argument('-d_reid', default='CPU', choices=DEVICE_KINDS,
#                        help='Optional. Target device for Face Reidentification '
#                             'model. Default value is CPU.')
#     infer.add_argument('-v', '--verbose', action='store_true',
#                        help='Optional. Be more verbose.')
#     infer.add_argument('-t_fd', metavar='[0..1]', type=float, default=0.6,
#                        help='Optional. Probability threshold for face detections.')
#     infer.add_argument('-t_id', metavar='[0..1]', type=float, default=0.3,
#                        help='Optional. Cosine distance threshold between two vectors '
#                             'for face identification.')
#     infer.add_argument('-exp_r_fd', metavar='NUMBER', type=float, default=1.15,
#                        help='Optional. Scaling ratio for bboxes passed to face recognition.')
    
#     infer.add_argument('-d_od', default='CPU', choices=['CPU', 'GPU', 'HETERO'],
#                        help='Optional. Target device for Object Detection '
#                             'model. Default value is CPU.')
    
#     infer.add_argument('-t_od', metavar='[0..1]', type=float, default=0.5,
#                        help='Optional. Probability threshold for object detections.')

#     return parser


# def write_to_csv(face_identities, face_name):
#     csv_path = "/home/mridul/ERP_attendence_system/app/model/archi/output.csv"
#     current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     with open(csv_path, mode='a', newline='') as file:
#         writer = csv.writer(file)
#         for identity in face_identities:
#             if identity.id != FaceIdentifier.UNKNOWN_ID:
#                 writer.writerow([current_time, identity.id, face_name])
#                 try:
#                     send_data_to_server({"face_name": int(identity.id), "time": current_time, "name": face_name})
#                 except Exception as e:
#                     log.error(f"Failed to send data to server: {e}")

# def send_data_to_server(data):
#     # Convert NumPy int64 to Python int
#     data = {k: int(v) if isinstance(v, np.int64) else v for k, v in data.items()}
    
#     # converting dict to json
#     json_data = json.dumps(data)
#     response = requests.post(url, json=json_data)
#     if response.status_code == 200:
#         print("Data sent successfully")
#     else:
#         print(f"Failed to send data. Status code: {response.status_code}")


# def get_present_individuals():
#     present = set()
#     today = datetime.now().strftime("%Y-%m-%d")
    
#     with open("/home/mridul/ERP_attendence_system/app/model/archi/output.csv", "r") as file:
#         reader = csv.reader(file)
#         for row in reader:
#             if row[0].startswith(today):
#                 present.add(row[1])
    
#     return list(present)


# class CustomDetectionModel(DetectionModel):
#     def postprocess(self, outputs, meta):
#         print("Outputs type:", type(outputs))
#         print("Outputs shape:", np.array(outputs).shape if isinstance(outputs, (list, np.ndarray)) else "Not a list or array")
#         print("Outputs content:", outputs)

#         detections = []
#         if isinstance(outputs, dict):
#             # Handle dictionary output
#             for key, output in outputs.items():
#                 print(f"Processing output for key: {key}")
#                 print(f"Output shape: {output.shape}")
#                 self.process_output(output, detections)
#         elif isinstance(outputs, (list, np.ndarray)):
#             # Handle list or numpy array output
#             for output in outputs:
#                 print(f"Processing output with shape: {output.shape}")
#                 self.process_output(output, detections)
#         else:
#             print(f"Unexpected output type: {type(outputs)}")

#         return detections

#     def process_output(self, output, detections):
#         if len(output.shape) == 2:
#             # Assuming output shape is (N, 7) where N is the number of detections
#             for detection in output:
#                 if detection[2] > self.confidence_threshold:
#                     x_min, y_min, x_max, y_max = detection[3:7]
#                     detections.append(DetectionModel.Detection(x_min, y_min, x_max, y_max, detection[2], int(detection[1])))
#         elif len(output.shape) == 3:
#             # Assuming output shape is (1, N, 7)
#             for detection in output[0]:
#                 if detection[2] > self.confidence_threshold:
#                     x_min, y_min, x_max, y_max = detection[3:7]
#                     detections.append(DetectionModel.Detection(x_min, y_min, x_max, y_max, detection[2], int(detection[1])))
#         else:
#             print(f"Unexpected output shape: {output.shape}")

# class FrameProcessor:
#     QUEUE_SIZE = 16

#     def __init__(self, args):
#         self.allow_grow = args.allow_grow and not args.no_show
#         self.face_count = {} 

#         log.info('OpenVINO Runtime')
#         log.info('\tbuild: {}'.format(get_version()))
#         core = Core()

#         self.face_detector = FaceDetector(core, args.m_fd,
#                                           args.fd_input_size,
#                                           confidence_threshold=args.t_fd,
#                                           roi_scale_factor=args.exp_r_fd)
#         self.landmarks_detector = LandmarksDetector(core, args.m_lm)
#         self.face_identifier = FaceIdentifier(core, args.m_reid,
#                                               match_threshold=args.t_id,
#                                               match_algo=args.match_algo)

#         self.face_detector.deploy(args.d_fd)
#         self.landmarks_detector.deploy(args.d_lm, self.QUEUE_SIZE)
#         self.face_identifier.deploy(args.d_reid, self.QUEUE_SIZE)

#         log.debug('Building faces database using images from {}'.format(args.fg))
#         self.faces_database = FacesDatabase(args.fg, self.face_identifier,
#                                             self.landmarks_detector,
#                                             self.face_detector if args.run_detector else None, args.no_show)
#         self.face_identifier.set_faces_database(self.faces_database)
#         log.info('Database is built, registered {} identities'.format(len(self.faces_database)))

#         # Object detection model
#         object_detection_model_adapter = OpenvinoAdapter(
#             create_core(), 
#             args.m_od,  # Path to the object detection model
#             device=args.d_od,
#             plugin_config=get_user_config(args.d_od, '', None),
#             max_num_requests=args.num_infer_requests,
#             model_parameters={'input_layouts': args.layout}
#         )
        
#         configuration = {
#         'resize_type': None,
#         'mean_values': None,
#         'scale_values': None,
#         'reverse_input_channels': False,
#         'path_to_labels': None,
#         'confidence_threshold': 0.5,
#         'input_size': (600, 600),  # The CTPN specific
#         'num_classes': None,  # The NanoDet and NanoDetPlus specific
#         }
#         self.object_detector = CustomDetectionModel(object_detection_model_adapter, configuration)
#         self.object_detector_pipeline = AsyncPipeline(self.object_detector)

#         self.palette = ColorPalette(len(self.object_detector.labels) if self.object_detector.labels else 100)


#     def update_face_count(self, face_identities):
#         for identity in face_identities:
#             if identity.id != FaceIdentifier.UNKNOWN_ID:
#                 face_name = self.face_identifier.get_identity_label(identity.id)
#                 if face_name not in self.face_count:
#                     self.face_count[face_name] = 1
#                 else:
#                     self.face_count[face_name] += 1

#                 if self.face_count[face_name] == 50:
#                     print(f"Face '{face_name}' has been recognized 50 times!")
#                     write_to_csv(face_identities, face_name)  # Add this line


#     def process(self, frame):
#         orig_image = frame.copy()

#         rois = self.face_detector.infer((frame,))
#         if self.QUEUE_SIZE < len(rois):
#             log.warning('Too many faces for processing. Will be processed only {} of {}'
#                         .format(self.QUEUE_SIZE, len(rois)))
#             rois = rois[:self.QUEUE_SIZE]

#         landmarks = self.landmarks_detector.infer((frame, rois))
#         face_identities, unknowns = self.face_identifier.infer((frame, rois, landmarks))
#         self.update_face_count(face_identities) 
#         if self.allow_grow and len(unknowns) > 0:
#             for i in unknowns:
#                 # This check is preventing asking to save half-images in the boundary of images
#                 if rois[i].position[0] == 0.0 or rois[i].position[1] == 0.0 or \
#                     (rois[i].position[0] + rois[i].size[0] > orig_image.shape[1]) or \
#                     (rois[i].position[1] + rois[i].size[1] > orig_image.shape[0]):
#                     continue
#                 crop_image = crop(orig_image, rois[i])
#                 name = self.faces_database.ask_to_save(crop_image)
#                 if name:
#                     id = self.faces_database.dump_faces(crop_image, face_identities[i].descriptor, name)
#                     face_identities[i].id = id
#         self.object_detector_pipeline.submit_data(frame, 0, {'frame': frame, 'start_time': perf_counter()})
#         self.object_detector_pipeline.await_all()
#         detection_results = self.object_detector_pipeline.get_result(0)
#         objects = detection_results[0] if detection_results else []

#         return [rois, landmarks, face_identities], face_identities, objects


# def draw_detections(frame, frame_processor, detections, output_transform):
#     size = frame.shape[:2]
#     frame = output_transform.resize(frame)
#     for roi, landmarks, identity in zip(*detections):
#         text = frame_processor.face_identifier.get_identity_label(identity.id)
#         if identity.id != FaceIdentifier.UNKNOWN_ID:
#             text += ' %.2f%%' % (100.0 * (1 - identity.distance))

#         xmin = max(int(roi.position[0]), 0)
#         ymin = max(int(roi.position[1]), 0)
#         xmax = min(int(roi.position[0] + roi.size[0]), size[1])
#         ymax = min(int(roi.position[1] + roi.size[1]), size[0])
#         xmin, ymin, xmax, ymax = output_transform.scale([xmin, ymin, xmax, ymax])
#         cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 220, 0), 2)

#         for point in landmarks:
#             x = xmin + output_transform.scale(roi.size[0] * point[0])
#             y = ymin + output_transform.scale(roi.size[1] * point[1])
#             cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 255), 2)
#         textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
#         cv2.rectangle(frame, (xmin, ymin), (xmin + textsize[0], ymin - textsize[1]), (255, 255, 255), cv2.FILLED)
#         cv2.putText(frame, text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

#     return frame

# def draw_detections(frame, detections, palette, labels, output_transform):
#     for detection in detections:
#         class_id = int(detection.id)
#         color = palette[class_id]
#         det_label = labels[class_id] if labels and len(labels) >= class_id else '#{}'.format(class_id)
#         xmin, ymin, xmax, ymax = detection.get_coords()
#         xmin, ymin, xmax, ymax = output_transform.scale([xmin, ymin, xmax, ymax])
#         cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
#         cv2.putText(frame, '{} {:.1%}'.format(det_label, detection.score),
#                     (xmin, ymin - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
#     return frame

# def center_crop(frame, crop_size):
#     fh, fw, _ = frame.shape
#     crop_size[0], crop_size[1] = min(fw, crop_size[0]), min(fh, crop_size[1])
#     return frame[(fh - crop_size[1]) // 2 : (fh + crop_size[1]) // 2,
#                  (fw - crop_size[0]) // 2 : (fw + crop_size[0]) // 2,
#                  :]


# def main():
#     args = build_argparser().parse_args()
#     core = Core()
#     cap = open_images_capture(args.input, args.loop)
#     detector_person = Detector(core, args.model_od,
#                                device=args.device,
#                                label_class=args.person_label)
#     frame_processor = FrameProcessor(args)

#     frame_num = 0
#     metrics = PerformanceMetrics()
#     presenter = None
#     output_transform = None
#     video_writer = cv2.VideoWriter()

#     while True:
#         start_time = perf_counter()
#         frame = cap.read()
#         if frame is None:
#             if frame_num == 0:
#                 raise ValueError("Can't read an image from the input")
#             break
        
#         if frame_num == 0:
#             output_transform = OutputTransform(frame.shape[:2], args.output_resolution)
#             bboxes = detector_person.detect(frame)
#             if args.output_resolution:
#                 output_resolution = output_transform.new_resolution
#             else:
#                 output_resolution = (frame.shape[1], frame.shape[0])
#             presenter = monitors.Presenter(args.utilization_monitors, 55,
#                                            (round(output_resolution[0] / 4), round(output_resolution[1] / 8)))
#             if args.output and not video_writer.open(args.output, cv2.VideoWriter_fourcc(*'MJPG'),
#                                                      cap.fps(), output_resolution):
#                 raise RuntimeError("Can't open video writer")

#         detections, face_identities, objects = frame_processor.process(frame)
        
#         presenter.drawGraphs(frame)
#         frame = draw_detections(frame, frame_processor.face_detector, frame_processor.palette, frame_processor.face_identifier.face_identifier.get_identity_label, output_transform)
#         frame = draw_detections(frame, objects, frame_processor.palette, frame_processor.object_detector.labels, output_transform)
#         metrics.update(start_time, frame)

#         frame_num += 1
#         if video_writer.isOpened() and (args.output_limit <= 0 or frame_num <= args.output_limit):
#             video_writer.write(frame)

#         if not args.no_show:
#             cv2.imshow('Face recognition and object detection demo', frame)
#             key = cv2.waitKey(1)
#             # Quit
#             if key in {ord('q'), ord('Q'), 27}:
#                 break
#             presenter.handleKey(key)

#     metrics.log_total()
#     for rep in presenter.reportMeans():
#         log.info(rep)


# if __name__ == '__main__':
#     sys.exit(main() or 0)
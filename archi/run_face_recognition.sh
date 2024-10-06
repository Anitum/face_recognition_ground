#!/bin/bash

# Set directories
VIDEO_DIR="/home/mridul/ERP_attendence_system/stock_videos"
MODEL_DIR="/home/mridul/ERP_attendence_system/face_recognition/open_model_zoo/demos/face_recognition_demo/python"  # Replace with the actual path to the 'intel' directory
FACE_GALLERY="/home/mridul/ERP_attendence_system/knownFaces"

# Face detection models
FD_MODELS=(
    "intel/face-detection-retail-0004/FP32/face-detection-retail-0004.xml"
    "intel/face-detection-retail-0005/FP32/face-detection-retail-0005.xml"
    "intel/face-detection-adas-0001/FP32/face-detection-adas-0001.xml"
)

# Landmark regression model
LM_MODEL="intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml"

# Face reidentification models
REID_MODELS=(
    "public/face-recognition-resnet100-arcface-onnx/FP32/face-recognition-resnet100-arcface-onnx.xml"
    "public/facenet-20180408-102900/FP32/facenet-20180408-102900.xml"

)

# Loop through all videos
for video in "$VIDEO_DIR"/*.mp4; do
    # Loop through face detection models
    for fd_model in "${FD_MODELS[@]}"; do
        # Loop through face reidentification models
        for reid_model in "${REID_MODELS[@]}"; do
            echo "Processing video: $video"
            echo "Face detection model: $fd_model"
            echo "Face reidentification model: $reid_model"
            
            python ./face_recognition_demo.py \
                -i "$video" \
                -m_fd "$MODEL_DIR/$fd_model" \
                -m_lm "$MODEL_DIR/$LM_MODEL" \
                -m_reid "$MODEL_DIR/$reid_model" \
                --verbose \
                -fg "$FACE_GALLERY"
            
            echo "Finished processing"
            echo "------------------------"
        done
    done
done

#!/bin/bash

echo "Downloading saved models..."

mkdir -p saved_models/lane_segmentation_model
wget -O saved_models/lane_segmentation_model/best_yolo11_lane_segmentation.pt "https://drive.google.com/file/d/1oRB1lRfrE9xLkG089mOYAMgnodIZsFWk/view?usp=sharing"

mkdir -p saved_models/object_detection_model
wget -O saved_models/object_detection_model/yolo11m-seg.pt "https://drive.google.com/file/d/1oSvavrDpkzI8OH6F2XQfZxfM8Bf5uX6D/view?usp=sharing"

mkdir -p saved_models/regression_model
wget -O saved_models/regression_model/model.ckpt.data-00000-of-00001 "https://drive.google.com/file/d/1FkNwBdHAoirMd7ZdPGlpZzczT1Z4yOy_/view?usp=sharing"
wget -O saved_models/regression_model/model.ckpt.index "https://drive.google.com/file/d/18OPR3FuO9ppny1VXXqAubOTDgF5wu4XD/view?usp=sharing"
wget -O saved_models/regression_model/model.ckpt.meta "https://drive.google.com/file/d/1vh85gatTV1AEuHW_tr9FOD8obod1ZFCb/view?usp=sharing"

echo "✅ All models downloaded."

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import cv2
import numpy as np
import colorsys
import time
from typing import List, Tuple
import concurrent.futures
import tensorflow.compat.v1 as tf
from ultralytics import YOLO
from src.models import model
from subprocess import call

tf.disable_v2_behavior()

# ================================
# 🧠 Steering Angle Prediction
# ================================
class SteeringAnglePredictor:
    def __init__(self, model_path: str):
        self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, model_path)

    def predict_angle(self, image) -> float:
        with self.sess.as_default():
            rad = model.y.eval(feed_dict={model.x: [image], model.keep_prob: 1.0})[0][0]
            return rad * 180.0 / np.pi

# ===============================
# 🧠 YOLO Segmentation
# ===============================
class ImageSegmentation:
    def __init__(self, lane_model_path: str, object_model_path: str):
        self.lane_model = YOLO(lane_model_path)
        self.object_model = YOLO(object_model_path)
        self.colors = self._generate_colors(len(self.object_model.names))

    @staticmethod
    def _generate_colors(num_classes: int) -> List[Tuple[int, int, int]]:
        colors = []
        for i in range(num_classes):
            hue = i / num_classes
            rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
            colors.append(tuple(int(x * 255) for x in rgb))
        return colors

    def process(self, img: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        overlay = img.copy()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_lane = executor.submit(self.lane_model.predict, img, conf=0.5)
            future_object = executor.submit(self.object_model.predict, img, conf=0.5)
            lane_results = future_lane.result()
            object_results = future_object.result()

        self._draw_lane_overlay(overlay, lane_results)
        self._draw_object_overlay(overlay, object_results)
        return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    def _draw_lane_overlay(self, overlay: np.ndarray, lane_results):
        for result in lane_results:
            if result.masks is None:
                continue
            for mask in result.masks.xy:
                points = np.int32([mask])
                cv2.fillPoly(overlay, points, (144, 238, 144))  # Light green

    def _draw_object_overlay(self, overlay: np.ndarray, object_results):
        for result in object_results:
            if result.masks is None:
                continue
            for mask, box in zip(result.masks.xy, result.boxes):
                class_id = int(box.cls[0])
                color = self.colors[class_id]
                points = np.int32([mask])
                cv2.fillPoly(overlay, points, color)

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                label = f"{self.object_model.names[class_id]}: {box.conf[0]:.2f}"
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(overlay, (x1, y1 - label_h - 5), (x1 + label_w, y1), color, -1)
                cv2.putText(overlay, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

# =================================
# 🚘 Self-Driving Car Simulator
# =================================
class SelfDrivingCarSimulator:
    def __init__(self, steering_model, segmentation_model, data_path, img_path):
        self.steering_model = steering_model
        self.segmentation_model = segmentation_model
        self.data_path = data_path
        self.img = cv2.imread(img_path, 0)
        if self.img is None:
            raise FileNotFoundError(f"Steering wheel image not found: {img_path}")
        self.smoothed_angle = 0
        self.rows, self.cols = self.img.shape
        os.makedirs("output_frames", exist_ok=True)

    def start_simulation(self, frame_interval: float = 1 / 30):
        i = 0
        while True:
            start_time = time.time()
            image_path = f"{self.data_path}/{i}.jpg"
            full_image = cv2.imread(image_path)
            if full_image is None:
                print(f"Image {image_path} not found, ending simulation.")
                break

            resized_image = cv2.resize(full_image[-150:], (200, 66)) / 255.0

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_steering = executor.submit(self.steering_model.predict_angle, resized_image)
                future_segmentation = executor.submit(self.segmentation_model.process, full_image)
                degrees = future_steering.result()
                segmented_image = future_segmentation.result()

            self._update_display(degrees, segmented_image, full_image)
            i += 1

            delay = frame_interval - (time.time() - start_time)
            if delay > 0:
                time.sleep(delay)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    def _update_display(self, degrees, segmented_image, full_image):
        print(f"Predicted steering angle: {degrees:.2f} degrees")

        if self.smoothed_angle == 0:
            self.smoothed_angle = degrees
        else:
            diff = degrees - self.smoothed_angle
            self.smoothed_angle += 0.2 * (abs(diff) ** (2.0 / 3.0)) * (diff / abs(diff))

        M = cv2.getRotationMatrix2D((self.cols / 2, self.rows / 2), -self.smoothed_angle, 1)
        rotated_steering = cv2.warpAffine(self.img, M, (self.cols, self.rows))

        # Display all
        cv2.imshow("Original Frame", full_image)
        cv2.imshow("Segmented View", segmented_image)
        cv2.imshow("Steering Wheel", rotated_steering)

        # Resize for hconcat
        target_shape = (480, 270)
        full_image_resized = cv2.resize(full_image, target_shape)
        segmented_resized = cv2.resize(segmented_image, target_shape)
        rotated_resized = cv2.cvtColor(cv2.resize(rotated_steering, target_shape), cv2.COLOR_GRAY2BGR)

        combined = cv2.hconcat([full_image_resized, segmented_resized, rotated_resized])
        frame_id = len(os.listdir("output_frames"))
        cv2.imwrite(f"output_frames/frame_{frame_id:03d}.jpg", combined)

# =================================
# 🚀 Entry Point
# =================================
if __name__ == "__main__":
    steering_model_path = "saved_models/regression_model/model.ckpt"
    lane_model_path = "saved_models/lane_segmentation_model/best_yolo11_lane_segmentation.pt"
    object_model_path = "saved_models/object_detection_model/yolo11m-seg.pt"
    data_folder = "data/driving_dataset/data"
    steering_image_path = "data/steer-wheel.png"

    steering_model = SteeringAnglePredictor(steering_model_path)
    segmentation_model = ImageSegmentation(lane_model_path, object_model_path)

    simulator = SelfDrivingCarSimulator(
        steering_model=steering_model,
        segmentation_model=segmentation_model,
        data_path=data_folder,
        img_path=steering_image_path
    )

    simulator.start_simulation()
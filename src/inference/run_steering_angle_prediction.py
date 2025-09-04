import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
from subprocess import call

from src.models import model  # Your TensorFlow model definition
# from ultralytics import YOLO  # Optional: for segmentation if needed

tf.disable_v2_behavior()

# ===============================
# Steering Prediction from Model
# ===============================
class SteeringAnglePredictor:
    def __init__(self, model_path):
        self.session = tf.InteractiveSession()
        self.saver = tf.train.Saver()
        self.saver.restore(self.session, model_path)
        self.smoothed_angle = 0

    def predict_angle(self, image):
        # Missing keep_prob value; commonly 1.0 for inference
        degrees = self.session.run(model.y, feed_dict={model.x: [image], model.keep_prob: 1.0})
        return float(degrees[0][0])

    def smooth_angle(self, predicted_angle):
        if self.smoothed_angle == 0:
            self.smoothed_angle = predicted_angle
        else:
            self.smoothed_angle += 0.2 * (abs(predicted_angle - self.smoothed_angle) / abs(predicted_angle)) * (predicted_angle - self.smoothed_angle)
        return self.smoothed_angle

    def close(self):
        self.session.close()

# ============================
# Driving Simulator Component
# ============================
class DrivingSimulator:
    def __init__(self, predictor, data_dir, steering_image_path, is_windows=False):
        self.predictor = predictor
        self.data_dir = data_dir
        self.steering_image = cv2.imread(steering_image_path, 0)
        self.is_windows = is_windows
        self.rows, self.cols = self.steering_image.shape

    def start_simulation(self):
        i = 0
        while cv2.waitKey(10) != ord('q'):
            image_path = os.path.join(self.data_dir, f"{i}.jpg")
            full_image = cv2.imread(image_path)
            if full_image is None:
                print(f"Image {image_path} not found.")
                break

            # Resize and normalize image
            resized_image = cv2.resize(full_image[-150:], (200, 66)) / 255.0

            predicted_angle = self.predictor.predict_angle(resized_image)
            smoothed_angle = self.predictor.smooth_angle(predicted_angle)

            if not self.is_windows:
                call("clear")
            print(f"Predicted Steering Angle: {predicted_angle:.2f} degrees")

            self.display_frames(full_image, smoothed_angle)
            i += 1

        cv2.destroyAllWindows()

    def display_frames(self, full_image, smoothed_angle):
        cv2.imshow("frame", full_image)

        # Rotate steering wheel image
        M = cv2.getRotationMatrix2D((self.cols / 2, self.rows / 2), -smoothed_angle, 1)
        rotated_steering = cv2.warpAffine(self.steering_image, M, (self.cols, self.rows))
        cv2.imshow("steering wheel", rotated_steering)

# ===========================
# Main Entry Point
# ===========================
if __name__ == "__main__":
    model_path = "saved_models/regression_model/model.ckpt"
    data_dir = "data/driving_dataset/data"
    steering_image_path = "data/steer-wheel.png"

    is_windows = os.name == 'nt'
    predictor = SteeringAnglePredictor(model_path)
    simulator = DrivingSimulator(predictor, data_dir, steering_image_path, is_windows)

    try:
        simulator.start_simulation()
    finally:
        predictor.close()
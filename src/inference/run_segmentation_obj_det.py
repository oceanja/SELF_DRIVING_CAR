import os
import cv2
import numpy as np
import colorsys
from typing import List, Tuple
from ultralytics import YOLO  # Assuming you're using YOLOv8 or similar

class SegmentationVisualizer:
    def __init__(self, model_path_1: str, model_path_2: str, conf_threshold: float):
        self.model_1 = YOLO(model_path_1)  # Lane segmentation model
        self.model_2 = YOLO(model_path_2)  # Object detection model
        self.conf_threshold = conf_threshold

        self.class_names_1 = self.model_1.names
        self.class_names_2 = self.model_2.names
        self.colors_2 = self._generate_colors(len(self.class_names_2))
        self.lane_color = (144, 238, 144)  # Light green for lanes

    def _generate_colors(self, num_classes: int) -> List[Tuple[int, int, int]]:
        colors = []
        for i in range(num_classes):
            hue = i / num_classes
            rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
            colors.append(tuple(int(c * 255) for c in rgb))
        return colors

    def _get_lane_color(self, class_id: int) -> Tuple[int, int, int]:
        class_name = self.class_names_1[class_id]
        if class_name.lower() == "lane1":
            return self.lane_color
        else:
            return (0, 255, 0)

    def process_image(self, img: np.ndarray, alpha: float = 0.5,
                      show_labels: bool = True, show_conf: bool = True,
                      show_boxes: bool = True, box_thresh: float = 0.5) -> np.ndarray:
        overlay = img.copy()

        results_1 = self.model_1.predict(img, conf=self.conf_threshold)
        overlay = self._apply_model_results(
            overlay, results_1, self.class_names_1, self.colors_2,
            alpha, box_thresh, show_labels, show_conf, show_boxes, lane_model=True
        )

        results_2 = self.model_2.predict(img, conf=self.conf_threshold)
        final_img = self._apply_model_results(
            overlay, results_2, self.class_names_2, self.colors_2,
            alpha, box_thresh, show_labels, show_conf, show_boxes, lane_model=False
        )

        return final_img

    def _apply_model_results(self, overlay, results, class_names, colors,
                             alpha, box_thresh, show_labels, show_conf,
                             show_boxes, lane_model=False):
        for result in results:
            if result.masks is None:
                continue

            for mask, box in zip(result.masks.xy, result.boxes):
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                if confidence < box_thresh:
                    continue

                class_name = class_names[class_id]
                color = self._get_lane_color(class_id) if lane_model else colors[class_id]

                points = np.int32([mask])
                cv2.fillPoly(overlay, points, color)

                if show_boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

                if show_labels or show_conf:
                    label = []
                    if show_labels:
                        label.append(class_name)
                    if show_conf:
                        label.append(f"{confidence:.2f}")
                    label_text = " ".join(label)
                    (label_w, label_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    label_x, label_y = (x1, y1 - 10)
                    cv2.rectangle(overlay, (label_x, label_y - label_h), (label_x + label_w, label_y), color, -1)
                    cv2.putText(overlay, label_text, (label_x, label_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return cv2.addWeighted(overlay, alpha, overlay, 1 - alpha, 0)

    def display_images_with_segmentation(self, input_folder, display_time=100):
        image_files = [img for img in os.listdir(input_folder) if img.lower().endswith((".png", ".jpg", ".jpeg"))]
        image_files.sort(key=lambda x: int(x.split('.')[0]))

        for image_file in image_files:
            image_path = os.path.join(input_folder, image_file)
            image = cv2.imread(image_path)
            if image is not None:
                segmented_image = self.process_image(image)
                cv2.imshow('Original Image Viewer', image)
                cv2.imshow('Segmented Image Viewer', segmented_image)
                print(f"Displaying {image_file}")

                if cv2.waitKey(display_time) == ord('q'):
                    break
            else:
                print(f"Failed to load {image_file}")

        cv2.destroyAllWindows()

# Run the visualizer
if __name__ == "__main__":
    visualizer = SegmentationVisualizer(
        model_path_1="saved_models/lane_segmentation_model/best_yolo11_lane_segmentation.pt",
        model_path_2="saved_models/object_detection_model/yolo11m-seg.pt",
        conf_threshold=0.5
    )
    input_folder = 'data/driving_dataset'
    visualizer.display_images_with_segmentation(input_folder, display_time=100)
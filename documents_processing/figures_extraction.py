from pdf2image import convert_from_path
import numpy as np
from collections import defaultdict
import os
from typing import Dict, List
import cv2
import supervision as sv  # pip install supervision
from ultralytics import YOLO
from PIL import Image
import base64
import fitz
from io import BytesIO
from ultralytics import YOLOv10


needed_features = ["Table", "Picture"]

yolo_model_name = "yolov10x_best.pt"
model_path = os.path.join("models", yolo_model_name)

def _load_yolo_model(model_name: str):
    if not os.path.exists(model_path):
        os.makedirs("models", exist_ok=True)
        os.system(f"wget https://github.com/moured/YOLOv10-Document-Layout-Analysis/releases/download/doclaynet_weights/{model_name}")
        os.system(f"mv {model_name} {model_path}")

    return YOLO(model_path)

image_segmentation_model = _load_yolo_model(yolo_model_name)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def _calculate_area(bbox):
    """Calculate the area of the bounding box."""
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)


def _calculate_overlap(bbox1, bbox2):
    """Calculate the area of overlap between two bounding boxes."""
    x1_overlap = max(bbox1[0], bbox2[0])
    y1_overlap = max(bbox1[1], bbox2[1])
    x2_overlap = min(bbox1[2], bbox2[2])
    y2_overlap = min(bbox1[3], bbox2[3])

    if x1_overlap < x2_overlap and y1_overlap < y2_overlap:
        overlap_area = (x2_overlap - x1_overlap) * (y2_overlap - y1_overlap)
    else:
        overlap_area = 0

    return overlap_area


def filter_bboxes(bboxes, threshold=0.8):
    """Filter bounding boxes to remove those contained in others."""
    filtered_bboxes = []
    for i, bbox1 in enumerate(bboxes):
        contained = False
        area1 = _calculate_area(bbox1)

        for j, bbox2 in enumerate(bboxes):
            if i != j:
                # area2 = _calculate_area(bbox2)
                overlap_area = _calculate_overlap(bbox1, bbox2)

                # Check if bbox1 is more than 80% contained in bbox2
                if overlap_area / area1 > threshold:
                    contained = True
                    break

        if not contained:
            filtered_bboxes.append(i)

    return filtered_bboxes


def _remove_faces_png(image_data):
    """
    Function to remove faces from an image and return the modified image data.
    """

    # Convert image data to numpy array and read it
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Remove faces
    for x, y, w, h in faces:
        img[y : y + h, x : x + w] = np.zeros((h, w, 3), np.uint8)

    # Convert the modified image back to PNG bytes
    is_success, buffer = cv2.imencode(".png", img)
    output_image_data = BytesIO(buffer).getvalue()

    return output_image_data


def extract_figures(
    saved_pages_images_path: os.PathLike,
    pdf_file_path: os.PathLike,
    pdf_saved_name: str,
    min_confidence: float = 0.2,
) -> Dict[str, List[str]]:

    one_file_saved_images_path = os.path.join(
        saved_pages_images_path, pdf_saved_name.replace(".pdf", "")
    )
    os.makedirs(one_file_saved_images_path, exist_ok=True)

    extracted_images = convert_from_path(pdf_file_path, dpi=300, first_page=1)
    
    n_pages = len(extracted_images)
    
    # i want to extract metadata from the first 3 pages and last 2 pages
    relevant_pages_for_metadata_extraction = sorted(list(set(list(range(min(3, n_pages))) + list(range(max(3, n_pages-2), n_pages)))))

    figures_paths = defaultdict(list)
    metadata_pages_paths = []
    for page_idx, page in enumerate(extracted_images):
        img_name = f"page_{page_idx}.png"
        one_img_path = os.path.join(one_file_saved_images_path, img_name)
        page.save(one_img_path, "PNG")
        
        if page_idx in relevant_pages_for_metadata_extraction:
            metadata_pages_paths.append(one_img_path)

        image = cv2.imread(one_img_path)

        results = image_segmentation_model(source=one_img_path, conf=0.2, iou=0.8, verbose=False)[0]
        # show results with "supervision" library

        detections = sv.Detections.from_ultralytics(results)

        detections = detections[detections.confidence > min_confidence]

        filtered_bboxes = filter_bboxes(detections.xyxy)
        detections = detections[filtered_bboxes]

        for one_class in needed_features:
            one_class_saved_folder = os.path.join(
                one_file_saved_images_path,
                one_class,
                # f"{pdf_saved_name.replace('.pdf', '')}",
            )
            os.makedirs(one_class_saved_folder, exist_ok=True)

            detections_one_class_type = detections[
                detections.data["class_name"] == one_class
            ]

            # Loop through each detection in detections_pictures_only
            for img_idx, bbox in enumerate(detections_one_class_type.xyxy):
                # Extract the bounding box coordinates
                x_min, y_min, x_max, y_max = map(int, bbox)

                # Crop the region from the image using the bounding box coordinates
                cropped_image = image[y_min:y_max, x_min:x_max]

                # Convert the cropped region to a PNG bytes object
                cropped_pil_image = Image.fromarray(cropped_image)
                cropped_image_bytes = BytesIO()
                cropped_pil_image.save(cropped_image_bytes, format='PNG')
                cropped_image_bytes = cropped_image_bytes.getvalue()
                
                # Remove faces from the cropped image
                cropped_image_without_faces = _remove_faces_png(cropped_image_bytes)

                # Save the modified cropped image
                one_fig_path = os.path.join(
                    one_class_saved_folder, img_name.replace(".png", f"_{img_idx}.png")
                )
                with open(one_fig_path, "wb") as f:
                    f.write(cropped_image_without_faces)

                figures_paths[one_class].append(one_fig_path)

    return figures_paths, metadata_pages_paths


def _extract_first_page_to_base64(pdf_path):
    """
    Extracts the first page of a PDF and encodes it into base64 format.

    Args:
    pdf_path (str): The file path of the PDF document.

    Returns:
    str: The base64 encoded string of the first page image.
    """
    try:
        # Open the PDF file
        doc = fitz.open(pdf_path)

        # Extract the first page
        first_page = doc.load_page(0)

        # Convert the page to an image (pixmap)
        pixmap = first_page.get_pixmap()

        # Store the image in a more common format such as PNG
        image_data = pixmap.tobytes("png")

        # Remove faces from the image
        image_data = _remove_faces_png(image_data)

        # Encode the image data to base64
        base64_encoded_image = base64.b64encode(image_data).decode()

        # Close the PDF document
        doc.close()

        return base64_encoded_image
    except Exception as e:
        return f"An error occurred: {e}"

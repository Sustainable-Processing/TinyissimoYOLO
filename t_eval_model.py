"""
This is a script to evaluate the model using the ONNX runtime. 
The script loads the model, runs it on an image, and then processes the output to draw bounding boxes on the image.

Model tested: v1_small.onnx
Image tested: 2008_000021.jpg
Dataset: VOC2007


Findings: 24 x 30 output from the model.

24 rows:
- 4 bounding box coordinates
- 20 class scores

30 columns:
- Each column represents a potential object detection.

Algorithm to convert the output to bounding boxes and mask it to the filter.


@Author: Tahsin Hasem
@Date: 10th July 2023
"""

import onnx
from PIL import Image
import numpy as np
import pandas as pd
import json

# change to tensor
import torch
import os
import torchvision.transforms as transforms
import onnxruntime
import cv2


CLASS_NAMES = {
    0: "aeroplane",
    1: "bicycle",
    2: "bird",
    3: "boat",
    4: "bottle",
    5: "bus",
    6: "car",
    7: "cat",
    8: "chair",
    9: "cow",
    10: "diningtable",
    11: "dog",
    12: "horse",
    13: "motorbike",
    14: "person",
    15: "pottedplant",
    16: "sheep",
    17: "sofa",
    18: "train",
    19: "tvmonitor",
}


THRESHOLD = 0.18  # Threshold for the confidence score
IOU_THRESHOLD = 0.5


def save_output_data(outputs, name=""):
    outputs_tensor = torch.from_numpy(outputs[0])

    output_numpy = outputs_tensor.numpy().reshape(-1, outputs[0].shape[-1])
    print("output_numpy shape:", output_numpy.shape)

    output_list = output_numpy.tolist()
    # Save the list as JSON
    with open(f"model_raw_data/model_output_{name}.json", "w") as json_file:
        json.dump(output_list, json_file)


def get_image_tensor(path, width, height):
    img = Image.open(path).convert("RGB")
    converted_img = img.resize((width, height))

    transform = transforms.Compose([transforms.ToTensor()])
    img_t = transform(converted_img)
    print("Image Tesnro Shape:", img_t.shape)
    img_t = img_t.unsqueeze(0)  # Add batch dimension

    # Expected shape: (1, 3, 160, 192)
    return img_t


def verify_and_load_onnx_runtime_session(onnx_model_path):
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)

    return onnxruntime.InferenceSession(onnx_model_path)


def get_original_image(image_path):
    """
    get numpy array of image from image path
    """

    img = cv2.imread(image_path)

    return img


def draw_detections(img, box, score, class_id):
    """
    Draws bounding boxes and labels on the input image based on the detected objects.

    Args:
        img: The input image to draw detections on.
        box: Detected bounding box.
        score: Corresponding detection score.
        class_id: Class ID for the detected object.

    Returns:
        None
    """

    CLASS_NAMES = {
        0: "aeroplane",
        1: "bicycle",
        2: "bird",
        3: "boat",
        4: "bottle",
        5: "bus",
        6: "car",
        7: "cat",
        8: "chair",
        9: "cow",
        10: "diningtable",
        11: "dog",
        12: "horse",
        13: "motorbike",
        14: "person",
        15: "pottedplant",
        16: "sheep",
        17: "sofa",
        18: "train",
        19: "tvmonitor",
    }

    # Extract the coordinates of the bounding box
    x1, y1, w, h = box

    # Retrieve the color for the class ID
    color = (0, 0, 255)

    # Draw the bounding box on the image
    cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

    # Create the label text with class name and score
    label = f"{CLASS_NAMES[class_id]}: {score:.2f}"

    # Calculate the dimensions of the label text
    (label_width, label_height), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )

    # Calculate the position of the label text
    label_x = x1
    label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

    # Draw a filled rectangle as the background for the label text
    cv2.rectangle(
        img,
        (label_x, label_y - label_height),
        (label_x + label_width, label_y + label_height),
        color,
        cv2.FILLED,
    )

    # Draw the label text on the image
    cv2.putText(
        img,
        label,
        (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


model_runtime = verify_and_load_onnx_runtime_session("v1_small.onnx")


def run(image_path):

    MODEL_IMAGE_WIDTH = 160
    MODEL_IMAGE_HEIGHT = 192

    img_name = os.path.basename(image_path).split(".")[0]
    input_image = get_image_tensor(image_path, MODEL_IMAGE_WIDTH, MODEL_IMAGE_HEIGHT)

    input_name = model_runtime.get_inputs()[0].name
    outputs = model_runtime.run(None, {input_name: input_image.numpy()})

    save_output_data(outputs, img_name)

    output = outputs[0]

    # Transpose the output to the correct shape for processing. Expected shape (30, 24)
    outputs_2 = np.transpose(np.squeeze(output[0]))
    print("outputs_2 shape:", outputs_2.shape)
    assert outputs_2.shape == (30, 24)

    rows = outputs_2.shape[0]
    boxes = []
    scores = []
    class_ids = []

    original_image = get_original_image(image_path)

    original_height, original_width, _ = original_image.shape

    # Calculate the scaling factors for the bounding box coordinates. The model was trained on 160x192 images. Image being used is 160x192, so the scaling factors are 1.
    x_factor = (
        original_width / MODEL_IMAGE_WIDTH
    )  # <- TODO: Change to adjust to camera resolution
    y_factor = (
        original_height / MODEL_IMAGE_HEIGHT
    )  # <- TODO: Change to adjust to camera resolution

    for i in range(rows):

        classes_scores = outputs_2[i][4:]
        max_score = np.amax(classes_scores)

        if max_score > THRESHOLD:
            class_id = np.argmax(classes_scores)
            print("max_score:", max_score, "class_id:", class_id, "i:", i)

            x, y, w, h = (
                outputs_2[i][0],
                outputs_2[i][1],
                outputs_2[i][2],
                outputs_2[i][3],
            )

            left = int((x - w / 2) * x_factor)
            top = int((y - h / 2) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)

            # Ensure the bounding box coordinates are within the image dimensions
            left = max(0, min(left, 160))
            top = max(0, min(top, 192))
            width = max(0, min(width, 160))
            height = max(0, min(height, 192))

            class_ids.append(class_id)
            scores.append(max_score)
            boxes.append([left, top, width, height])

    # Perform non-maximum suppression to remove overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, scores, THRESHOLD, IOU_THRESHOLD)

    # Iterate over the selected indices after non-maximum suppression
    for i in indices:
        # Get the box, score, and class ID corresponding to the index
        box = boxes[i]
        score = scores[i]
        class_id = class_ids[i]

        # Draw the detection on the input image
        draw_detections(original_image, box, score, class_id)

    save_output_image(original_image, image_path)


def save_output_image(image, image_path):
    image_name = os.path.basename(image_path)

    if os.path.exists("output_images") is False:
        os.mkdir("output_images")

    output_path = os.path.join("output_images", image_name)
    print("output_path:", output_path)
    cv2.imwrite(output_path, image)


images_dir = "gap9_tests"

for image_name in os.listdir(images_dir):
    image_path = os.path.join(images_dir, image_name)
    print("image_path:", image_path)
    run(image_path=image_path)

"""
inferencing_engines.py
=============================================
Where the inferencing engines/classes reside.
"""
import tensorflow as tf
import numpy as np
import cv2

from utils import text_detection, sort_poly, xy_maxmin, get_text, read_labels, dequantize


class InferenceEngine:
    """Base inferencing class that wraps Tensorflow Lite"""
    def __init__(self, model_path_str):
        """
        Initializes InferenceEngine. Creates interpreter, allocates tensors, and grabs input and output details.

        :param model_path_str: Path to Tensorflow Lite Model
        """
        # load trained model
        self.interpreter = tf.lite.Interpreter(model_path=model_path_str)
        self.interpreter.allocate_tensors()

        # Get input and output tensors
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def get_input_shape(self, index=0):
        """
        Gets the shape of an input tensor

        :param index: Index of input tensor
        :return: Shape of tensor
        """
        return self.input_details[index]['shape']

    def fill_input_tensors(self, list_of_numpy_arrays):
        """
        Fills the interpreter's input tensors

        :param list_of_numpy_arrays: A list of the input tensors
        """
        for index, numpy_array in enumerate(list_of_numpy_arrays):
            input_shape = self.get_input_shape(index)
            self.interpreter.set_tensor(self.input_details[index]['index'], numpy_array.reshape(input_shape))

    def invoke(self):
        """
        Invokes the interpreter. Basically has the interpreter run the needed calculation to have the output
        tensors ready.
        """
        self.interpreter.invoke()

    def get_output_tensors(self):
        """
        Gets the output tensors tensors from the interpreter, assuming the interpreter has been invoked.

        :return: A list of output tensors (list of numpy arrays)
        """
        out = []
        for detail in self.output_details:
            out.append(self.interpreter.get_tensor(detail['index']))
        return out

    def resize_to_network_input(self, image):
        """
        Resizes a numpy array to bee the same as the 0th index of the input tensors

        :param image: The numpy array to resize
        :return: A resized numpy array
        """
        input_shape = self.get_input_shape()
        return cv2.resize(image, (input_shape[1], input_shape[2]))


class ObjectDetectionEngine(InferenceEngine):
    """Class that performs Object Detection"""
    def __init__(self, model_path_str, label_path_dir, score_threshold):
        """
        Initiates the ObjectDetectionEngine class.

        :param model_path_str: Path to SSD object detection model
        :param label_path_dir: Path to coco format labels
        :param score_threshold: Threshold for detection in the form of a float between 0 and 1
        """
        InferenceEngine.__init__(self, model_path_str)
        self.labels = read_labels(label_path_dir)
        self.score_threshold = score_threshold

    def change_score_threshold(self, score_threshold):
        """
        Changes the threshold for object detection

        :param score_threshold: A float between 0 and 1
        """
        self.score_threshold = score_threshold

    def run_inference(self, image):
        """
        Performs the inferencing

        :param image: An image(numpy array) to perform inferencing on
        :return: List of the names of detected objects, returns empty list if no object is detected
        """
        img_resized = self.resize_to_network_input(image)
        self.fill_input_tensors([img_resized])
        self.invoke()
        out_list = self.get_output_tensors()

        classes = out_list[1][0]
        scores = out_list[2][0]

        ret_classes = []
        for index, cls in enumerate(classes):
            cls += 1
            if scores[index] > self.score_threshold and int(cls) != 0:
                ret_classes.append(self.labels[int(cls)])
        return ret_classes


class TextDetectionEngine(InferenceEngine):
    """Class that performs Optical Text Recognition"""
    def __init__(self, model_path_str):
        """
        Initiates the TextDetectionEngine class

        :param model_path_str: A path to the EAST model
        """
        InferenceEngine.__init__(self, model_path_str)

    def run_inference(self, image):
        """
        Performs the inferencing and OCR

        :param image: An image(numpy array) to perform inferencing on
        :return: List of text detected, returns empty list of no object is detected
        """
        input_shape = self.get_input_shape()
        img_resized = self.resize_to_network_input(image)
        img_resized = (img_resized / 127.5) - 1
        img_resized = img_resized.astype(np.float32)
        self.fill_input_tensors([img_resized])
        self.invoke()
        out_list = self.get_output_tensors()

        # model outputs 3 tensors now, normalized to between 0 and 1
        score_map = dequantize(out_list[0], 128, 127)
        geo_loc_map = dequantize(out_list[1], 128, 127)
        geo_angle = dequantize(out_list[2], 128, 127)
        score_map = (score_map + 1) * 0.5
        geo_loc_map = (geo_loc_map + 1) * input_shape[1] / 2
        geo_angle = 0.7853981633974483 * geo_angle
        geo_map = np.concatenate((geo_loc_map, geo_angle), axis=3)

        boxes = text_detection(score_map=score_map, geo_map=geo_map)

        if boxes is not None:
            boxes = boxes[:, :8].reshape((-1, 4, 2))
            boxes[:, :, 0] /= input_shape[1] / image.shape[1]
            boxes[:, :, 1] /= input_shape[2] / image.shape[0]

            output_text = []
            for box in boxes:
                box = sort_poly(box.astype(np.int32))
                if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                    continue
                (x_max, x_min), (y_max, y_min) = xy_maxmin(box[:, 0], box[:, 1])
                sub_img = image[y_min:y_max, x_min:x_max]
                txt = get_text(sub_img)
                if txt != '':
                    output_text.append(get_text(sub_img))
            return output_text
        return []

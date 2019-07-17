"""
inferencing_engines.py
=============================================
Where the inferencing engines/classes reside.
"""
try:
    from edgetpu.basic.basic_engine import BasicEngine
    use_tpu = True
except:
    import tensorflow as tf
    use_tpu = False

import numpy as np
import cv2

from utils import text_detection, sort_poly, xy_maxmin, get_text, read_labels, dequantize, quantize


class InferenceEngine:
    """Base inferencing class that wraps Tensorflow Lite"""
    def __init__(self, model_path, output_shapes):
        """
        Initializes InferenceEngine. Creates interpreter, allocates tensors, and grabs input and output details.

        :param model_path: Path to Tensorflow Lite Model
        :param output_shapes: List of tuples, each tuple containing the shape of an output tensor
        """
        # load trained model
        if use_tpu:
            self.TPU_engine = BasicEngine(model_path)
        else:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()

            # Get input and output tensors
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        self.output_shapes = output_shapes

    def get_input_shape(self):
        """
        Gets the shape of the input tensor of the model

        :return: Input shape as a tuple
        """
        if use_tpu:
            return self.TPU_engine.get_input_tensor_shape()
        else:
            return self.input_details[0]['shape']

    def invoke(self, input_tensor, lambda_preprocess=None):
        """
        Invokes the interpreter. Basically has the interpreter run the needed calculation to have the output
        tensors ready.

        :param input_tensor: Input Tensor (nD numpy array)
        :param lambda_preprocess: Lambda function to apply to input tensors after resizing
        :return: A list of output tensors (list of numpy arrays)
        """
        input_shape = self.input_details[0]['shape']
        resized_tensor = cv2.resize(input_tensor, (input_shape[1], input_shape[2]))
        if lambda_preprocess:
            resized_tensor = lambda_preprocess(resized_tensor)

        output = []
        if use_tpu:
            resized_tensor = resized_tensor.flatten()
            _, raw_results = self.TPU_engine.RunInference(resized_tensor)

            so_far = 0
            for index, shape in enumerate(self.output_shapes):
                output_size = self.TPU_engine.get_output_tensor_size(index)
                output.append(raw_results[so_far:so_far + output_size].reshape(shape))
                so_far += output_size
        else:
            self.interpreter.set_tensor(self.input_details[0]['index'], resized_tensor.reshape(input_shape))
            self.interpreter.invoke()

            for index, detail in enumerate(self.output_details):
                output.append(self.interpreter.get_tensor(detail['index']).reshape(self.output_shapes[index]))

        return output


class ObjectDetectionEngine(InferenceEngine):
    """Class that performs Object Detection"""
    def __init__(self, model_path, label_path, score_threshold):
        """
        Initiates the ObjectDetectionEngine class.

        :param model_path: Path to SSD object detection model
        :param label_path: Path to coco format labels
        :param score_threshold: Threshold for detection in the form of a float between 0 and 1
        """
        InferenceEngine.__init__(self, model_path, [(1, 10, 4), (1, 10), (1, 10), (1)])
        self.labels = read_labels(label_path)
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
        out_list = self.invoke(image)

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
    def __init__(self, model_path):
        """
        Initiates the TextDetectionEngine class

        :param model_path: A path to the EAST model
        """
        InferenceEngine.__init__(self, model_path, [(1, 80, 80, 1), (1, 80, 80, 4), (1, 80, 80, 1)])

    def run_inference(self, image):
        """
        Performs the inferencing and OCR

        :param image: An image(numpy array) to perform inferencing on
        :return: List of text detected, returns empty list of no object is detected
        """
        input_shape = self.get_input_shape()
        preprocess_function = lambda img : quantize((img / 127.5) - 1, 128, 127)
        out_list = self.invoke(image, preprocess_function)

        # model outputs 3 tensors now, normalized to between 0 and 1
        score_map = dequantize(out_list[0], 128, 127)
        geo_loc_map = dequantize(out_list[1], 128, 127)
        geo_angle = dequantize(out_list[2], 128, 127)
        score_map = (score_map + 1) * 0.5
        geo_loc_map = (geo_loc_map + 1) * 256
        geo_angle = 0.7853981633974483 * geo_angle
        geo_map = np.concatenate((geo_loc_map, geo_angle), axis=3)

        boxes = text_detection(score_map=score_map, geo_map=geo_map)

        if boxes is not None:
            boxes = boxes[:, :8].reshape((-1, 4, 2))
            boxes[:, :, 0] /= input_shape[1] / image.shape[0]
            boxes[:, :, 1] /= input_shape[2] / image.shape[1]

            output_text = []
            for box in boxes:
                box = sort_poly(box.astype(np.int32))
                if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                    continue
                (x_max, x_min), (y_max, y_min) = xy_maxmin(box[:, 0], box[:, 1])

                if x_max > image.shape[0]:
                    x_max = image.shape[0]
                if x_min < 0:
                    x_min = 0
                if y_max > image.shape[1]:
                    y_max = image.shape[1]
                if y_min < 0:
                    y_min = 0

                cv2.polylines(image, [box.astype(np.int32).reshape(-1, 1, 2)], True, color=(255, 255, 0), thickness=2)

                sub_img = image[y_min:y_max, x_min:x_max]
                txt = get_text(sub_img)
                if txt != '':
                    output_text.append(get_text(sub_img))
            return output_text
        return []

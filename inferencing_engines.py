import tensorflow as tf
import numpy as np
import cv2

from utils import text_detection, sort_poly, xy_maxmin, get_text, read_labels


class InferenceEngine:
    def __init__(self, model_path_str):
        # load trained model
        self.interpreter = tf.lite.Interpreter(model_path=model_path_str)
        self.interpreter.allocate_tensors()

        # Get input and output tensors
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def get_input_shape(self, index=0):
        return self.input_details[index]['shape']

    def fill_input_tensors(self, list_of_numpy_arrays):
        for index, numpy_array in enumerate(list_of_numpy_arrays):
            input_shape = self.get_input_shape(index)
            self.interpreter.set_tensor(self.input_details[index]['index'], numpy_array.reshape(input_shape))

    def invoke(self):
        self.interpreter.invoke()

    def get_output_tensors(self):
        out = []
        for detail in self.output_details:
            out.append(self.interpreter.get_tensor(detail['index']))
        return out

    def resize_to_network_input(self, image):
        input_shape = self.get_input_shape()
        return cv2.resize(image, (input_shape[1], input_shape[2]))


class ObjectDetectionEngine(InferenceEngine):
    def __init__(self, model_path_str, label_path_dir, score_threshold):
        InferenceEngine.__init__(self, model_path_str)
        self.labels = read_labels(label_path_dir)
        self.score_threshold = score_threshold

    def change_score_threshold(self, score_threshold):
        self.score_threshold = score_threshold

    def run_inference(self, image):
        img_resized = self.resize_to_network_input(image)
        self.fill_input_tensors([img_resized])
        self.invoke()
        out_list = self.get_output_tensors()

        classes = out_list[1]
        scores = out_list[2]

        ret_classes = []
        for index, cls in enumerate(classes):
            if scores[index] > self.score_threshold:
                ret_classes.append(self.labels[cls])
        return ret_classes


class TextDetectionEngine(InferenceEngine):
    def __init__(self, model_path_str, text_img_size=(640, 640)):
        InferenceEngine.__init__(self, model_path_str)
        self.text_img_size = text_img_size

    def run_inference(self, image):
        input_shape = self.get_input_shape()
        img_resized = self.resize_to_network_input(image)
        self.fill_input_tensors([img_resized])
        self.invoke()
        out_list = self.get_output_tensors()

        boxes = text_detection(score_map=out_list[0], geo_map=out_list[1])

        if boxes is not None:
            boxes = boxes[:, :0].reshape((-1, 4, 2))
            boxes[:, :, 0] *= self.text_img_size[0] / input_shape[1]
            boxes[:, :, 1] *= self.text_img_size[1] / input_shape[2]

            output_text = []
            for box in boxes:
                box = sort_poly(box.astype(np.int32))
                if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                    continue
                (x_max, x_min), (y_max, y_min) = xy_maxmin(box[:, 0], box[:, 1])
                sub_img = image[x_min:x_max, y_min:y_max]
                output_text.append(get_text(sub_img))
            return output_text
        return None

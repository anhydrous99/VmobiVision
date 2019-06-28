iMobi
=====

This is the documentation for iMobi's python software. A python program that does headless object detection
and optical character recognition. This project uses OpenCV to interface with the camera, tensorflow lite for
inferencing, and several other packages. For object detection, the Single Shot MultiBox (SSD) model
<https://arxiv.org/abs/1512.02325> is used with MobileNet V2 <https://arxiv.org/abs/1801.04381> as the
base convolution architecture. For text detection, an EAST model <https://arxiv.org/abs/1704.03155> is used, again,
with the base convolution architecture begin the MobileNet V2 architecture. `Tesseract
<https://github.com/tesseract-ocr/tesseract>`_ is then used to acquire the text.
One of the qoals is to have all models quantized and trained with `quantization-aware training
<https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/quantize>`_ for use with edge-TPUs.

Currently, only object detection is quantized and therefore runnable on a coral's TPU. The reason for this is the
difference in scaling between the predicted location mapping and their scores. The current model's location mapping
ranges from 0 to 256 and the score are between 0 and 1. Since scaling both to unsigned 8 bit integers, quantized
between 0 and 255, would result in the score only being able to be 0 or 1 and not in between. To fix this, we need
to retrain the model so that the scores are between 0 and 256 or so that the location mapping is between 0 and 1.
Not to mention, the input images are scales to between -1 and 1, it would result in a better model if it is scaled
to either between 0 and 256 or between 0 and 1. Once that is done, quantization of the model would be trivial.
This will be done at my EAST model github page where the progress can be seen <https://github.com/anhydrous99/EAST>.

The project uses the keyboard python package that takes low level control of the keyboard, allowing the ability
to use assign global hotkey. This, however, requires sudo permission. We can then assign these keys to specific
embedded buttons or activation devices, the potential here is limitless.

The OCR pipeline starts with the image being processed through the EAST model. The model produces a set of probable
locations on the image where text appears. The probable location is then cut out and passed through tesseract,
the industry standard for OCR in PDFs. Tesseract returns the probable text on the parts of the images. Take a look
at both the EAST github page and the references for more information.

Future work here could include doing away with tesseract and implementing a tesseract model allowing to the full
quantization of the OCR pipeline. In my opinion, the clearest path to do this is to find a model that
tesseract uses implemented for tensorflow on github and altering it for our needs.

To add more classes that to the object detector learning transfer can be done where the model that was converted
to the tensorflow lite model is used as a template for the new model with different or more classes. You can follow use
the following tutorial to learn how to do this <https://www.tensorflow.org/tutorials/images/transfer_learning>. Be
sure to embed `quantized aware training
<https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/quantize>`_ when altering models. Then you can
use the tflite_convert utility or python `TFLiteConverter
<https://www.tensorflow.org/api_docs/python/tf/lite/TFLiteConverter>`_ tensorflow class to convert the model to a
fully quantized one.::

    OUTPUT_FILE=ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tflite
    OUTPUT_ARRAYS='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'
    tflite_convert --graph_def_file=tflite_graph.pb \
                   --output_file=${OUTPUT_FILE} \
                   --output_format=TFLITE \
                   --input_shape=1,300,300,3 \
                   --input_arrays=normalized_input_image_tensor \
                   --output_arrays=${OUTPUT_ARRAYS} \
                   --inference_type=QUANTIZED_UINT8 \
                   --allow_custom_ops \
                   --mean_values=127 \
                   --std_dev_values=127 \
                   --default_ranges_min=0 --default_ranges_max=6

To run on the edge TPU on a coral, the tensorflow lite model then needs to compiled according to this guide
<https://coral.withgoogle.com/docs/edgetpu/models-intro/>. To convert the EAST model to a quantized tensorflow lite
model I created a private `github repo <https://github.com/anhydrous99/EASTkerastotensorflowlite>`_ that uses the
TFLiteConverter class, it is still a work in progress as the quantized EAST model. If you need access to the github
project or want the project to be public let me know.

Currently, to train the EAST model you can follow the instructions on the `github page
<https://github.com/anhydrous99/EAST>`_. If you would like to make changes to the EAST project create a fork and
submit pull requests of the changes.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   references
   license



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

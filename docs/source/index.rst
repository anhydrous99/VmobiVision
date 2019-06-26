iMobi
=====

This is the documentation for iMobi's python software. A python program that does headless object detection
and optical character recognition. This project uses OpenCV to interface with the camera, tensorflow lite for
inferencing, and several other packages. For object detection, the Single Shot MultiBox (SSD) model
<https://arxiv.org/abs/1512.02325> is used with MobileNet V2 <https://arxiv.org/abs/1801.04381> as the
base convolution architecture. For text detection, an EAST model <https://arxiv.org/abs/1704.03155> is used, again,
with the base convolution architecture begin the MobileNet V2 architecture. `Tesseract
<https://github.com/tesseract-ocr/tesseract>`_ is then used to acquire the text.
All models are quantized and trained with `quantization-aware training
<https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/quantize>`_
for use with edge-TPUs, I'll add that functionality when I get a `coral <https://coral.withgoogle.com/>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   license



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

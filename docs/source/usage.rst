Usage
=====
    :usage:
      run.py [-h] [-o OBJECT_DETECTION_MODEL] [-t TEXT_DETECTION_MODEL] [-l OBJECT_DETECTION_LABEL] [--no_text_to_speech]

    Object Detection & Optical Text Recognition software

    :optional arguments:
      -h, --help
        shows the help message and exit
      -o OBJECT_DETECTION_MODEL, --object_detection_model OBJECT_DETECTION_MODEL
                            The model file for object detection
      -t TEXT_DETECTION_MODEL, --text_detection_model TEXT_DETECTION_MODEL
                            The model file for text detection
      -l OBJECT_DETECTION_LABEL, --object_detection_label OBJECT_DETECTION_LABEL
                            Label file for object detection
      --no_text_to_speech   Use this to disable text to speech
Since it takes low level control of keyboard, in some systems, it requires sudo user access.

---------------------------------------------------------------------------------------

    :key bindings:
      While the program is running you can press these keys to perform these actions

      - o - to switch to object detection mode
      - t - to switch to optical character recognition mode
      - q - to quit the program

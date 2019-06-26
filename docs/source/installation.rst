Installation
============

Ubuntu & Debian based systems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To install in ubuntu I would recommend you use a python virtual envirnment. To do this run the following commands::

    sudo apt update
    sudo apt install python3-dev python3-venv python3-pip libespeak-dev libtesseract-dev
    python3 -m venv venv
    source venv/bin/activate
    pip install pyttsx3 Sphinx sphinx-glpi-theme opencv-python-headless setuptools numpy \
                scipy tensorflow lanms-proper pytesseract Pillow keyboard

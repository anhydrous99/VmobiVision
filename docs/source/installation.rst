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

Mac OS X - Mojave
^^^^^^^^^^^^^^^^^
Assuming you are using a fresh install of python 3.7, are on the project root directory, and have brew install you can run following::

    brew install tesseract espeak
    python3 -m venv venv
    source venv/bin/activate
    pip install pyttsx3 Sphinx sphinx-glpi-theme opencv-python-headless setuptools numpy \
                scipy tensorflow pytesseract Pillow keyboard

The lanms-proper package isn't available through the standard pypl repos but you can install it from the github project.
First make sure you have xcode and git installed then in an arbitrary directory run::

    git clone --recursive https://github.com/safijari/lanms
    cd lanms
    python setup.py install
    cd ..
    rm -r lanms

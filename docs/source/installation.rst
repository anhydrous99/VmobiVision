Installation
============

Ubuntu & Debian based systems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To install in ubuntu I would recommend you use a python virtual envirnment. To do this run the following commands::

    sudo apt update
    sudo apt install build-essential git cmake python3-dev python3-venv python3-pip libespeak-dev libtesseract-dev
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

Raspberry Pi
^^^^^^^^^^^^
On the Pi we need to have the camera's kernel module load on startup. You can do that by running the following::

    echo 'bcm2835-v4l2' | sudo tee -a /etc/modules
    sudo reboot

Unfortunately, we have to build alot of the dependencies that are required on the pi. So,
to do that, you can run::

    sudo apt update
    sudo apt install build-essential git cmake python3-dev python3-venv python3-pip \
                     python3-numpy libespeak-dev libtesseract-dev
    python3 -m venv venv
    source venv/bin/activate
    pip3 install cython pytest setuptools
    pip3 install https://github.com/numpy/numpy/releases/download/v1.16.4/numpy-1.16.4.zip
    pip3 install pyttsx3 Sphinx sphinx-glpi-theme scipy tensorflow \
                 pytesseract Pillow keyboard
    ENABLE_HEADLESS=1 pip3 install git+https://github.com/skvark/opencv-python
    git clone --recursive https://github.com/safijari/lanms
    cd lanms
    python setup.py install
    cd ..
    rm -r lanms

Coral
^^^^^
TODO

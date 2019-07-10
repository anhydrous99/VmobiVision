Installation
============

Ubuntu & Debian based systems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To install in ubuntu I would recommend you use a python virtual envirnment. To do this run the following commands::

    sudo apt update
    sudo apt install build-essential git cmake python3-dev python3-venv python3-pip libespeak-dev libtesseract-dev
    git clone https://github.com/anhydrous99/Vmobi
    cd Vmobi
    python3 -m venv venv
    source venv/bin/activate
    pip install pyttsx3 Sphinx opencv-python-headless setuptools numpy scipy tensorflow lanms-proper \
                pytesseract Pillow keyboard

Mac OS X - Mojave
^^^^^^^^^^^^^^^^^
Assuming you are using a fresh install of python 3.7, are on the project root directory, and have brew install you can run following::

    brew install tesseract espeak git cmake
    git clone https://github.com/anhydrous99/Vmobi
    cd Vmobi
    python3 -m venv venv
    source venv/bin/activate
    pip install pyttsx3 Sphinx opencv-python-headless setuptools numpy scipy tensorflow pytesseract Pillow keyboard
    pip install https://vmobipypi.blob.core.windows.net/pypi-blob/lanms_proper-1.0.0-cp37-cp37m-macosx_10_9_x86_64.whl

The lanms-proper package isn't available through the standard pypl repos but you can install from my pypi repo (last
command above).

Raspberry Pi
^^^^^^^^^^^^
On the Pi we need to have the camera's kernel module load on startup. You can do that by running the following::

    echo 'bcm2835-v4l2' | sudo tee -a /etc/modules
    sudo reboot

Unfortunately, we have to build alot of the dependencies that are required on the pi. So,
to do that, you can run::

    sudo apt update
    sudo apt install build-essential git cmake python3-dev python3-venv python3-pip \
                     libespeak-dev libtesseract-dev libhdf5-dev
    git clone https://github.com/anhydrous99/Vmobi
    cd Vmobi
    python3 -m venv venv
    source venv/bin/activate
    pip3 install cython pytest setuptools
    pip3 install https://github.com/numpy/numpy/releases/download/v1.16.4/numpy-1.16.4.zip
    pip3 install pyttsx3 Sphinx scipy tensorflow pytesseract Pillow keyboard
    ENABLE_HEADLESS=1 pip3 install git+https://github.com/skvark/opencv-python
    git clone --recursive https://github.com/safijari/lanms
    cd lanms
    python setup.py install
    cd ..

Jetson Nano
^^^^^^^^^^^
Installing the dependencies on the nano is similar to the pi except pypl has no binary for tensorflow.
You can how ever download the tensorflow binary provided by NVIDIA. We will have to build numpy, scipy, opencv-python
and lanms, aswell.::

    sudo apt update
    sudo apt install build-essential gfortran git cmake python3-dev python3-venv \
                     python3-pip libespeak-dev libtesseract-dev libhdf5-dev      \
                     libopenblas-dev
    git clone https://github.com/anhydrous99/Vmobi
    cd Vmobi
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install cython pytest setuptools
    pip install https://github.com/numpy/numpy/releases/download/v1.16.4/numpy-1.16.4.zip
    pip install https://github.com/scipy/scipy/releases/download/v1.2.2/scipy-1.2.2.zip
    pip install pyttsx3 Sphinx pytesseract Pillow keyboard
    ENABLE_HEADLESS=1 pip install git+https://github.com/skvark/opencv-python
    pip install https://developer.download.nvidia.com/compute/redist/jp/v42/tensorflow-gpu/tensorflow_gpu-1.13.1+nv19.5-cp36-cp36m-linux_aarch64.whl
    git clone --recursive https://github.com/safijari/lanms
    cd lanms
    python setup.py install
    cd ..

Make sure not to delete the lanms folder.

Coral
^^^^^
Unfortunatly, for the coral, compilation of alot of the dependencies is required. Fret not, I have compiled most into
pip packages for install. This project checks for the edgetpu python api first then it falls back to tensorflow lite.
So, when using mendel in the coral dev board, since the edgetpu python api is already installed, you wont need to
install tensorflow.::

    sudo apt update
    sudo apt install build-essential gfortran git cmake python3-dev python3-venv \
                     python3-pip libespeak-dev libtesseract-dev libhdf5-dev      \
                     libopenblas-dev swig libjpeg-dev
    git clone https://github.com/anhydrous99/Vmobi
    cd Vmobi
    python3 -m venv --system-site-packages venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install https://vmobipypi.blob.core.windows.net/pypi-blob/numpy-1.16.4-cp35-cp35m-linux_aarch64.whl
    pip install https://vmobipypi.blob.core.windows.net/pypi-blob/scipy-1.2.2-cp35-cp35m-linux_aarch64.whl
    pip install pyttsx3 Sphinx pytesseract Pillow keyboard
    pip install https://vmobipypi.blob.core.windows.net/pypi-blob/opencv_python_headless-4.1.0+da7d022-cp35-cp35m-linux_aarch64.whl
    pip install https://vmobipypi.blob.core.windows.net/pypi-blob/lanms_proper-1.0.0-cp35-cp35m-linux_aarch64.whl

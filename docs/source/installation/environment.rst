===========
Environment
===========

Conda (preferred)
=================
Once you have Conda installed on your machine (Miniconda and Anaconda Distribution each suffice), use the `env.yml`_ file to create a Conda-enabled virtual environment. This can be completed using the following commands:

.. _env.yml: https://github.com/TUL-DEV/QuantUS/blob/main/env.yml

.. code-block:: shell

   git clone https://github.com/TUL-Dev/QuantUS.git
   cd QuantUS
   conda env create -f env.yml

From here, this environment can be activated using the :code:`conda activate QuantUS-env` command.

VirtualEnv (deprecated)
=======================

Deprecated but continued for backwards compatibility, this environment uses `pip`_ and the `virtualenv`_ Python library to create a Pip-enabled virtual environment. This can be completed using the following commands. Note
 `#Unix` commands should only be run on Unix machines and similarly, `#Windows` commands should only be run on Windows machines.

.. _pip: https://pypi.org/project/pip/
.. _virtualenv: https://pypi.org/project/virtualenv/

.. code-block:: shell

   git clone https://github.com/TUL-Dev/QuantUS.git
   cd QuantUS
   pip install --upgrade pip
   python -m pip install virtualenv
   virtualenv --python="python3.9" venv
   source venv/bin/activate # Unix
   call venv\bin\activate.bat # Windows
   pip install -r requirements.txt

Following this example, this environment can be accessed via the :code:`source venv/bin/activate` command from the repository directory.

Troubleshooting for Mac
-----------------------

To run these commands, make sure you have `HomeBrew`_ installed.

If you encounter an error after running :code:`pip install -r requirements.txt`, try the following code and then run the command again:

.. code-block:: shell

   brew install qt5

   brew link qt5 --force

   pip install wheel setuptools pip --upgrade

   pip install pyqt5==5.15.9 --config-settings --confirm-license= --verbose

If an error persists after running :code:`python main.py`, try :code:`export QT_QPA_PLATFORM_PLUGIN_PATH=/opt/homebrew/Cellar/qt@5/5.15.13_1/plugins`.

.. _HomeBrew: https://brew.sh/

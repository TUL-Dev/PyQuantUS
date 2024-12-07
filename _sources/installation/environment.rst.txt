===========
Environment
===========

Python
======

To access all features in QuantUS from any major operating system, 
all that’s required is `Python3`_! Non-Conda `Python3.11.8`_ is recommended
for the best experience, but any Python version above 3.9 should suffice for
basic use.

.. _Python3.11.8: https://www.python.org/downloads/release/python-3118/
.. _Python3: https://www.python.org/downloads/

Once Python is installed, the next step is to create a virtual environment. In the 
following sections, we will refer to the path accessing the Python version as :code:`$PYTHON`.

Virtual Environment
===================

Complete the following steps. Note lines commented with :code:`# Unix` should only be 
run on MacOS or Linux while lines commented with :code:`# Windows` should only be run on Windows.

.. code-block:: shell

   git clone https://github.com/TUL-Dev/QuantUS.git
   cd QuantUS
   $PYTHON pip install virtualenv
   $PYTHON -m virtualenv .venv
   source .venv/bin/activate # Unix
   call .venv\bin\activate.bat # Windows
   pip install -r requirements.txt

Following this example, this environment can be accessed via the :code:`source .venv/bin/activate` 
command from the repository directory on Unix systems, and by :code:`call .venv\bin\activate.bat` on Windows.

===========
Environment
===========

First, download `Python3.11.8`_ (non-Conda version) from the Python
website. Once installed, note its path. We will refer to this path as :code:`$PYTHON`
in the following sections.

Next, complete the following steps. Note lines commented with :code:`# Unix` should only be
run on MacOS or Linux while lines commented with :code:`# Windows` should only be run on Windows.

.. code-block:: shell

   git clone https://github.com/TUL-Dev/QuantUS.git
   cd QuantUS
   $PYTHON -m pip install virtualenv
   $PYTHON -m virtualenv .venv
   source .venv/bin/activate # Unix
   .venv\Scripts\activate # Windows cmd
   sudo apt-get update & sudo apt-get install python3-dev # Linux
   pip install -r requirements.txt


.. _Python3.11.8: https://www.python.org/downloads/release/python-3118/

Following this example, this environment can be accessed via the :code:`source .venv/bin/activate`
command from the repository directory on Unix systems, and by :code:`.venv\Scripts\activate` on Windows.

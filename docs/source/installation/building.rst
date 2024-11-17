========
Building
========

After configuring a Python virtual environment, finish preparing QuantUS to be run using the following commands:

.. code-block:: shell

   # Using Python virtual env (Mac/Linux)
   chmod +x saveQt.sh
   ./saveQt.sh

.. code-block:: shell

   # Using Python virtual env (Windows)
   ren saveQt.sh saveQt.bat
   .\saveQt.bat

Note for Windows devices, to finish preparing QuantUS to be run with the Philips RF parser, compile `philips_rf_parser_windows.c`_ into a `philips_rf_parser.exe` executable using a Windows C compiler of choice.

.. _philips_rf_parser_windows.c: https://github.com/TUL-DEV/QuantUS/blob/main/src/Parsers/philips_rf_parser_windows.c

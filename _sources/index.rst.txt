QuantUS
=======

**QuantUS** is an open-source quantitative ultrasound analysis tool designed to
facilitate the standardization and reproducibility of ultrasound data.
Both a desktop application and command line interface (CLI), the software supports 
quantitative ultrasound spectroscopy (QUS) on radiofrequency (RF) and in-phase and 
quadrature (IQ) data. It also features 2D/3D dynamic contrast-enhanced ultrasound 
(DCE-US) time intensity curve (TIC) analysis for monitoring bolus injections of 
contrast over time.

.. image:: mbfSc.png
   :width: 600
   :alt: MBF Parametric Map Example
   :align: center

| 

In addition, QuantUS also provides parametric map generation for all forms of analysis,
creating extra modalities of viewing ultrasound data, as shown above. The software is compatible with 
all major operating systems (i.e. Mac OS X, Windows, and Linux).

.. note::
   This project is under active development.


----------------
Getting Started
----------------

.. toctree::
   :caption: Overview
   :maxdepth: 2
   
   overview
   usage/index

.. toctree::
   :caption: Developers
   :maxdepth: 1

   installation/index
   PyQuantUS API<modules>

===========
Overview
===========

QuantUS is a free, standalone, native graphical user interface (GUI) facilitating ultrasound research compatible on Mac OS X, Windows, and Linux. The tool provides an end-to-end workflow for analysis using radiomics approaches including:

* 2D Quantitative Ultrasound Spectroscopy (QUS) via spectral analysis of radiofrequency (RF) and in-phase and quadrature (IQ) data
* 2D Dynamic Contrast-Enhanced Ultrasound (DCE-US) Perfusion Imaging Analysis with optional Motion Compensation
* 3D DCE-US Perfusion Imaging Analysis

QUS Overview
============

Given user-inputted RF or IQ ultrasound data, this feature runs spectral analysis to compute quantitative ultrasound parameters and parametric maps on a custom region of interest (ROI). In QuantUS, the midband fit (MBF), spectral slope (SS), and spectral intercept (SI) spectral parameters as described by `El Kaffas et al.`_ have been validated and used in numerous ultrasound studies. Additionally, the backscatter coefficient, attenuation coefficient, Nakagami parameter, effective scatterer size, and effecive scatterer concentration have all been implemented into the UI and are in the validation process.

The QUS feature of QuantUS also supports a CLI for Terason and Canon transducers. More information and an example can be found in the Jupyter notebooks `scCanonQus.ipynb`_ and `terasonQus.ipynb`_ within our codebase.

.. _scCanonQus.ipynb: https://github.com/TUL-DEV/QuantUS/blob/main/CLI-Demos/scCanonQus.ipynb
.. _terasonQus.ipynb: https://github.com/TUL-DEV/QUantUS/blob/main/CLI-Demos/terasonQus.ipynb
.. _El Kaffas et al.: https://pubmed.ncbi.nlm.nih.gov/26233222/
.. image:: mbfSc.png
   :width: 600
   :alt: MBF Parametric Map Example
   :align: center

DCE-US Overview
===============

For both 2D and 3D cine loops, QuantUS performs quantitative analysis on bolus contrast injections by computing a time intensity curve (TIC) for a given ROI or volume of interest (VOI). 2D cine loops can optionally run a 2D motion compensation algorithm developed by `Tiyarattanachai et al.`_ before the TIC is computed to reduce motion-induced noise.

From here, a lognormal curve is fitted, returning the area under the curve (AUC), peak enhancement (PE), mean transit time (MTT), time to peak (TP), normalization (TMPPV), and region area/volume values. For processors with high computational power, a parametric map of each parameter can be generated in both 2D and 3D as well.

.. _Tiyarattanachai et al.: https://pubmed.ncbi.nlm.nih.gov/35970658/
.. image:: 3dDceusParamap.png
   :width: 600
   :alt: 3D DCE-US Parametric Map Example
   :align: center
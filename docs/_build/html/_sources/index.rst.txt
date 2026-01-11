bngmetric
=========

An open source Python implementation of the UK's Biodiversity Net Gain (BNG) Statutory Metric.

.. image:: https://img.shields.io/badge/python-3.12+-blue.svg
   :target: https://www.python.org/downloads/

.. image:: https://img.shields.io/badge/JAX-enabled-green.svg
   :target: https://github.com/google/jax


Overview
--------

**bngmetric** provides two complementary capabilities:

1. **Python Implementation of the BNG Metric**

   A straightforward, validated implementation of the UK's Statutory Biodiversity Metric.
   Use it as a drop-in replacement for spreadsheet-based calculations, with the benefits
   of programmatic access, automation, and integration with existing data pipelines.

2. **JAX-Powered Advanced Analysis**

   Built on JAX, enabling gradient-based methods for sophisticated analyses that aren't
   possible with traditional spreadsheet tools:

   - **Sensitivity analysis**: Compute gradients to understand how changes in area,
     condition, or habitat type affect biodiversity units
   - **Uncertainty propagation**: Propagate classification uncertainty (e.g., from
     remote sensing or ecologist assessments) through the metric using probabilistic methods
   - **Optimisation**: Find optimal habitat creation/enhancement strategies subject to constraints


Installation
------------

.. code-block:: bash

   pip install bngmetric

Or install from source:

.. code-block:: bash

   git clone https://github.com/pdh21/bngmetric.git
   cd bngmetric
   pip install -e .


Quick Start
-----------

Basic Usage
^^^^^^^^^^^

Calculate biodiversity units for a habitat parcel:

.. code-block:: python

   import pandas as pd
   from bngmetric.core_calculations import calculate_bng_from_dataframe

   # Define baseline habitats
   baseline = pd.DataFrame({
       'Habitat': ['Grassland - Lowland meadows', 'Woodland - Lowland mixed deciduous woodland'],
       'Condition': ['Moderate', 'Good'],
       'Area': [2.5, 1.0],
       'Strategic_Significance': [1.15, 1.0]
   })

   # Calculate baseline biodiversity units
   units = calculate_bng_from_dataframe(baseline)
   print(f"Baseline units: {units:.2f}")


Habitat Creation
^^^^^^^^^^^^^^^^

.. code-block:: python

   from bngmetric.creation import calculate_creation_bng_from_dataframe

   # Define habitat creation proposal
   creation = pd.DataFrame({
       'Habitat': ['Grassland - Lowland meadows'],
       'Condition': ['Good'],
       'Area': [3.0],
       'Strategic_Significance': [1.15]
   })

   units = calculate_creation_bng_from_dataframe(creation)
   print(f"Creation units: {units:.2f}")


Off-site with Spatial Risk
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from bngmetric.creation import calculate_offsite_creation_bng_from_dataframe

   # Off-site creation with spatial risk multiplier
   offsite = pd.DataFrame({
       'Habitat': ['Grassland - Lowland meadows'],
       'Condition': ['Good'],
       'Area': [3.0],
       'Strategic_Significance': [1.15],
       'Spatial_Risk': ['Neighbouring LPA/NCA']  # 0.75 multiplier
   })

   units = calculate_offsite_creation_bng_from_dataframe(offsite)


Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   usage/basic
   usage/creation_enhancement
   usage/offsite

.. toctree::
   :maxdepth: 2
   :caption: Advanced Analysis

   advanced/sensitivity
   advanced/uncertainty

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/core
   api/creation
   api/enhancement
   api/constants


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

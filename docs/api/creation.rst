Creation API
============

The ``creation`` module provides functions for calculating biodiversity units
from habitat creation, including both on-site and off-site scenarios.

.. module:: bngmetric.creation


On-site Creation Functions
--------------------------

.. autofunction:: get_temporal_multiplier

.. autofunction:: get_difficulty_multiplier

.. autofunction:: calculate_creation_bng_unit

.. autofunction:: calculate_batched_creation_bng_units

.. autofunction:: calculate_creation_bng_from_dataframe

.. autofunction:: calculate_total_creation_units_from_jax_arrays


Off-site Creation Functions
---------------------------

.. autofunction:: get_spatial_risk_multiplier

.. autofunction:: calculate_offsite_creation_bng_unit

.. autofunction:: calculate_batched_offsite_creation_bng_units

.. autofunction:: calculate_offsite_creation_bng_from_dataframe

.. autofunction:: calculate_total_offsite_creation_units_from_jax_arrays


DataFrame Column Requirements
-----------------------------

On-site Creation
^^^^^^^^^^^^^^^^

- ``Habitat``: Habitat type string (must match ``HABITAT_TYPE_TO_ID`` keys)
- ``Condition``: Target condition string
- ``Area``: Area in hectares
- ``Strategic_Significance``: Strategic significance multiplier (1.0, 1.1, or 1.15)

Off-site Creation
^^^^^^^^^^^^^^^^^

All on-site columns plus:

- ``Spatial_Risk``: One of:
   - ``'Inside LPA/NCA'`` (multiplier = 1.0)
   - ``'Neighbouring LPA/NCA'`` (multiplier = 0.75)
   - ``'Beyond neighbouring LPA/NCA'`` (multiplier = 0.5)

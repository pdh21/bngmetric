Sensitivity Analysis
====================

One of the key advantages of bngmetric's JAX implementation is the ability to
compute **exact gradients** of biodiversity units with respect to input
parameters. This enables rigorous sensitivity analysis to understand which
factors most influence your BNG calculations.


Why Sensitivity Analysis?
-------------------------

Understanding sensitivity helps answer questions like:

- Which parcels contribute most to total biodiversity units?
- How much does a 10% increase in area affect the outcome?
- Which habitat types are most sensitive to condition changes?
- Where should I focus habitat quality improvements for maximum gain?


Computing Gradients with JAX
----------------------------

JAX's automatic differentiation computes exact gradients efficiently:

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from bngmetric.core_calculations import calculate_total_bng_from_jax_arrays
   from bngmetric.constants import HABITAT_TYPE_TO_ID, CONDITION_CATEGORY_TO_ID

   # Define habitat parcels
   habitat_ids = jnp.array([
       HABITAT_TYPE_TO_ID['Grassland - Lowland meadows'],
       HABITAT_TYPE_TO_ID['Woodland - Lowland mixed deciduous woodland'],
       HABITAT_TYPE_TO_ID['Heathland and shrub - Lowland heathland']
   ])
   condition_ids = jnp.array([
       CONDITION_CATEGORY_TO_ID['Moderate'],
       CONDITION_CATEGORY_TO_ID['Good'],
       CONDITION_CATEGORY_TO_ID['Poor']
   ])
   areas = jnp.array([2.5, 1.0, 0.5])
   strategic = jnp.array([1.15, 1.0, 1.1])

   # Define function to differentiate
   def total_units(areas_param):
       return calculate_total_bng_from_jax_arrays(
           habitat_ids, condition_ids, areas_param, strategic
       )

   # Compute gradient with respect to area
   grad_fn = jax.grad(total_units)
   gradients = grad_fn(areas)

   print("Gradient of total units with respect to each parcel's area:")
   for i, (area, grad) in enumerate(zip(areas, gradients)):
       print(f"  Parcel {i}: area={area:.1f}ha, dUnits/dArea={grad:.2f}")


Interpreting Gradients
----------------------

The gradient tells you the **marginal change** in biodiversity units per unit
change in the input:

.. code-block:: python

   # A gradient of 5.0 means:
   # Increasing this parcel's area by 1 hectare increases total units by 5.0

   # Identify the most sensitive parcel
   most_sensitive = jnp.argmax(gradients)
   print(f"Parcel {most_sensitive} has the highest marginal contribution")


Sensitivity to Strategic Significance
-------------------------------------

.. code-block:: python

   def total_units_strategic(strategic_param):
       return calculate_total_bng_from_jax_arrays(
           habitat_ids, condition_ids, areas, strategic_param
       )

   grad_strategic = jax.grad(total_units_strategic)
   grads = grad_strategic(strategic)

   print("Sensitivity to strategic significance multiplier:")
   for i, grad in enumerate(grads):
       print(f"  Parcel {i}: dUnits/dStrategic={grad:.2f}")


Value and Gradient Together
---------------------------

Use ``jax.value_and_grad`` to compute both the function value and gradient
in a single pass:

.. code-block:: python

   value_and_grad_fn = jax.value_and_grad(total_units)
   total, gradients = value_and_grad_fn(areas)

   print(f"Total units: {total:.2f}")
   print(f"Gradients: {gradients}")


Sensitivity for Creation/Enhancement
------------------------------------

The same approach works for creation and enhancement calculations:

.. code-block:: python

   from bngmetric.creation import calculate_total_creation_units_from_jax_arrays

   def creation_units(areas_param):
       return calculate_total_creation_units_from_jax_arrays(
           habitat_ids, condition_ids, areas_param, strategic
       )

   grad_creation = jax.grad(creation_units)
   creation_grads = grad_creation(areas)


Jacobian for Multiple Outputs
-----------------------------

For per-parcel gradients (Jacobian matrix):

.. code-block:: python

   from bngmetric.core_calculations import calculate_batched_baseline_bng_units

   def parcel_units(areas_param):
       return calculate_batched_baseline_bng_units(
           habitat_ids, condition_ids, areas_param, strategic
       )

   # Jacobian: d(each output) / d(each input)
   jacobian = jax.jacobian(parcel_units)(areas)
   print(f"Jacobian shape: {jacobian.shape}")  # (n_parcels, n_parcels)


Practical Applications
----------------------

**1. Identifying High-Value Interventions**

.. code-block:: python

   # Which parcel gives best return on additional area?
   best_parcel = jnp.argmax(gradients)
   print(f"Prioritise expanding parcel {best_parcel}")

**2. Optimising Habitat Distribution**

.. code-block:: python

   # Use gradients to guide optimisation
   from jax.example_libraries import optimizers

   # Example: maximise units subject to total area constraint
   # (Requires custom optimisation loop)

**3. Comparing Scenarios**

.. code-block:: python

   # Compare sensitivity between creation and enhancement
   creation_sensitivity = jax.grad(creation_units)(areas)
   enhancement_sensitivity = jax.grad(enhancement_units)(areas)

   for i in range(len(areas)):
       print(f"Parcel {i}: Creation={creation_sensitivity[i]:.2f}, "
             f"Enhancement={enhancement_sensitivity[i]:.2f}")

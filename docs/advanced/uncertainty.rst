Uncertainty Propagation
=======================

A powerful application of bngmetric's JAX foundation is **probabilistic
propagation of uncertainty** through the BNG metric. This enables rigorous
quantification of how input uncertainties affect biodiversity unit calculations.


Sources of Uncertainty
----------------------

BNG assessments involve several sources of uncertainty:

1. **Habitat Classification Uncertainty**

   - Remote sensing classification errors
   - Ecologist misidentification
   - Boundary delineation errors

2. **Condition Assessment Uncertainty**

   - Subjective condition scoring
   - Temporal variability
   - Sampling limitations

3. **Area Measurement Uncertainty**

   - GIS digitisation errors
   - Survey precision

4. **Temporal Uncertainty**

   - Will the habitat reach target condition?
   - How long will it actually take?


Confusion Matrix Approach
-------------------------

For habitat classification from remote sensing or field surveys, uncertainty
is often characterised by a **confusion matrix** showing the probability of
each true class being assigned to each predicted class.

.. code-block:: python

   import jax.numpy as jnp

   # Example confusion matrix (3 habitat types)
   # Rows = true class, Columns = predicted class
   confusion_matrix = jnp.array([
       [0.85, 0.10, 0.05],  # True: Grassland
       [0.08, 0.88, 0.04],  # True: Woodland
       [0.05, 0.05, 0.90],  # True: Heathland
   ])

   # Each row sums to 1.0 (probability distribution over predictions)


Monte Carlo Propagation
-----------------------

Use Monte Carlo sampling to propagate classification uncertainty:

.. code-block:: python

   import jax
   import jax.numpy as jnp
   import jax.random as random
   from bngmetric.core_calculations import calculate_batched_baseline_bng_units
   from bngmetric.constants import HABITAT_TYPE_TO_ID, CONDITION_CATEGORY_TO_ID

   # Define the observed (potentially misclassified) habitats
   observed_habitat_ids = jnp.array([0, 1, 2])  # What we classified
   condition_ids = jnp.array([2, 0, 4])  # Moderate, Good, Poor
   areas = jnp.array([2.5, 1.0, 0.5])
   strategic = jnp.array([1.15, 1.0, 1.1])

   # Confusion matrix (probability of true class given observed class)
   # In practice, derive this from validation data
   confusion = jnp.array([
       [0.85, 0.10, 0.05],
       [0.08, 0.88, 0.04],
       [0.05, 0.05, 0.90],
   ])

   def sample_true_habitats(key, observed_ids, confusion_matrix):
       """Sample true habitat IDs given observed IDs and confusion matrix."""
       true_ids = []
       for obs_id in observed_ids:
           subkey, key = random.split(key)
           # Sample from confusion matrix row for this observed class
           probs = confusion_matrix[obs_id]
           true_id = random.categorical(subkey, jnp.log(probs))
           true_ids.append(true_id)
       return jnp.array(true_ids)

   # Monte Carlo simulation
   n_samples = 1000
   key = random.PRNGKey(42)
   results = []

   for i in range(n_samples):
       key, subkey = random.split(key)
       true_habitats = sample_true_habitats(subkey, observed_habitat_ids, confusion)
       units = calculate_batched_baseline_bng_units(
           true_habitats, condition_ids, areas, strategic
       )
       results.append(jnp.sum(units))

   results = jnp.array(results)

   print(f"Mean units: {jnp.mean(results):.2f}")
   print(f"Std dev: {jnp.std(results):.2f}")
   print(f"95% CI: [{jnp.percentile(results, 2.5):.2f}, {jnp.percentile(results, 97.5):.2f}]")


Vectorised Monte Carlo
----------------------

For efficiency, vectorise across samples using ``vmap``:

.. code-block:: python

   from functools import partial

   @partial(jax.vmap, in_axes=(0, None, None, None, None))
   def compute_units_sample(habitat_ids, condition_ids, areas, strategic, key):
       """Compute units for one sample of habitat classifications."""
       return jnp.sum(calculate_batched_baseline_bng_units(
           habitat_ids, condition_ids, areas, strategic
       ))

   # Generate all samples at once
   keys = random.split(key, n_samples)
   # ... sample habitats for each key
   # ... compute all results in parallel


Condition Uncertainty
---------------------

Similarly, propagate uncertainty in condition assessments:

.. code-block:: python

   # Condition confusion matrix
   # True condition vs assessed condition
   condition_confusion = jnp.array([
       [0.80, 0.15, 0.05, 0.00, 0.00],  # True: Good
       [0.10, 0.75, 0.10, 0.05, 0.00],  # True: Fairly Good
       [0.05, 0.15, 0.65, 0.10, 0.05],  # True: Moderate
       [0.00, 0.05, 0.15, 0.70, 0.10],  # True: Fairly Poor
       [0.00, 0.00, 0.05, 0.15, 0.80],  # True: Poor
   ])

   # Sample true conditions given assessed conditions
   # Then propagate through the metric


Area Uncertainty
----------------

For continuous parameters like area, use Gaussian uncertainty:

.. code-block:: python

   # Areas with uncertainty (mean, std)
   area_means = jnp.array([2.5, 1.0, 0.5])
   area_stds = jnp.array([0.1, 0.05, 0.02])  # e.g., 4% relative uncertainty

   def sample_areas(key, means, stds):
       return means + stds * random.normal(key, shape=means.shape)

   # Monte Carlo with area uncertainty
   results = []
   for i in range(n_samples):
       key, subkey = random.split(key)
       sampled_areas = sample_areas(subkey, area_means, area_stds)
       sampled_areas = jnp.maximum(sampled_areas, 0)  # Ensure non-negative
       units = calculate_batched_baseline_bng_units(
           habitat_ids, condition_ids, sampled_areas, strategic
       )
       results.append(jnp.sum(units))


Combined Uncertainty
--------------------

Propagate multiple sources of uncertainty simultaneously:

.. code-block:: python

   def sample_all_inputs(key, observed_habitats, assessed_conditions,
                         area_means, area_stds, habitat_confusion, condition_confusion):
       keys = random.split(key, 3)

       # Sample true habitats
       true_habitats = sample_true_habitats(keys[0], observed_habitats, habitat_confusion)

       # Sample true conditions
       true_conditions = sample_true_conditions(keys[1], assessed_conditions, condition_confusion)

       # Sample areas
       true_areas = sample_areas(keys[2], area_means, area_stds)

       return true_habitats, true_conditions, true_areas

   # Full uncertainty propagation
   all_results = []
   for i in range(n_samples):
       key, subkey = random.split(key)
       habs, conds, areas = sample_all_inputs(
           subkey, observed_habitat_ids, observed_condition_ids,
           area_means, area_stds, habitat_confusion, condition_confusion
       )
       units = jnp.sum(calculate_batched_baseline_bng_units(
           habs, conds, areas, strategic
       ))
       all_results.append(units)


Reporting Uncertainty
---------------------

Present results with appropriate uncertainty bounds:

.. code-block:: python

   results = jnp.array(all_results)

   report = {
       'mean': float(jnp.mean(results)),
       'std': float(jnp.std(results)),
       'median': float(jnp.median(results)),
       'ci_lower': float(jnp.percentile(results, 2.5)),
       'ci_upper': float(jnp.percentile(results, 97.5)),
       'p10': float(jnp.percentile(results, 10)),
       'p90': float(jnp.percentile(results, 90)),
   }

   print(f"Biodiversity Units: {report['mean']:.1f} "
         f"(95% CI: {report['ci_lower']:.1f} - {report['ci_upper']:.1f})")


Integration with NumPyro
------------------------

For more sophisticated Bayesian inference, bngmetric integrates with NumPyro:

.. code-block:: python

   import numpyro
   import numpyro.distributions as dist
   from numpyro.infer import MCMC, NUTS

   def bng_model(observed_habitats, areas, strategic):
       # Prior on true habitat classifications
       habitat_probs = numpyro.sample(
           'habitat_probs',
           dist.Dirichlet(jnp.ones(n_habitat_types))
       )

       # Sample true habitats
       true_habitats = numpyro.sample(
           'true_habitats',
           dist.Categorical(probs=habitat_probs),
           sample_shape=(len(observed_habitats),)
       )

       # Compute units (deterministic given habitats)
       units = calculate_batched_baseline_bng_units(
           true_habitats, condition_ids, areas, strategic
       )

       return jnp.sum(units)

   # Run MCMC for posterior inference
   # ...

This enables full Bayesian treatment of classification uncertainty, including
incorporation of prior ecological knowledge.

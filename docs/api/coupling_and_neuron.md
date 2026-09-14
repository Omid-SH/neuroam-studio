# Coupling and NEURON

Field-to-compartment sampling, SWC morphology and anatomical registration,
and the guarded NEURON driver. See
[Design § 6](../DESIGN.md#6-coupling-and-neuron) and
[§ 7](../DESIGN.md#7-anatomical-registration).

```{note}
`neuroam.neuron_link` imports the `neuron` package lazily, inside the
functions that need it — this page builds and is importable without NEURON
installed, but calling most of these functions still requires
`pip install neuron`.
```

## `neuroam.coupling`

```{eval-rst}
.. automodule:: neuroam.coupling
   :members:
   :undoc-members:
   :show-inheritance:
```

## `neuroam.morphology`

```{eval-rst}
.. automodule:: neuroam.morphology
   :members:
   :undoc-members:
   :show-inheritance:
```

## `neuroam.neuron_link`

```{eval-rst}
.. automodule:: neuroam.neuron_link
   :members:
   :undoc-members:
   :show-inheritance:
```

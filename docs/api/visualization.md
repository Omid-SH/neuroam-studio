# Visualization

2D matplotlib plots, the interactive PyVista viewer, and the composable
Plotly `Scene` (context/field/activity views, the live placement editor).
See [Design § 9](../DESIGN.md#9-visualization-viz3dscene).

```{note}
`neuroam.viewer` needs `pyvista` and `neuroam.viz3d` needs `plotly` +
`scikit-image` at call time (both imported lazily) — this page builds
without either installed, matching how the rest of the package stays
importable with only numpy/scipy.
```

## `neuroam.viz`

```{eval-rst}
.. automodule:: neuroam.viz
   :members:
   :undoc-members:
   :show-inheritance:
```

## `neuroam.viewer`

```{eval-rst}
.. automodule:: neuroam.viewer
   :members:
   :undoc-members:
   :show-inheritance:
```

## `neuroam.viz3d`

```{eval-rst}
.. automodule:: neuroam.viz3d
   :members:
   :undoc-members:
   :show-inheritance:
```

"""NeuroAM Studio — config-driven multiscale Admittance Method + NEURON pipeline.

Core objects:

- :class:`neuroam.materials.MaterialLibrary` / :class:`neuroam.materials.Material`
- :class:`neuroam.model.VoxelModel` — voxel anatomy + electrodes (legacy-compatible)
- :func:`neuroam.assembly.assemble` — direct sparse admittance assembly (no netlist)
- :func:`neuroam.solver.solve` / :func:`neuroam.solver.solve_basis` — fields
- :mod:`neuroam.fields` — E, J, exports, ROI dose metrics
- :mod:`neuroam.coupling` / :mod:`neuroam.neuron_link` — field-to-NEURON
- :func:`neuroam.pipeline.run` — JSON-config pipeline (ASCENT-style)
"""

__version__ = "0.4.0"

from .materials import Material, MaterialLibrary            # noqa: F401
from .model import VoxelModel, Source                        # noqa: F401
from .assembly import assemble, assemble_uniform, assemble_mrm, System  # noqa: F401
from .solver import solve, solve_basis, superpose, SolveResult  # noqa: F401
from .waveforms import Waveform                              # noqa: F401

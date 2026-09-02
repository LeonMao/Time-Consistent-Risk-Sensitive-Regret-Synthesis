# Third-party and data provenance

This archive contains project source code, project-generated CSV/JSON results,
and project-generated or author-drawn reference figures. It does not bundle
third-party Python packages; those packages are resolved from their upstream
distributions using `uv.lock` or `requirements-research.txt` and remain subject
to their own licenses.

The Intel Research Lab benchmark in
`03_EXPERIMENT_AND_BENCHMARK_CODE/stage3_3_intel_lab_benchmark.py` is an
author-created schematic topological abstraction of public occupancy-map
geometry. The original occupancy-map image or dataset is not redistributed.
The associated topology result figure is generated from this abstraction.

The first three method figures are original author drawings and are distributed
as final PNG assets. No external artwork and no method-figure generation script
is included.

The direct runtime dependencies are NumPy, SciPy, psutil, pandas, Matplotlib,
and Pillow. Consult each upstream distribution for its license and notices.

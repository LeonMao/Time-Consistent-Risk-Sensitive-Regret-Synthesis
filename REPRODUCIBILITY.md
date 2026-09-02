# Reproducibility protocol

## Environment

The lockfile targets CPython 3.13.5 and uv 0.11.24. Direct dependencies are
exactly pinned in `pyproject.toml`, `uv.lock`, and
`requirements-research.txt`. The top-level runner fixes common BLAS thread
counts, `PYTHONHASHSEED`, and Matplotlib's noninteractive backend.

Use a fresh output directory for each run. The runner refuses a nonempty output
directory unless `--allow-existing` is supplied.

## Verification levels

1. **Core method tests.** `run_core_tests.py` checks automata, posterior and
   knowledge updates, hard feasibility, nested risk recursion, regret-oracle
   behavior, and explicit/factored solver agreement.
2. **Deterministic scientific recomputation.** The unified driver reconstructs
   every Robotica Stage 3.3 P0/P1/P2.3 table input and the final Stage 3.5
   evidence. `verify_robotica_recomputed.py` requires all 17 Stage 3.3 files;
   `verify_reproduced_results.py` checks the Stage 3.5 set. Numeric CSV fields
   use a tight tolerance only to absorb decimal serialization, while categorical
   fields, keys, row sets, and completeness are exact.
3. **Claim verification.** `verify_robotica_claims.py` independently recomputes
   the headline statistics from frozen evidence and asserts 11 claim groups.
4. **Figure verification.** The seven data-driven paper figures are regenerated
   from included evidence and checked against the packaged canvases. Renderer
   anti-aliasing may differ slightly across systems; the verifier enforces the
   documented pixel bounds. The first three method illustrations are
   author-drawn PNGs and therefore have no plotting scripts.
5. **Release integrity.** `verify_release.py` rejects manuscripts, office/PDF
   files, caches, temporary outputs, ZIPs, and the method-figure generator; it
   then verifies the exact file inventory against `SHA256SUMS.txt`.

## Runtime evidence

Operation counts, policy equality, objectives, satisfaction flags, and
deterministic table values are recomputable scientific evidence. Wall-clock
timing is hardware- and operating-system-dependent. The paper's timing table is
therefore preserved as a controlled five-seed measurement with warmups,
repetitions, medians, and bootstrap intervals. A new machine should reproduce
the qualitative/structural comparison, not bit-identical seconds.

## Provenance boundary

The public Stage 3.3 batch drivers were reconstructed from the released core,
benchmark definitions, frozen parameter specifications, and reference tables;
they deterministically regenerate all 17 reference files. This is transparent
scientific reconstruction, not a claim that the public driver source is a
byte-for-byte copy of an earlier internal orchestration script. The Stage 3.5
driver is the final preserved experiment chain.

The formal arguments in the paper remain mathematical proofs rather than
machine-checked proof artifacts. The package verifies their executable premises
and all reported experimental conclusions.

## Recommended clean acceptance sequence

```bash
uv run --locked --isolated python verify_release.py
uv run --locked --isolated python reproduce.py --quick
uv run --locked --isolated python reproduce.py --full --output-root artifacts/full
uv run --locked --isolated python reproduce.py --figures --output-root artifacts/figures
```

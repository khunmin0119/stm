# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Automated Strut-and-Tie Model (STM) generation for deep concrete beams using a Multi-Agent System (MAS) that combines SIMP topology optimization, local LLM agents (Ollama), and code-based structural verification per KDS 14 20 24 (Korean Design Standards).

## Running the Pipeline

**Prerequisites:** Ollama running locally at `http://localhost:11434` with models `deepseek-r1:14b` and `qwen2.5:7b` (or `qwen3:32b`).

```bash
pip install -r requirements.txt   # numpy, matplotlib, anthropic (anthropic listed but Ollama is actually used)
pip install scipy                 # needed by cdr_calc.py
```

**Main entry point** — `mas_stm_final.py`:
```python
from mas_stm_final import MASSTMGenerator
mas = MASSTMGenerator(topology_model="deepseek-r1:14b", reviewer_model="qwen2.5:7b",
                      max_retries=5, n_candidates=3, output_dir="plots_8node")
result = mas.generate(beam_length=6900, beam_height=2000, beam_width=500,
                      loads=[(2225, -2000), (4675, -2000)],
                      support_positions=[(225, 'pin'), (6675, 'roller')], verbose=True)
```

**Nodal zone verification** — `nodal_zone_design.py`:
```python
from nodal_zone_design import design_all_candidates
result = design_all_candidates(json_path='stm_result.json', fck=27, fy=400, bw=500,
                                bearing_plate=450, cover=40, stirrup_dia=16, main_bar_dia=32,
                                phi=0.75, save_dir='plots_results', verbose=True)
```

## Architecture

The pipeline runs in sequential phases inside `MASSTMGenerator.generate()`:

1. **Phase 0 — SIMP Topology Optimization** (`simp_optimize()`): Density-based optimization identifies natural force paths in the beam. Parameters: `volfrac=0.3`, `penal=3.0`, `rmin=1.5`.

2. **Phase 1A — LLM Node Layout** (`_decide_node_layout()`): LLM interprets SIMP results to decide intermediate node count (0/1/2 pairs) and positions. Uses `OllamaClient` with `force_json=True` for structured output.

3. **Phase 1B — Ground Structure** (`enumerate_ground_structure()`): Deterministic enumeration of all valid member connections with constraints (angle 25-65°, no crossing).

4. **Phase 2 — Engineering Review** (`_engineering_review()`): LLM reviewer scores design on load path continuity, force flow angles, symmetry, and completeness.

5. **Phase 3 — Truss Analysis** (`solve_truss()`): Direct stiffness method solves 2D truss. Scoring metric: `Sum(|Tension| × Length)` — lower is better.

6. **Phase 4 — Nodal Zone Design** (`nodal_zone_design.py`): KDS 14 20 24 verification of strut/tie widths, node type classification (CCC/CCT/CTT).

## Key Data Structures

```python
# STM topology
stm = {
    'nodes': {'A': (x, y), ...},              # node_id -> (x_mm, y_mm)
    'connections': [['A', 'B'], ...],          # member pairs
    'supports': {'A': 'pin', 'F': 'roller'},
}

# Analysis results
result = {
    'member_forces': {('A', 'B'): force_N, ...},  # + tension, - compression
    'reactions': {'A': (Rx_N, Ry_N), ...},
}
```

Output is saved to `stm_result.json` with all candidates and the best-scoring STM.

## Key Files

| File | Purpose |
|------|---------|
| `mas_stm_final.py` | Main pipeline: SIMP + LLM agents + truss analysis |
| `mas_stm_cooper_ver6_4.py` | Grammar-based rule selection variant |
| `nodal_zone_design.py` | KDS 14 20 24 nodal zone verification & visualization |
| `cdr_calc.py` | Strut/tie width calculation with Nelder-Mead optimization |
| `archive/` | Previous implementation versions |

## Language & Comments

Code comments and commit messages are in Korean. The project follows Korean structural engineering conventions (KDS standards, units in mm/kN/MPa).

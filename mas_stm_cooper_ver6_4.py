"""
MAS-based STM Generator — Grammar-based Version 6.4
====================================================
Pipeline:
  Phase 0:   SIMP topology optimization (code)
  Phase 1:   Grammar-based STM generation
             LLM selects rules, code executes them
             Rules: DIRECT_STRUT, SPLIT_SPAN, ADD_VERTICAL, DONE
  Phase 2:   KDS feedback loop (code + LLM)
  Phase 3:   Truss analysis + scoring (code)

Agents:
  - Designer:  qwen3:32b (grammar rule selection)
  - Builder:   code (rule execution)
  - Analyzer:  code (truss + KDS)
"""

import json
import math
import re
import time
import urllib.request
import os
from copy import deepcopy
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Global font setting (Times New Roman with fallbacks)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'STIXGeneral', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'


# ===========================================================
# 0-A. Geometry Helpers
# ===========================================================

def compute_chord_positions(H, cover, stirrup_dia, main_bar_dia,
                            b=None, loads=None, support_positions=None,
                            fck=27, fy=400):
    """
    Compute top/bottom chord y-coordinates.

    Bottom chord (tie centroid):
        wt = 2 * (cover + stirrup + bar + bar/2), rounded up to 50mm
        y_bot = wt / 2

    Top chord (compression zone centroid):
        Computed from equivalent stress block depth:
        1. Mu = max factored moment from statics
        2. Solve for As from φ·As·fy·(d - As·fy/(2·0.85·fck·b)) = Mu
        3. a = As·fy / (0.85·fck·b)
        4. c = a / β1
        5. ws = ceil(c / 50) * 50
        6. y_top = H - ws / 2

    If loads are not provided, falls back to geometry-based estimate.

    Returns:
        y_bot, y_top, wt, ws
    """
    # Bottom chord (unchanged)
    wt = 2 * (cover + stirrup_dia + main_bar_dia + main_bar_dia / 2)
    wt = math.ceil(wt / 50) * 50
    y_bot = wt / 2

    # Top chord: stress block method if loads are provided
    if b is not None and loads is not None and support_positions is not None:
        # Effective depth
        d = H - cover - stirrup_dia - main_bar_dia / 2

        # Compute reactions from statics
        sorted_sup = sorted(support_positions, key=lambda s: s[0])
        x_left = sorted_sup[0][0]
        x_right = sorted_sup[-1][0]
        span = x_right - x_left

        if span > 0:
            # Ry_right from moment about left support
            M_about_left = sum(abs(fy_kN) * (lx - x_left) for lx, fy_kN in loads)
            Ry_right = M_about_left / span  # kN
            Ry_left = sum(abs(fy_kN) for _, fy_kN in loads) - Ry_right

            # Max moment at each load point (kN·mm)
            sorted_loads = sorted(loads, key=lambda l: l[0])
            Mu_max = 0
            for lx, lfy in sorted_loads:
                M = Ry_left * (lx - x_left)  # kN × mm = kN·mm
                for lx2, lfy2 in sorted_loads:
                    if lx2 < lx:
                        M -= abs(lfy2) * (lx - lx2)
                Mu_max = max(Mu_max, abs(M))

            # Convert to N·mm
            Mu_Nmm = Mu_max * 1e3  # kN·mm → N·mm

            # Solve quadratic for As
            # φ·As·fy·(d - As·fy/(2·0.85·fck·b)) = Mu
            # → k·As² - d·As + Mu/(φ·fy) = 0
            phi_flex = 0.85  # flexural strength reduction factor
            k = fy / (2 * 0.85 * fck * b)
            c_coeff = Mu_Nmm / (phi_flex * fy)

            discriminant = d ** 2 - 4 * k * c_coeff

            if discriminant >= 0:
                As = (d - math.sqrt(discriminant)) / (2 * k)
                a = As * fy / (0.85 * fck * b)
                # β1 (KDS)
                if fck <= 28:
                    beta1 = 0.85
                else:
                    beta1 = max(0.65, 0.85 - 0.007 * (fck - 28))
                c_depth = a / beta1
                ws = math.ceil(c_depth / 50) * 50
                # Minimum ws
                ws_min = 2 * (cover + stirrup_dia + main_bar_dia)
                ws = max(ws, ws_min)
                y_top = H - ws / 2
                return y_bot, y_top, wt, ws

    # Fallback: geometry-based estimate
    ws = 2 * (cover + stirrup_dia + main_bar_dia)
    ws = math.ceil(ws / 50) * 50
    y_top = H - ws / 2

    return y_bot, y_top, wt, ws


def generate_node_id(index):
    """
    Generate node ID from integer index.
    0-25 → A-Z, 26-51 → AA-AZ, 52-77 → BA-BZ, ...
    """
    if index < 26:
        return chr(ord('A') + index)
    else:
        first = chr(ord('A') + (index // 26) - 1)
        second = chr(ord('A') + (index % 26))
        return first + second


def detect_load_symmetry(loads, beam_length, tol=100):
    """
    Detect if loads are symmetric about beam center.

    Args:
        loads: list of (x_mm, Fy_kN)
        beam_length: total beam length in mm
        tol: tolerance in mm (default 100)

    Returns:
        True if all load x-positions have a mirror counterpart (with matching Fy)
    """
    if not loads:
        return True

    mid = beam_length / 2.0
    sorted_loads = sorted(loads, key=lambda l: l[0])

    # Pair up from outside in
    n = len(sorted_loads)
    for i in range(n // 2):
        lx_left, fy_left = sorted_loads[i]
        lx_right, fy_right = sorted_loads[n - 1 - i]
        mirror_x = beam_length - lx_left
        if abs(lx_right - mirror_x) > tol:
            return False
        if abs(abs(fy_left) - abs(fy_right)) > 0.01 * max(abs(fy_left), 1):
            return False

    # Odd count: middle load must be exactly at center
    if n % 2 == 1:
        lx_mid, _ = sorted_loads[n // 2]
        if abs(lx_mid - mid) > tol:
            return False

    return True


def resolve_mirror_mode(mirror_mode, loads, beam_length):
    """
    Resolve mirror_mode to boolean decision.

    Args:
        mirror_mode: 'auto' | 'always' | 'never'
        loads, beam_length: used for 'auto' detection

    Returns:
        (use_mirror: bool, reason: str)
    """
    if mirror_mode == 'always':
        return True, 'forced by mirror_mode=always'
    if mirror_mode == 'never':
        return False, 'forced by mirror_mode=never'
    # auto: detect symmetry
    if detect_load_symmetry(loads, beam_length):
        return True, 'auto: symmetric loads detected'
    return False, 'auto: asymmetric loads detected'


# ===========================================================
# 0-B. SIMP Topology Optimization
# ===========================================================

def simp_optimize(nelx, nely, volfrac=0.3, penal=3.0, rmin=1.5, maxiter=80,
                  loads=None, supports=None, beam_length=6900, beam_height=2000):
    """SIMP topology optimization — finds optimal material distribution."""
    print(f"  [SIMP] Mesh: {nelx}x{nely}, vf={volfrac}")
    nu = 0.3
    k = np.array([1/2-nu/6, 1/8+nu/8, -1/4-nu/12, -1/8+3*nu/8,
                  -1/4+nu/12, -1/8-nu/8, nu/6, 1/8-3*nu/8])
    KE = (1.0/(1.0-nu**2)) * np.array([
        [k[0],k[1],k[2],k[3],k[4],k[5],k[6],k[7]],
        [k[1],k[0],k[7],k[6],k[5],k[4],k[3],k[2]],
        [k[2],k[7],k[0],k[5],k[6],k[3],k[4],k[1]],
        [k[3],k[6],k[5],k[0],k[7],k[2],k[1],k[4]],
        [k[4],k[5],k[6],k[7],k[0],k[1],k[2],k[3]],
        [k[5],k[4],k[3],k[2],k[1],k[0],k[7],k[6]],
        [k[6],k[3],k[4],k[1],k[2],k[7],k[0],k[5]],
        [k[7],k[2],k[1],k[4],k[3],k[6],k[5],k[0]]])
    ndof = 2*(nelx+1)*(nely+1)
    dx = beam_length/nelx
    nele = nelx*nely
    edofMat = np.zeros((nele, 8), dtype=int)
    for ey in range(nely):
        for ex in range(nelx):
            ei = ey*nelx+ex
            n1 = ey*(nelx+1)+ex; n2 = n1+1
            n3 = (ey+1)*(nelx+1)+ex+1; n4 = (ey+1)*(nelx+1)+ex
            edofMat[ei] = [2*n1,2*n1+1,2*n2,2*n2+1,2*n3,2*n3+1,2*n4,2*n4+1]
    iK = np.kron(edofMat, np.ones((1,8),dtype=int)).flatten().astype(int)
    jK = np.kron(edofMat, np.ones((8,1),dtype=int)).flatten().astype(int)
    F = np.zeros(ndof)
    if loads:
        for lx_mm, lfy_kN in loads:
            ix = max(0, min(nelx, int(round(lx_mm/dx))))
            F[2*(0*(nelx+1)+ix)+1] = -abs(lfy_kN)
    fixed = []
    if supports:
        for sx_mm, stype in supports:
            ix = max(0, min(nelx, int(round(sx_mm/dx))))
            node = nely*(nelx+1)+ix
            if stype=='pin': fixed.extend([2*node, 2*node+1])
            elif stype=='roller': fixed.append(2*node+1)
    fixed = np.array(sorted(set(fixed)))
    free = np.setdiff1d(np.arange(ndof), fixed)
    H = np.zeros((nele, nele))
    for i1 in range(nelx):
        for j1 in range(nely):
            e1 = j1*nelx+i1
            for i2 in range(max(i1-int(np.ceil(rmin)-1),0), min(i1+int(np.ceil(rmin)),nelx)):
                for j2 in range(max(j1-int(np.ceil(rmin)-1),0), min(j1+int(np.ceil(rmin)),nely)):
                    e2 = j2*nelx+i2
                    H[e1,e2] = max(0, rmin-np.sqrt((i1-i2)**2+(j1-j2)**2))
    Hs = H.sum(axis=1)
    x = volfrac*np.ones(nele); xPhys = x.copy()
    loop = 0; change = 1.0; Emin = 1e-9; E0 = 1.0
    while change > 0.01 and loop < maxiter:
        loop += 1
        sK = ((KE.flatten()[np.newaxis]).T*(Emin+xPhys**penal*(E0-Emin))).flatten(order='F')
        K = np.zeros((ndof,ndof)); np.add.at(K, (iK, jK), sK); K = 0.5*(K+K.T)
        u = np.zeros(ndof)
        try: u[free] = np.linalg.solve(K[np.ix_(free,free)], F[free])
        except: break
        ce = np.zeros(nele)
        for ei in range(nele): Ue = u[edofMat[ei]]; ce[ei] = Ue @ KE @ Ue
        c = np.sum((Emin+xPhys**penal*(E0-Emin))*ce)
        dc = -penal*xPhys**(penal-1)*(E0-Emin)*ce
        dv = np.ones(nele)
        dc = H @ (x*dc) / Hs / np.maximum(x, 1e-3)
        dv = H @ dv / Hs
        l1, l2 = 0.0, 1e9; move = 0.2
        while (l2-l1)/(l1+l2+1e-10) > 1e-3:
            lmid = 0.5*(l1+l2)
            xnew = np.maximum(0.001, np.maximum(x-move, np.minimum(1.0, np.minimum(x+move, x*np.sqrt(-dc/dv/lmid)))))
            xPhys = (H @ xnew) / Hs
            if xPhys.sum() > volfrac*nele: l1 = lmid
            else: l2 = lmid
        change = np.max(np.abs(xnew-x)); x = xnew.copy()
        if loop%10==0 or loop<=2:
            print(f"  [SIMP] iter={loop:3d} c={c:.2f} vol={xPhys.sum()/nele:.3f} ch={change:.4f}")
    print(f"  [SIMP] Done: {loop} iters")
    return {'density': xPhys.reshape(nely,nelx), 'nelx':nelx, 'nely':nely,
            'beam_length':beam_length, 'beam_height':beam_height}


def auto_simp_mesh(beam_length, beam_height, target_elem_size=100):
    """
    Auto-compute SIMP mesh dimensions from beam size.
    target_elem_size: approximate element edge length in mm (default 100mm).
    """
    nelx = max(20, int(round(beam_length / target_elem_size)))
    nely = max(10, int(round(beam_height / target_elem_size)))
    return nelx, nely


def summarize_simp_for_llm(simp_result, loads, supports, threshold=0.3):
    """
    Convert SIMP density to text description for LLM.
    Uses actual load/support positions (not hardcoded).
    """
    density = simp_result['density']
    nely, nelx = density.shape
    L = simp_result['beam_length']
    H = simp_result['beam_height']
    dx = L / nelx
    dy = H / nely

    # Sort loads and supports by x for consistent naming
    sorted_loads = sorted(loads, key=lambda p: p[0])
    sorted_supports = sorted(supports, key=lambda p: p[0])

    # For each column, find vertical extent of high-density diagonal regions
    top_band, bot_band = 2, 2
    diag_info = []

    for ix in range(nelx):
        col = density[:, ix]
        mid_high = []
        for j in range(top_band, nely - bot_band):
            if col[j] > threshold:
                y_mm = H - (j + 0.5) * dy
                mid_high.append(y_mm)
        if mid_high:
            x_mm = (ix + 0.5) * dx
            y_top_d = max(mid_high)
            y_bot_d = min(mid_high)
            diag_info.append((x_mm, y_bot_d, y_top_d))

    # Find where diagonal paths cross mid-height
    mid_y = H / 2
    mid_x = L / 2
    left_cross = [x for x, yb, yt in diag_info if x < mid_x and yb < mid_y < yt]
    right_cross = [x for x, yb, yt in diag_info if x >= mid_x and yb < mid_y < yt]

    left_x = int(np.mean(left_cross)) if left_cross else None
    right_x = int(np.mean(right_cross)) if right_cross else None

    lines = [
        "SIMP TOPOLOGY OPTIMIZATION RESULT:",
        f"  Volume fraction: {int(threshold*100+0.5)}% threshold applied",
        "",
        "IDENTIFIED FORCE PATHS:",
        f"  1. TOP CHORD: Horizontal compression band at y~{H}mm",
        f"  2. BOTTOM CHORD: Horizontal tension band at y~0mm",
    ]

    # Dynamically describe diagonal struts from actual load/support positions
    for i, (lx, lfy) in enumerate(sorted_loads):
        # Find nearest support
        nearest_sup = min(sorted_supports, key=lambda s: abs(s[0] - lx))
        lines.append(
            f"  {3+i}. DIAGONAL STRUT: From load (x={lx:.0f}) "
            f"toward support (x={nearest_sup[0]:.0f})"
        )

    # Describe force paths qualitatively (no specific x-values to avoid LLM misuse)
    lines.append(f"\n  Diagonal force paths run from each load toward the nearest support.")
    lines.append(f"  Intermediate nodes should be placed BETWEEN supports and loads,")
    lines.append(f"  inside the spans where diagonal struts form — not at the load points.")

    return "\n".join(lines)


def plot_simp_result(simp_result, loads=None, supports=None, save_path=None, title=None):
    """Visualize SIMP density."""
    density = simp_result['density']
    L, H = simp_result['beam_length'], simp_result['beam_height']
    fig, ax = plt.subplots(1,1,figsize=(14,5))
    ax.imshow(1.0-density, cmap='gray', origin='upper', extent=[0,L,0,H], aspect='auto', vmin=0, vmax=1)
    if loads:
        for lx, lfy in loads:
            ax.annotate('', xy=(lx,H-50), xytext=(lx,H+150), arrowprops=dict(arrowstyle='->',color='red',lw=2.5))
            ax.text(lx, H+200, f'{abs(lfy):.0f}kN', ha='center', fontsize=11, color='red', fontweight='bold')
    if supports:
        for sx, stype in supports:
            m = '^' if stype=='pin' else 'o'
            ax.plot(sx, 0, marker=m, markersize=15, color='#3b82f6', markeredgecolor='navy', markeredgewidth=2, zorder=5)
    ax.set_xlim(-50, L+50); ax.set_ylim(-100, H+300)
    ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)')
    ax.set_title(title or 'SIMP Result', fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  >> Saved: {save_path}")
    plt.close()


# ===========================================================
# 1. Ollama API Client
# ===========================================================

class OllamaClient:
    """Local Ollama API client"""

    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url

    def chat(self, model: str, system: str, user: str,
             temperature: float = 0.3, force_json: bool = True) -> str:
        url = f"{self.base_url}/api/chat"
        request_body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 8192
            }
        }
        if force_json:
            request_body["format"] = "json"

        payload = json.dumps(request_body)
        req = urllib.request.Request(
            url, data=payload.encode('utf-8'),
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                result = json.loads(resp.read().decode('utf-8'))
                return result['message']['content']
        except Exception as e:
            print(f"  [Ollama Error] {e}")
            return None

    def is_available(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except:
            return False


# ===========================================================
# 2. Agent Prompts
# ===========================================================

NODE_LAYOUT_PROMPT = """You are the NODE LAYOUT AGENT for Strut-and-Tie Model (STM) design.

Your job: Decide how many intermediate intermediate nodes to add and where to place them.
You will be given SIMP topology optimization results showing where forces flow in the beam.
Use this physical information to decide optimal node positions.

## CONTEXT
A deep beam STM has FIXED nodes (supports at bottom, loads at top) and INTERMEDIATE nodes.
Each intermediate x-position creates a vertical pair of nodes (one on top chord, one on bottom chord).
If loading is symmetric, intermediate positions must also be symmetric about beam center.

## WHAT YOU DECIDE
1. How many intermediate nodes: 0 (simplest) up to the maximum specified in the user prompt
2. The x-position of each intermediate node (left-half only if symmetric; right-half mirrors automatically)

## HOW TO USE SIMP RESULTS
- SIMP shows diagonal force paths as high-density regions
- Place intermediate nodes WHERE diagonal paths are strongest
- The intersection of diagonal paths with chord lines = ideal node positions
- Diagonal angles closer to 45 degrees = better force distribution

## OUTPUT FORMAT
{"n_intermediate": 0 or 1 or 2, "x_positions": [] or [x1] or [x1, x2], "reasoning": "how SIMP guided this"}"""


TOPOLOGY_AGENT_PROMPT = """You are the TOPOLOGY AGENT in a multi-agent Strut-and-Tie Model design team.

Your role: Interpret SIMP topology optimization results and propose an initial node layout.
You are the FIRST agent in the pipeline — your proposal will be critiqued and possibly revised.

## CONTEXT
A deep beam STM has FIXED nodes (supports, loads) and INTERMEDIATE nodes.
Each intermediate x-position creates a vertical pair of nodes (one on top chord, one on bottom chord).

## YOUR JOB
1. Read the SIMP density distribution description
2. Identify where diagonal force paths are strongest
3. Propose x-positions for intermediate intermediate nodes
4. Explain your reasoning — the Critic will scrutinize it

## OUTPUT FORMAT
{"n_intermediate": N, "x_positions": [...], "reasoning": "why these positions based on SIMP"}"""


CRITIC_AGENT_PROMPT = """You are the CRITIC AGENT in a multi-agent Strut-and-Tie Model design team.

Your role: Rigorously critique a proposed node layout from the Topology Agent.
You do NOT propose alternatives — you only identify problems.
Be specific and engineering-grounded. If the proposal is good, say so honestly.

## WHAT TO EVALUATE
1. **Diagonal angle quality**: do x-positions produce diagonals near 45°?
   - Struts steeper than 60° or shallower than 30° are concerning
2. **Force path coverage**: does each load have a clear diagonal route to a support?
3. **Panel balance**: are panels between intermediate nodes too wide or too narrow?
   - Very wide panels (> 1.5 × beam height) need intermediate support
   - Very narrow panels (< 0.3 × beam height) create steep struts
4. **Symmetry**: if loading is symmetric, does the layout respect it?
5. **SIMP alignment**: do the proposed positions actually follow SIMP's force paths?

## OUTPUT FORMAT
{
  "severity": "major" | "minor" | "none",
  "issues": ["specific problem 1", "specific problem 2"],
  "affected_positions": [x-positions that should be reconsidered],
  "recommendation": "what direction the Revision Agent should explore"
}

If severity is "none", the layout is acceptable as-is. Issues list can be empty."""


REVISION_AGENT_PROMPT = """You are the REVISION AGENT in a multi-agent Strut-and-Tie Model design team.

Your role: Produce a REVISED node layout that addresses the Critic's concerns.
You receive: the original proposal, SIMP data, and the Critic's specific issues.

## YOUR JOB
1. Read the Critic's issues carefully
2. Revise x-positions to fix those issues WITHOUT breaking what was already good
3. If the Critic says "shift left", don't shift too aggressively — be measured
4. Explain how each change addresses a specific Critic issue

## IMPORTANT
- You are NOT starting from scratch. Build on the original proposal.
- If the Critic's severity is "minor", small adjustments are enough
- If "major", substantive restructuring may be needed
- If "none", return the original unchanged

## OUTPUT FORMAT
{"n_intermediate": N, "x_positions": [...], "reasoning": "which Critic issue each change addresses"}"""


DECISION_AGENT_PROMPT = """You are the DECISION AGENT in a multi-agent Strut-and-Tie Model design team.

Your role: Choose between the original proposal (from Topology Agent) and the revised version (from Revision Agent).
You are the final arbiter — your choice goes forward to physical analysis.

## HOW TO DECIDE
1. Re-read the Critic's issues
2. Check if the Revision actually addressed those issues
3. Consider: does the revision introduce NEW problems while fixing old ones?
4. Trust the Critic's severity rating:
   - "major" → revision is usually better (unless it's clearly worse)
   - "minor" → either could work; pick the one with cleaner reasoning
   - "none" → original was already fine

## OUTPUT FORMAT
{
  "choice": "original" | "revised",
  "justification": "one-sentence reason based on Critic's issues"
}"""


ENGINEERING_REVIEWER_PROMPT = """You are an ENGINEERING REVIEWER for Strut-and-Tie Models (STM).

The STM has already passed all structural rule checks by code.
Your job: assess ENGINEERING QUALITY with specific criteria.

## CHECKLIST (score each 1-3, then average)
1. LOAD PATH: Does EVERY load point have at least one diagonal strut going toward a support?
   - 3: All loads have direct diagonal paths to supports
   - 2: Most loads have diagonal paths
   - 1: Some loads rely only on horizontal transfer (no diagonal)

2. FORCE FLOW: Do struts follow natural compression paths (roughly 45 from loads to supports)?
   - 3: Diagonals are 35-55 degrees (near optimal)
   - 2: Diagonals are 25-35 or 55-65 degrees (acceptable but not ideal)
   - 1: Very steep or very shallow diagonals dominate

3. SYMMETRY: If loading is symmetric, is the STM symmetric?
   - 3: Perfectly symmetric
   - 2: Minor asymmetry
   - 1: Clearly asymmetric

4. COMPLETENESS: Are there unsupported spans (long horizontal runs without diagonal bracing)?
   - 3: Every panel has diagonal bracing
   - 2: One panel without diagonal
   - 1: Multiple panels without diagonals

Respond with ONLY a valid JSON object:
{
  "load_path_score": 1-3,
  "force_flow_score": 1-3,
  "symmetry_score": 1-3,
  "completeness_score": 1-3,
  "total_score": average of above (1.0-3.0),
  "assessment": "ACCEPTABLE" or "QUESTIONABLE",
  "issues": ["list of specific problems found"],
  "suggestion": "specific improvement if QUESTIONABLE"
}"""




# ===========================================================
# Grammar-based STM Generation
# ===========================================================

GRAMMAR_RULES_PROMPT = """You are an STM DESIGN AGENT using grammar rules.
You see the current state of a Strut-and-Tie Model and choose which rule to apply next.

## AVAILABLE RULES
1. DIRECT_STRUT: Connect a load point directly to the nearest support with a diagonal strut.
   Use when: A load has no diagonal connection to a support yet.

2. SPLIT_SPAN: Add an intermediate node between a SUPPORT and a LOAD point.
   Use when: A support-to-load span has a diagonal angle below 25 degrees (too shallow).
   IMPORTANT: Only split spans between a support and a load. Never between two loads.

3. ADD_VERTICAL: Add a vertical tie between top and bottom nodes at the same x-position.
   Use when: A load node needs a vertical tie to transfer load downward.

4. DONE: The STM is complete. All loads have force paths to supports.
   Use when: Every load has at least one diagonal path to a support, and no span is excessively long.

## DECISION PROCESS (follow this ORDER strictly)
1. FIRST: Does every load node have a diagonal strut toward a support? If ANY load lacks one -> DIRECT_STRUT
2. SECOND: Are any diagonal strut angles below 25 degrees? If too shallow -> SPLIT_SPAN
3. THIRD: Do nodes at the same x need vertical ties? If so -> ADD_VERTICAL
4. ONLY if everything is connected -> DONE

IMPORTANT: Always handle DIRECT_STRUT before SPLIT_SPAN. Loads without diagonal paths are the highest priority.

## OUTPUT FORMAT
{"rule": "RULE_NAME", "target": "description of where to apply", "reasoning": "why this rule now"}"""

# ===========================================================
# 3. Ground Structure: Deterministic Connection Enumeration
# ===========================================================

def enumerate_ground_structure(nodes, beam_height, y_bot=None, y_top=None):
    """
    Deterministically generate all valid connections for an STM.

    Args:
        nodes: dict {nid: [x, y], ...}
        beam_height: total beam height in mm
        y_bot: bottom chord y-coordinate (if None, inferred from nodes)
        y_top: top chord y-coordinate (if None, inferred from nodes)

    Returns:
        fixed_connections: list of [n1, n2]
        diagonal_candidates: list of dicts
    """
    H = beam_height

    # Compute dy from actual chord positions
    if y_bot is not None and y_top is not None:
        dy = y_top - y_bot
    else:
        # Fallback: infer from node positions
        all_y = [y for _, (x, y) in nodes.items()]
        dy = max(all_y) - min(all_y)

    # Separate top and bottom nodes
    top_nodes = sorted(
        [(nid, x, y) for nid, (x, y) in nodes.items() if y > H / 2],
        key=lambda t: t[1]
    )
    bot_nodes = sorted(
        [(nid, x, y) for nid, (x, y) in nodes.items() if y <= H / 2],
        key=lambda t: t[1]
    )

    fixed = []

    # 1. Horizontal: adjacent top nodes
    for i in range(len(top_nodes) - 1):
        fixed.append([top_nodes[i][0], top_nodes[i + 1][0]])

    # 2. Horizontal: adjacent bottom nodes
    for i in range(len(bot_nodes) - 1):
        fixed.append([bot_nodes[i][0], bot_nodes[i + 1][0]])

    # 3. Vertical: same x-coordinate pairs
    for t_nid, t_x, t_y in top_nodes:
        for b_nid, b_x, b_y in bot_nodes:
            if abs(t_x - b_x) < 50:
                fixed.append([t_nid, b_nid])

    # 4. Diagonal candidates: panel-adjacent pairs with valid angles
    all_xs = sorted(set(round(x) for _, (x, y) in nodes.items()))

    def segments_cross(p1, p2, p3, p4):
        def cp(o, a, b):
            return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
        d1, d2 = cp(p3, p4, p1), cp(p3, p4, p2)
        d3, d4 = cp(p1, p2, p3), cp(p1, p2, p4)
        return ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
               ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0))

    candidates = []
    candidate_id = 0

    for i in range(len(all_xs) - 1):
        left_x, right_x = all_xs[i], all_xs[i + 1]

        panel_top = [n for n in top_nodes if abs(round(n[1]) - left_x) < 50 or abs(round(n[1]) - right_x) < 50]
        panel_bot = [n for n in bot_nodes if abs(round(n[1]) - left_x) < 50 or abs(round(n[1]) - right_x) < 50]

        for t_nid, t_x, t_y in panel_top:
            for b_nid, b_x, b_y in panel_bot:
                dx_m = abs(t_x - b_x)
                if dx_m < 50:
                    continue  # vertical, already in fixed

                # Check panel adjacency
                lo_x, hi_x = min(t_x, b_x), max(t_x, b_x)
                between = [nx for nx in all_xs if lo_x + 50 < nx < hi_x - 50]
                if between:
                    continue

                # Check angle (KDS: ≥25° lower bound only; no upper limit per code)
                angle = math.degrees(math.atan2(dy, dx_m))
                if angle < 25:
                    continue

                # Check crossing with fixed connections
                crosses_fixed = False
                for fn1, fn2 in fixed:
                    if fn1 == t_nid or fn1 == b_nid or fn2 == t_nid or fn2 == b_nid:
                        continue
                    if fn1 in nodes and fn2 in nodes:
                        if segments_cross(nodes[t_nid], nodes[b_nid], nodes[fn1], nodes[fn2]):
                            crosses_fixed = True
                            break
                if crosses_fixed:
                    continue

                length = math.sqrt(dx_m**2 + dy**2)
                candidates.append({
                    'id': candidate_id,
                    'nodes': [t_nid, b_nid],
                    'angle': angle,
                    'length': length,
                    'panel': (left_x, right_x),
                    'description': f"{t_nid}({t_x:.0f},{t_y:.0f})-{b_nid}({b_x:.0f},{b_y:.0f}) "
                                   f"angle={angle:.1f} deg, panel=[{left_x},{right_x}]"
                })
                candidate_id += 1

    # Mark crossing pairs among diagonal candidates
    for i in range(len(candidates)):
        candidates[i]['crosses_with'] = []
        for j in range(len(candidates)):
            if i == j:
                continue
            n1a, n1b = candidates[i]['nodes']
            n2a, n2b = candidates[j]['nodes']
            if n1a == n2a or n1a == n2b or n1b == n2a or n1b == n2b:
                continue
            if segments_cross(nodes[n1a], nodes[n1b], nodes[n2a], nodes[n2b]):
                candidates[i]['crosses_with'].append(j)

    return fixed, candidates


def build_stm_from_selection(nodes, supports, fixed_connections, diagonal_candidates, selected_indices):
    """Build STM data from fixed connections + selected diagonal indices."""
    connections = list(fixed_connections)
    for idx in selected_indices:
        if 0 <= idx < len(diagonal_candidates):
            connections.append(diagonal_candidates[idx]['nodes'])

    return {
        'nodes': nodes,
        'connections': connections,
        'supports': supports,
        'design_notes': f"Selected diagonals: {selected_indices}"
    }


# ===========================================================
# 3-B. Deterministic Structural Checks
# ===========================================================

def code_validate(stm_data, beam_length, beam_height, support_positions=None, loads=None):
    """Validate STM against structural rules."""
    errors = []
    warnings = []

    nodes = stm_data.get('nodes', {})
    connections = stm_data.get('connections', [])
    supports = stm_data.get('supports', {})

    if len(nodes) < 4:
        errors.append(f"Min 4 nodes required (got {len(nodes)})")
    if len(connections) < len(nodes):
        errors.append(f"Too few members: {len(connections)} for {len(nodes)} nodes (need >= {len(nodes)})")

    if nodes and connections:
        degree = {nid: 0 for nid in nodes}
        for n1, n2 in connections:
            if n1 in degree: degree[n1] += 1
            if n2 in degree: degree[n2] += 1
        for nid, d in degree.items():
            if d < 2:
                errors.append(f"Node {nid} has only {d} connection(s) (need >= 2)")

    for nid, (x, y) in nodes.items():
        if x < 0 or x > beam_length:
            errors.append(f"Node {nid}: x={x:.0f} outside beam (0~{beam_length:.0f})")
        if y < 0 or y > beam_height:
            errors.append(f"Node {nid}: y={y:.0f} outside beam (0~{beam_height:.0f})")

    if support_positions:
        for sup_x, sup_type in support_positions:
            found = False
            for nid, stype in supports.items():
                if stype == sup_type and nid in nodes:
                    nx, ny = nodes[nid]
                    if abs(nx - sup_x) > 50:
                        errors.append(f"Support {sup_type} node {nid}: x={nx:.0f} should be {sup_x:.0f}")
                    found = True
                    break
            if not found:
                errors.append(f"Missing {sup_type} support at x={sup_x:.0f}")

    # Vertical pair connectivity check
    if nodes and connections:
        top_nodes = {nid: (x, y) for nid, (x, y) in nodes.items() if y > beam_height / 2}
        bot_nodes = {nid: (x, y) for nid, (x, y) in nodes.items() if y <= beam_height / 2}
        conn_set = {frozenset([n1, n2]) for n1, n2 in connections}
        for t_nid, (t_x, t_y) in top_nodes.items():
            for b_nid, (b_x, b_y) in bot_nodes.items():
                if abs(t_x - b_x) < 50:
                    if frozenset([t_nid, b_nid]) not in conn_set:
                        errors.append(f"Vertical pair {t_nid}-{b_nid} must be connected (same x={t_x:.0f})")

    # Symmetry check — generalized for N loads
    if nodes and connections and loads and len(loads) >= 2:
        mid = beam_length / 2.0
        # Check if loads are symmetric about mid
        load_xs = sorted([lx for lx, _ in loads])
        is_symmetric = True
        for i in range(len(load_xs)):
            mirror_x = beam_length - load_xs[-(i+1)]
            if abs(load_xs[i] - mirror_x) > 100:
                is_symmetric = False
                break

        if is_symmetric:
            mirror_map = {}
            node_list = list(nodes.items())
            used = set()
            for nid1, (x1, y1) in node_list:
                if nid1 in used:
                    continue
                mirror_x = beam_length - x1
                for nid2, (x2, y2) in node_list:
                    if nid2 in used or nid2 == nid1:
                        continue
                    if abs(x2 - mirror_x) < 100 and abs(y1 - y2) < 100:
                        mirror_map[nid1] = nid2
                        mirror_map[nid2] = nid1
                        used.add(nid1)
                        used.add(nid2)
                        break
            conn_keys = {frozenset([n1, n2]) for n1, n2 in connections}
            for n1, n2 in connections:
                m1 = mirror_map.get(n1)
                m2 = mirror_map.get(n2)
                if m1 and m2:
                    if frozenset([m1, m2]) not in conn_keys:
                        warnings.append(f"Connection {n1}-{n2} has no mirror: expected {m1}-{m2}")

    # Diagonal angle check
    for n1, n2 in connections:
        if n1 not in nodes or n2 not in nodes:
            errors.append(f"Connection {n1}-{n2} references undefined node")
            continue
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        if dy < 50 or dx < 50:
            continue
        angle = math.degrees(math.atan2(dy, dx))
        if angle < 25:
            errors.append(f"Member {n1}-{n2}: angle={angle:.1f} deg below 25 deg minimum")

    # Skip-node check for diagonals
    if nodes and connections:
        all_xs = sorted(set(round(x) for _, (x, y) in nodes.items()))
        for n1, n2 in connections:
            if n1 not in nodes or n2 not in nodes:
                continue
            x1, y1 = nodes[n1]
            x2, y2 = nodes[n2]
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            if dy < 50 or dx < 50:
                continue
            lo_x, hi_x = min(x1, x2), max(x1, x2)
            between = [nx for nx in all_xs if lo_x + 50 < nx < hi_x - 50]
            if between:
                errors.append(f"Diagonal {n1}-{n2} skips intermediate x={between}")

    # Crossing check
    def segments_cross(p1, p2, p3, p4):
        def cp(o, a, b):
            return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
        d1, d2 = cp(p3, p4, p1), cp(p3, p4, p2)
        d3, d4 = cp(p1, p2, p3), cp(p1, p2, p4)
        return ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
               ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0))

    for i in range(len(connections)):
        n1, n2 = connections[i]
        if n1 not in nodes or n2 not in nodes: continue
        for j in range(i + 1, len(connections)):
            n3, n4 = connections[j]
            if n3 not in nodes or n4 not in nodes: continue
            if n1 == n3 or n1 == n4 or n2 == n3 or n2 == n4: continue
            if segments_cross(nodes[n1], nodes[n2], nodes[n3], nodes[n4]):
                errors.append(f"Members {n1}-{n2} and {n3}-{n4} cross")

    # Connectivity check
    if nodes:
        adj = {nid: set() for nid in nodes}
        for n1, n2 in connections:
            if n1 in adj and n2 in adj:
                adj[n1].add(n2)
                adj[n2].add(n1)
        start = list(nodes.keys())[0]
        visited = {start}
        queue = [start]
        while queue:
            cur = queue.pop(0)
            for nb in adj[cur]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        disconnected = set(nodes.keys()) - visited
        if disconnected:
            errors.append(f"Disconnected nodes: {disconnected}")

    has_pin = any(t == 'pin' for t in supports.values())
    has_roller = any(t == 'roller' for t in supports.values())
    if not has_pin: errors.append("Missing pin support")
    if not has_roller: errors.append("Missing roller support")

    m = len(connections)
    n = len(nodes)
    r = sum(2 if t == 'pin' else 1 for t in supports.values())
    if m + r < 2 * n:
        warnings.append(f"Underdetermined: m({m})+r({r})={m+r} < 2n={2*n}")

    return {'valid': len(errors) == 0, 'errors': errors, 'warnings': warnings}


# ===========================================================
# 3-C. Truss Analysis (Force Equilibrium Method)
# ===========================================================

def solve_truss(nodes, connections, supports, loads_dict):
    """
    Solve 2D truss using Force Equilibrium Method (Method of Joints).
    
    Unlike Direct Stiffness Method, this works for STM-type trusses
    where m+r < 2n (mechanisms). DSM fails for mechanisms because the
    stiffness matrix is singular. The force method directly solves for
    member forces using equilibrium equations.
    
    Approach:
      1. Compute reactions exactly from statics (SFx, SFy, SM)
      2. Set up equilibrium equations at each node: A * x = b
      3. Solve with least squares (handles over-determined systems)
    """
    node_ids = sorted(nodes.keys())
    n_nodes = len(node_ids)
    id_map = {nid: i for i, nid in enumerate(node_ids)}

    # Identify supports
    pin_nodes = [nid for nid, st in supports.items() if st == 'pin' and nid in nodes]
    roller_nodes = [nid for nid, st in supports.items() if st == 'roller' and nid in nodes]

    if not pin_nodes or not roller_nodes:
        return {'success': False, 'error': 'Missing pin or roller support',
                'member_forces': {}, 'reactions': {}}

    # Step 1: Exact reactions from statics
    pin_nid = pin_nodes[0]
    roller_nid = roller_nodes[0]
    px, py = nodes[pin_nid]
    rx_s, ry_s = nodes[roller_nid]

    total_fx = sum(fx for fx, fy in loads_dict.values())
    total_fy = sum(fy for fx, fy in loads_dict.values())

    # Moment about pin = 0 -> Ry_roller
    moment = 0.0
    for nid, (fx, fy) in loads_dict.items():
        if nid in nodes:
            nx, ny = nodes[nid]
            moment += fx * (ny - py) - fy * (nx - px)

    lever = rx_s - px
    if abs(lever) < 1.0:
        return {'success': False, 'error': 'Supports at same x-position',
                'member_forces': {}, 'reactions': {}}

    ry_roller = moment / lever
    ry_pin = -total_fy - ry_roller
    rx_pin = -total_fx

    reactions = {
        pin_nid: (rx_pin, ry_pin),
        roller_nid: (0.0, ry_roller)
    }

    # Step 2: Build equilibrium equations for member forces
    valid_connections = []
    for n1, n2 in connections:
        if n1 in nodes and n2 in nodes:
            x1, y1 = nodes[n1]; x2, y2 = nodes[n2]
            L = math.sqrt((x2-x1)**2 + (y2-y1)**2)
            if L >= 1.0:
                valid_connections.append((n1, n2))

    n_members = len(valid_connections)
    n_eq = 2 * n_nodes

    A = np.zeros((n_eq, n_members))
    b = np.zeros(n_eq)

    for mi, (n1, n2) in enumerate(valid_connections):
        x1, y1 = nodes[n1]; x2, y2 = nodes[n2]
        L = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        cx, cy = (x2-x1)/L, (y2-y1)/L
        i1, i2 = id_map[n1], id_map[n2]
        # Positive force = tension (pulls n1 toward n2)
        A[2*i1, mi] += cx;   A[2*i1+1, mi] += cy
        A[2*i2, mi] -= cx;   A[2*i2+1, mi] -= cy

    # RHS = -(external loads + reactions)
    for nid, (fx, fy) in loads_dict.items():
        if nid in id_map:
            idx = id_map[nid]
            b[2*idx] -= fx; b[2*idx+1] -= fy
    for nid, (r_x, r_y) in reactions.items():
        if nid in id_map:
            idx = id_map[nid]
            b[2*idx] -= r_x; b[2*idx+1] -= r_y

    # Step 3: Solve with least squares
    try:
        x_sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError as e:
        return {'success': False, 'error': f'Solver failed: {e}',
                'member_forces': {}, 'reactions': {}}

    # Check residual
    residual_vec = A @ x_sol - b
    residual_norm = float(np.linalg.norm(residual_vec))
    total_load = abs(total_fy) if abs(total_fy) > 0 else 1.0
    residual_pct = residual_norm / total_load * 100

    member_forces = {}
    for mi, (n1, n2) in enumerate(valid_connections):
        member_forces[(n1, n2)] = float(x_sol[mi])

    return {
        'success': True, 'error': None,
        'member_forces': member_forces,
        'reactions': reactions,
        'equilibrium_error_pct': round(residual_pct, 1)
    }



def check_physical_validity(truss_result, stm_data, loads):
    """
    Check physical validity of truss analysis results.

    Returns:
        (valid: bool, reason: str)
    """
    if not truss_result.get('success', False):
        return False, 'analysis failed'

    # 1. Reaction equilibrium: ΣRy ≈ ΣP
    total_load = sum(abs(fy) * 1000 for _, fy in loads)
    if total_load > 0 and truss_result.get('reactions'):
        total_ry = sum(abs(ry) for _, (rx, ry) in truss_result['reactions'].items())
        if abs(total_ry - total_load) / total_load > 0.05:
            return False, f'equilibrium violated: ΣRy={total_ry:.0f} vs ΣP={total_load:.0f}'

    # 2. All forces zero → load not transferred
    if truss_result.get('member_forces'):
        all_zero = all(abs(f) < 1.0 for f in truss_result['member_forces'].values())
        if all_zero:
            return False, 'all member forces zero — no load transfer'

    # 3. Roller should not carry significant horizontal force
    supports = stm_data.get('supports', {})
    if truss_result.get('reactions'):
        for nid, stype in supports.items():
            if stype == 'roller' and nid in truss_result['reactions']:
                rx, ry = truss_result['reactions'][nid]
                if total_load > 0 and abs(rx) > 0.01 * total_load:
                    return False, f'roller {nid} has Rx={rx:.0f}N'

    return True, 'ok'


def score_stm(stm_data, beam_length, beam_height, loads, support_positions):
    """
    Score STM using truss analysis. Lower = better.

    Normalization:
        Score = Sum(Ti * Li) / (P_ref * D_ref)
        where P_ref = max applied load (N), D_ref = beam diagonal (mm)
        This makes scores dimensionless and comparable across beam sizes.
        The /1000 converts mm to m in the denominator for convenient scale.
    """
    nodes = stm_data['nodes']
    connections = stm_data['connections']
    supports = stm_data['supports']

    loads_dict = {}
    for load_x, load_fy_kN in loads:
        for nid, (nx, ny) in nodes.items():
            if abs(nx - load_x) < 50 and ny > beam_height / 2:
                loads_dict[nid] = [0.0, float(load_fy_kN) * 1000.0]
                break

    if loads_dict:
        result = solve_truss(nodes, connections, supports, loads_dict)
        if result['success']:
            # Physical validity check
            valid, reason = check_physical_validity(result, stm_data, loads)
            if not valid:
                return 1e6  # Invalid STM — never selected as best

            # Equilibrium error check: reject if > 20%
            eq_err = result.get('equilibrium_error_pct', 0)
            if eq_err > 20:
                return 1e6  # Too much equilibrium error — unreliable

            total_TL = 0.0
            zero_force_count = 0
            for (n1, n2), force in result['member_forces'].items():
                if force > 0:
                    x1, y1 = nodes[n1]
                    x2, y2 = nodes[n2]
                    L = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                    total_TL += force * L
                elif abs(force) < 1.0:  # < 1N = effectively zero
                    zero_force_count += 1
            ref_load = max(abs(fy) * 1000 for _, fy in loads)
            ref_length = math.sqrt(beam_length**2 + beam_height**2)
            # /1000 converts N*mm to N*m for human-readable scale
            base_score = total_TL / (ref_load * ref_length * 1000) if ref_load > 0 else total_TL
            # Penalty: each zero-force member adds 50% to score
            return base_score * (1 + zero_force_count * 0.5)

    # Analysis failed (singular matrix, no loads, etc.) — never selected as best
    return 1e6


def score_stm_detailed(stm_data, beam_length, beam_height, loads, support_positions):
    """Full truss analysis with detailed results."""
    nodes = stm_data['nodes']
    connections = stm_data['connections']
    supports = stm_data['supports']

    loads_dict = {}
    for load_x, load_fy_kN in loads:
        for nid, (nx, ny) in nodes.items():
            if abs(nx - load_x) < 50 and ny > beam_height / 2:
                loads_dict[nid] = [0.0, float(load_fy_kN) * 1000.0]
                break

    if not loads_dict:
        return {'score': float('inf'), 'analysis_success': False, 'error': 'No load nodes'}

    result = solve_truss(nodes, connections, supports, loads_dict)
    if not result['success']:
        return {
            'score': 1e6,
            'analysis_success': False, 'error': result['error'],
            'member_forces': {}, 'reactions': {}
        }

    # Equilibrium error check
    eq_err = result.get('equilibrium_error_pct', 0)
    if eq_err > 20:
        return {
            'score': 1e6,
            'analysis_success': False,
            'error': f'Equilibrium error too high: {eq_err:.1f}%',
            'member_forces': {}, 'reactions': result['reactions'],
            'equilibrium_error_pct': eq_err
        }

    total_TL = 0.0
    member_details = {}
    for (n1, n2), force in result['member_forces'].items():
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]
        L = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        mtype = 'tie' if force > 0 else 'strut'
        TL = force * L if force > 0 else 0
        total_TL += TL
        member_details[(n1, n2)] = {
            'force_N': force, 'force_kN': force / 1000,
            'length_mm': L, 'type': mtype, 'TL': TL
        }

    ref_load = max(abs(fy) * 1000 for _, fy in loads)
    ref_length = math.sqrt(beam_length**2 + beam_height**2)
    normalized = total_TL / (ref_load * ref_length * 1000) if ref_load > 0 else total_TL

    # Zero-force penalty
    zero_force_count = sum(1 for v in member_details.values() if abs(v['force_N']) < 1.0)
    penalized_score = normalized * (1 + zero_force_count * 0.5)

    return {
        'score': penalized_score, 'raw_TL': total_TL,
        'zero_force_count': zero_force_count,
        'equilibrium_error_pct': result.get('equilibrium_error_pct', 0),
        'member_forces': member_details,
        'reactions': result['reactions'],
        'analysis_success': True, 'error': None
    }


def print_truss_analysis(detail):
    """Pretty-print truss analysis results."""
    if not detail.get('analysis_success'):
        print(f"  [Truss Analysis] FAILED: {detail.get('error', 'unknown')}")
        print(f"  [Score] 1e6 (penalty — excluded from best selection)")
        return

    print(f"  [Truss Analysis] SUCCESS")
    print(f"  Score (norm. Sigma Ti*Li): {detail['score']:.4f}")
    print(f"  Raw Sigma(Ti*Li): {detail['raw_TL']/1e6:.1f} kN*m")
    eq_err = detail.get('equilibrium_error_pct', 0)
    if eq_err > 0.1:
        print(f"  Equilibrium error: {eq_err:.1f}%")
    zfc = detail.get('zero_force_count', 0)
    if zfc > 0:
        print(f"  Zero-force members: {zfc} (penalty: score x{1 + zfc * 0.5:.1f})")

    print(f"\n  Reactions:")
    for nid, (rx, ry) in detail['reactions'].items():
        print(f"    {nid}: Rx={rx/1000:.1f} kN, Ry={ry/1000:.1f} kN")

    ties = [(n1, n2, info) for (n1, n2), info in detail['member_forces'].items() if info['type'] == 'tie']
    struts = [(n1, n2, info) for (n1, n2), info in detail['member_forces'].items() if info['type'] == 'strut']

    print(f"\n  Ties ({len(ties)}):")
    for n1, n2, info in sorted(ties, key=lambda x: -abs(x[2]['force_N'])):
        print(f"    {n1}-{n2}: {info['force_kN']:.1f} kN, L={info['length_mm']:.0f}mm, "
              f"T*L={info['TL']/1e6:.2f} kN*m")

    print(f"\n  Struts ({len(struts)}):")
    for n1, n2, info in sorted(struts, key=lambda x: -abs(x[2]['force_N'])):
        print(f"    {n1}-{n2}: {info['force_kN']:.1f} kN, L={info['length_mm']:.0f}mm")


# ===========================================================
# 4. JSON Parser
# ===========================================================

def parse_json_from_text(text):
    """Extract JSON from LLM response."""
    if text is None:
        return None

    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)
    text = text.strip()
    if not text:
        return None

    def fix_arithmetic(match):
        try:
            return str(eval(match.group(0)))
        except:
            return match.group(0)

    def preprocess(s):
        return re.sub(r'\d+\s*[\+\-\*/]\s*\d+', fix_arithmetic, s)

    def repair_truncated_json(s):
        s = s.rstrip().rstrip(',')
        open_braces = s.count('{') - s.count('}')
        open_brackets = s.count('[') - s.count(']')
        if 0 < open_braces <= 3 and open_brackets >= 0:
            s += ']' * open_brackets + '}' * open_braces
            return s
        if open_braces == 0 and 0 < open_brackets <= 3:
            s += ']' * open_brackets
            return s
        return None

    def try_parse(s):
        for variant in [s, preprocess(s)]:
            try:
                return json.loads(variant)
            except:
                pass
        return None

    result = try_parse(text)
    if result: return result

    try:
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            result = try_parse(match.group(1))
            if result: return result
    except: pass

    try:
        start = text.index('{')
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{': depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    result = try_parse(text[start:i+1])
                    if result: return result
    except: pass

    try:
        start = text.index('{')
        repaired = repair_truncated_json(preprocess(text[start:]))
        if repaired:
            result = json.loads(repaired)
            if result: return result
    except: pass

    return None


# ===========================================================
# 5. JSON Build Helper
# ===========================================================

def build_member_json(stm_data, detail=None):
    """
    Build member info list for JSON output.
    Extracted to avoid code duplication between best candidate and all candidates.

    Args:
        stm_data: dict with 'nodes', 'connections'
        detail: score_stm_detailed() result (or None)

    Returns:
        list of member dicts
    """
    members_json = []
    for n1, n2 in stm_data['connections']:
        if n1 not in stm_data['nodes'] or n2 not in stm_data['nodes']:
            continue
        x1, y1 = stm_data['nodes'][n1]
        x2, y2 = stm_data['nodes'][n2]
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        length = math.sqrt(dx**2 + dy**2)
        if dy < 50:
            mtype = "horizontal"
        elif dx < 50:
            mtype = "vertical"
        else:
            mtype = "diagonal"
        angle = math.degrees(math.atan2(dy, dx)) if (dy >= 50 and dx >= 50) else None

        force_kN = None
        if detail and detail.get('analysis_success'):
            for (fn1, fn2), info in detail.get('member_forces', {}).items():
                if (fn1 == n1 and fn2 == n2) or (fn1 == n2 and fn2 == n1):
                    force_kN = round(info['force_kN'], 1)
                    break

        member = {
            'nodes': [n1, n2],
            'type': mtype,
            'length_mm': round(length, 1),
        }
        if angle:
            member['angle_deg'] = round(angle, 1)
        if force_kN is not None:
            member['force_kN'] = force_kN

        members_json.append(member)

    return members_json


# ===========================================================
# 6. Visualization Functions
# ===========================================================

def _draw_beam(ax, L, H, loads=None, support_positions=None):
    """Draw beam outline, loads, supports."""
    beam_rect = mpatches.FancyBboxPatch(
        (0, 0), L, H, linewidth=1.5, edgecolor='#94a3b8',
        facecolor='#f1f5f9', alpha=0.3, boxstyle="square,pad=0"
    )
    ax.add_patch(beam_rect)
    if loads:
        for lx, lfy in loads:
            ax.annotate('', xy=(lx, H - 50), xytext=(lx, H + 200),
                        arrowprops=dict(arrowstyle='->', color='#dc2626', lw=2.5))
            ax.text(lx, H + 250, f'{abs(lfy):.0f} kN', ha='center',
                    fontsize=10, fontweight='bold', color='#dc2626')
    if support_positions:
        for sx, stype in support_positions:
            if stype == 'pin':
                tri = plt.Polygon([(sx-80, -50), (sx+80, -50), (sx, 0)],
                    closed=True, facecolor='#fef3c7', edgecolor='#d97706', linewidth=1.5)
                ax.add_patch(tri)
            else:
                circ = plt.Circle((sx, -50), 30, facecolor='#fef3c7',
                                  edgecolor='#d97706', linewidth=1.5)
                ax.add_patch(circ)
                ax.plot([sx-60, sx+60], [-80, -80], color='#d97706', linewidth=1.5)
    margin = 400
    ax.set_xlim(-margin, L + margin)
    ax.set_ylim(-200, H + 400)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15)
    ax.set_xlabel('X (mm)', fontsize=10)
    ax.set_ylabel('Y (mm)', fontsize=10)


def plot_nodes_only(nodes_data, beam_length, beam_height,
                    loads=None, support_positions=None,
                    save_path=None, title=None):
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    _draw_beam(ax, beam_length, beam_height, loads, support_positions)
    nodes = nodes_data.get('nodes', {})
    supports = nodes_data.get('supports', {})
    for nid, (x, y) in nodes.items():
        if nid in supports:
            marker = '^' if supports[nid] == 'pin' else 's'
            ax.plot(x, y, marker=marker, markersize=14, color='#f59e0b',
                    markeredgecolor='#d97706', markeredgewidth=2, zorder=4)
        else:
            ax.plot(x, y, 'o', markersize=12, color='#3b82f6',
                    markeredgecolor='#1d4ed8', markeredgewidth=2, zorder=4)
        offset_y = 80 if y > beam_height / 2 else -100
        ax.annotate(f'{nid}\n({x:.0f},{y:.0f})', (x, y + offset_y),
            fontsize=9, fontweight='bold', ha='center', va='center', color='#1e293b',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#dbeafe', edgecolor='#3b82f6', alpha=0.9))
    ax.set_title(title or 'Node Placement', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"  >> Saved: {save_path}")
    plt.close()


def plot_stm(stm_data, beam_length, beam_height,
             loads=None, support_positions=None,
             save_path=None, title=None, errors=None, status=None):
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    L, H = beam_length, beam_height
    _draw_beam(ax, L, H, loads, support_positions)
    nodes = stm_data.get('nodes', {})
    connections = stm_data.get('connections', [])
    supports = stm_data.get('supports', {})

    for n1_id, n2_id in connections:
        if n1_id not in nodes or n2_id not in nodes: continue
        x1, y1 = nodes[n1_id]
        x2, y2 = nodes[n2_id]
        dx, dy = abs(x2-x1), abs(y2-y1)
        member_has_error = errors and any(f"{n1_id}-{n2_id}" in e or f"{n2_id}-{n1_id}" in e for e in errors)
        if member_has_error: color, lw, ls = '#ef4444', 3.0, '--'
        elif dy < 50: color, lw, ls = '#2563eb', 2.0, '-'
        elif dx < 50: color, lw, ls = '#16a34a', 2.0, '-'
        else: color, lw, ls = '#dc2626', 2.5, '-'
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, linestyle=ls, zorder=2)
        mx, my = (x1+x2)/2, (y1+y2)/2
        if dy >= 50 and dx >= 50:
            angle = math.degrees(math.atan2(dy, dx))
            label = f'{n1_id}-{n2_id}\n{angle:.1f}'
        else:
            label = f'{n1_id}-{n2_id}'
        ax.annotate(label, (mx, my), fontsize=7, ha='center', va='bottom',
                    color='#475569', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.85))

    for nid, (x, y) in nodes.items():
        node_has_error = errors and any(f"Node {nid}" in e for e in errors)
        if node_has_error:
            ax.plot(x, y, 'X', markersize=14, color='#ef4444', markeredgecolor='#991b1b', markeredgewidth=2, zorder=5)
        elif nid in supports:
            marker = '^' if supports[nid] == 'pin' else 'o'
            ax.plot(x, y, marker=marker, markersize=14, color='#f59e0b',
                    markeredgecolor='#d97706', markeredgewidth=2, zorder=4)
        else:
            ax.plot(x, y, 'o', markersize=10, color='#1e293b',
                    markeredgecolor='#475569', markeredgewidth=1.5, zorder=4)
        offset_y = 80 if y > H/2 else -100
        ax.annotate(nid, (x, y + offset_y), fontsize=11, fontweight='bold',
                    ha='center', va='center', color='#1e293b',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#fef3c7', edgecolor='#f59e0b', alpha=0.9))

    if status:
        badge_color = '#16a34a' if status == 'PASS' else '#ef4444'
        ax.text(0.02, 0.95, status, transform=ax.transAxes, fontsize=16, fontweight='bold',
                color='white', bbox=dict(boxstyle='round,pad=0.4', facecolor=badge_color, alpha=0.9),
                verticalalignment='top', zorder=10)
    if errors:
        error_text = "ERRORS:\n" + "\n".join(f"- {e}" for e in errors[:5])
        ax.text(0.98, 0.95, error_text, transform=ax.transAxes, fontsize=8, color='#991b1b',
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#fef2f2', edgecolor='#fca5a5', alpha=0.95), zorder=10)

    ax.set_title(title or 'STM', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"  >> Saved: {save_path}")
    plt.close()


def plot_candidates_comparison(candidates_info, beam_length, beam_height,
                               loads=None, support_positions=None, save_path=None):
    n = len(candidates_info)
    if n == 0: return
    fig, axes = plt.subplots(1, n, figsize=(7*n, 6))
    if n == 1: axes = [axes]
    for idx, (stm, sc, cid) in enumerate(candidates_info):
        ax = axes[idx]
        _draw_beam(ax, beam_length, beam_height, loads, support_positions)
        nodes, connections, supports = stm['nodes'], stm['connections'], stm['supports']
        for n1_id, n2_id in connections:
            if n1_id not in nodes or n2_id not in nodes: continue
            x1, y1 = nodes[n1_id]; x2, y2 = nodes[n2_id]
            dx, dy = abs(x2-x1), abs(y2-y1)
            color = '#2563eb' if dy < 50 else ('#16a34a' if dx < 50 else '#dc2626')
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=2, zorder=2)
        for nid, (x, y) in nodes.items():
            if nid in supports:
                marker = '^' if supports[nid] == 'pin' else 'o'
                ax.plot(x, y, marker=marker, markersize=12, color='#f59e0b',
                        markeredgecolor='#d97706', markeredgewidth=2, zorder=4)
            else:
                ax.plot(x, y, 'o', markersize=8, color='#1e293b', zorder=4)
            offset_y = 60 if y > beam_height / 2 else -80
            ax.annotate(nid, (x, y + offset_y), fontsize=9, fontweight='bold', ha='center')
        is_best = (idx == 0)
        for spine in ax.spines.values():
            spine.set_edgecolor('#16a34a' if is_best else '#94a3b8')
            spine.set_linewidth(3 if is_best else 1)
        badge = "BEST" if is_best else ""
        ax.set_title(f'C{cid} score={sc:.4f} {badge}\n{len(nodes)}n, {len(connections)}m',
                     fontsize=11, fontweight='bold', pad=10)
    fig.suptitle('Candidate Comparison (lower score = better)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"  >> Saved: {save_path}")
    plt.close()


def plot_optimizer_comparison(before_stm, after_stm, before_score, after_score,
                              beam_length, beam_height, loads=None, support_positions=None, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    L, H = beam_length, beam_height
    for ax, stm, label, sc in [(ax1, before_stm, 'Before', before_score),
                                 (ax2, after_stm, 'After', after_score)]:
        _draw_beam(ax, L, H, loads, support_positions)
        nodes, connections, supports = stm['nodes'], stm['connections'], stm['supports']
        for n1_id, n2_id in connections:
            if n1_id not in nodes or n2_id not in nodes: continue
            x1, y1 = nodes[n1_id]; x2, y2 = nodes[n2_id]
            dx, dy = abs(x2-x1), abs(y2-y1)
            color = '#2563eb' if dy < 50 else ('#16a34a' if dx < 50 else '#dc2626')
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=2, zorder=2)
            mx, my = (x1+x2)/2, (y1+y2)/2
            if dy >= 50 and dx >= 50:
                angle = math.degrees(math.atan2(dy, dx))
                ax.annotate(f'{angle:.1f}', (mx, my), fontsize=8, ha='center', color='#475569',
                           fontweight='bold', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85))
        for nid, (x, y) in nodes.items():
            if nid in supports:
                marker = '^' if supports[nid] == 'pin' else 'o'
                ax.plot(x, y, marker=marker, markersize=12, color='#f59e0b',
                        markeredgecolor='#d97706', markeredgewidth=2, zorder=4)
            else:
                ax.plot(x, y, 'o', markersize=8, color='#1e293b', zorder=4)
            offset_y = 60 if y > H/2 else -80
            ax.annotate(nid, (x, y + offset_y), fontsize=10, fontweight='bold', ha='center')
        ax.set_title(f'{label}\nscore={sc:.4f} | {len(connections)} members',
                     fontsize=11, fontweight='bold', pad=10)
    title = f'Optimizer: score {before_score:.4f} -> {after_score:.4f}' if after_score < before_score \
        else 'Optimizer: No improvement (original kept)'
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"  >> Saved: {save_path}")
    plt.close()


# ===========================================================
# 7. MAS Controller (Versatile Version)
# ===========================================================

class MASSTMGenerator:
    """
    3-Agent MAS-based STM generator with Ground Structure approach.
    Pipeline: Phase 0 (SIMP)
           -> Phase 1 (Nodes + Ground Structure)
           -> Phase 2 (Engineering Review)
           -> Phase 3 (Truss Analysis + Scoring)
    """

    def __init__(self,
                 topology_model="qwen3:32b",
                 reviewer_model="qwen3:32b",
                 critic_model=None,
                 revision_model=None,
                 decision_model=None,
                 reasoning_model="nemotron-3-nano",
                 max_retries=5,
                 n_candidates=3,
                 ollama_url="http://localhost:11434",
                 output_dir="plots"):
        self.client = OllamaClient(ollama_url)
        # Backward compatibility: topology_model/reviewer_model
        self.topology_model = topology_model
        self.reviewer_model = reviewer_model
        # Cooperative agent models (default: reuse topology/reviewer)
        self.critic_model = critic_model or reviewer_model
        self.revision_model = revision_model or topology_model
        self.decision_model = decision_model or reviewer_model
        # Reasoning-specialized model for calculation
        self.reasoning_model = reasoning_model
        self.max_retries = max_retries
        self.n_candidates = n_candidates
        self.output_dir = output_dir
        self.log = []
        os.makedirs(output_dir, exist_ok=True)

    def _save_path(self, filename):
        return os.path.join(self.output_dir, filename)

    @staticmethod
    def compute_x1_range(sup_x, load_x, beam_length, dy):
        dx_min = dy / math.tan(math.radians(65))
        x1_lo = max(sup_x + dx_min, sup_x + 50)
        x1_hi = min(load_x - dx_min, load_x - 50)
        return x1_lo, x1_hi

    def _distribute_x1_values(self, sup_x, load_x, beam_length, dy, n):
        x1_lo, x1_hi = self.compute_x1_range(sup_x, load_x, beam_length, dy)
        if x1_hi <= x1_lo:
            return [int((sup_x + load_x) / 2)] * n
        margin = (x1_hi - x1_lo) * 0.05
        x1_lo_safe = x1_lo + margin
        x1_hi_safe = x1_hi - margin
        if n == 1:
            return [int((x1_lo_safe + x1_hi_safe) / 2)]
        return [int(x1_lo_safe + i / (n - 1) * (x1_hi_safe - x1_lo_safe)) for i in range(n)]

    # ── Stage 1-A: LLM decides node layout ──
    def _decide_node_layout(self, L, H, loads, supports, simp_summary="",
                            forced_n_intermediate=None, prev_layouts=None, feedback="",
                            y_bot=None, y_top=None, temperature=0.4,
                            use_mirror=True, design_hint=""):
        """
        LLM decides node layout.

        Args:
            use_mirror: if True, LLM decides left-half only (x ∈ [x1_lo, x1_hi]), code mirrors.
                        if False, LLM decides entire beam (x ∈ [sup_left+margin, sup_right-margin]).
        """
        # Compute dy from chord positions
        if y_bot is not None and y_top is not None:
            dy = y_top - y_bot
        else:
            dy = H - 275  # fallback for backward compatibility

        sorted_supports = sorted(supports, key=lambda s: s[0])
        sorted_loads = sorted(loads, key=lambda l: l[0])

        # Determine x-range and max_intermediate based on mirror mode
        if use_mirror:
            # Left-half only range
            sup_x = sorted_supports[0][0]
            load_x = sorted_loads[0][0]
            x_lo, x_hi = self.compute_x1_range(sup_x, load_x, L, dy)
            # Max intermediate from beam geometry (half-beam perspective)
            max_intermediate = max(1, int(round(L / H)) - 1)
            range_desc = "left-half only (right-half auto-mirrors)"
        else:
            # Whole-beam range — LLM decides all x-positions
            sup_left = sorted_supports[0][0]
            sup_right = sorted_supports[-1][0]
            # Minimum distance from support (to keep valid diagonal angles)
            dx_min = dy / math.tan(math.radians(65))
            x_lo = sup_left + max(dx_min, 50)
            x_hi = sup_right - max(dx_min, 50)
            # Max intermediate: about one per (1*H) span, doubled since covering whole beam
            max_intermediate = max(1, int(round(L / H)) * 2 - 2)
            range_desc = "entire beam (no auto-mirroring)"

        # 0 nodes: no LLM needed
        if forced_n_intermediate == 0:
            return {
                'n_intermediate': 0,
                'x_positions': [],
                'reasoning': 'Direct strut model (0 intermediate nodes)'
            }

        # Build user prompt — general for N loads / N supports
        user_prompt = f"Deep beam STM node layout decision:\n\n"
        user_prompt += f"Beam: {L}mm x {H}mm (L/H={L/H:.2f})\n"

        for i, (sx, stype) in enumerate(sorted_supports):
            user_prompt += f"Support {i+1}: x={sx:.0f}mm ({stype})\n"

        for i, (lx, lfy) in enumerate(sorted_loads):
            user_prompt += f"Load {i+1}: x={lx:.0f}mm ({abs(lfy):.0f}kN)\n"

        user_prompt += (
            f"Beam center: x={L/2:.0f}mm\n"
            f"Vertical distance between chords: dy={dy:.0f}mm\n\n"
            f"DECISION SCOPE: {range_desc}\n"
            f"VALID x-range for intermediate nodes: [{x_lo:.0f}, {x_hi:.0f}]\n\n"
        )

        if simp_summary:
            user_prompt += f"{simp_summary}\n\n"

        if forced_n_intermediate:
            user_prompt += (
                f"You MUST use {forced_n_intermediate} intermediate node(s).\n"
                f"Decide the x-position(s) for the {forced_n_intermediate} node(s).\n"
                f"Use SIMP force paths to choose optimal positions.\n\n"
            )
        else:
            user_prompt += (
                f"FIXED nodes: supports + loads (auto-generated)\n"
                f"You decide: how many intermediate positions (0 to {max_intermediate}) and their x-positions.\n"
                f"  0 nodes -> simplest (direct strut)\n"
            )
            for p in range(1, max_intermediate + 1):
                user_prompt += f"  {p} node(s) -> {p} intermediate vertical node(s)\n"
            user_prompt += "\n"

        if not use_mirror:
            # Build span descriptions for asymmetric mode
            all_fixed_xs = sorted(
                [sx for sx, _ in sorted_supports] + [lx for lx, _ in sorted_loads]
            )
            spans_desc = ""
            for i in range(len(all_fixed_xs) - 1):
                left_x = all_fixed_xs[i]
                right_x = all_fixed_xs[i + 1]
                span_len = right_x - left_x
                mid_x = int((left_x + right_x) / 2)
                spans_desc += f"  Span {i+1}: x={left_x:.0f} to x={right_x:.0f} ({span_len:.0f}mm, midpoint={mid_x})\n"

            user_prompt += (
                "IMPORTANT: You are deciding the ENTIRE BEAM's node layout.\n"
                "Do NOT assume symmetry — place nodes where forces actually flow.\n"
                "Place intermediate nodes BETWEEN supports and loads, not near them.\n\n"
                f"AVAILABLE SPANS:\n{spans_desc}\n"
                "Each intermediate node should be placed inside one of these spans.\n"
                "Longer spans with diagonal force paths benefit more from intermediate nodes.\n\n"
            )

        if prev_layouts:
            user_prompt += "PREVIOUS CANDIDATES (do NOT repeat these):\n"
            for i, pl in enumerate(prev_layouts):
                user_prompt += f"  C{i+1}: n_intermediate={pl.get('n_intermediate')}, x={pl.get('x_positions')}\n"
            user_prompt += "Choose a DIFFERENT n_intermediate or significantly different x-positions.\n\n"

        if feedback:
            user_prompt += f"FEEDBACK: {feedback}\n\n"

        if design_hint:
            user_prompt += f"DESIGN PHILOSOPHY: {design_hint}\n\n"

        user_prompt += 'Respond with ONLY a JSON: {"n_intermediate": N, "x_positions": [...], "reasoning": "..."}'

        response = self.client.chat(
            model=self.topology_model, system=NODE_LAYOUT_PROMPT,
            user=user_prompt, temperature=temperature
        )

        result = parse_json_from_text(response)
        if result and 'x_positions' in result:
            target_n = forced_n_intermediate if forced_n_intermediate else int(result.get('n_intermediate', 1))
            target_n = max(0, min(max_intermediate, target_n))
            x_positions = result.get('x_positions', [])

            validated = []
            for x in x_positions[:target_n]:
                x = int(x)
                x = max(int(x_lo), min(int(x_hi), x))
                validated.append(x)

            # Fill missing positions with evenly spaced values
            if len(validated) < target_n:
                full_set = [
                    int(x_lo + (x_hi - x_lo) * (i + 1) / (target_n + 1))
                    for i in range(target_n)
                ]
                for v in full_set:
                    if len(validated) >= target_n:
                        break
                    if v not in validated:
                        validated.append(v)

            validated = sorted(set(validated))[:target_n]

            return {
                'n_intermediate': len(validated),
                'x_positions': validated,
                'reasoning': result.get('reasoning', '')
            }

        # Fallback
        if forced_n_intermediate and forced_n_intermediate > 0:
            fallback_x = [
                int(x_lo + (x_hi - x_lo) * (i + 1) / (forced_n_intermediate + 1))
                for i in range(forced_n_intermediate)
            ]
            return {
                'n_intermediate': forced_n_intermediate,
                'x_positions': fallback_x,
                'reasoning': 'Fallback: LLM parse failed, using evenly spaced positions'
            }
        return None

    # ═══════════════════════════════════════════════════════════
    # Cooperative Multi-Agent Methods (core differentiator)
    # ═══════════════════════════════════════════════════════════

    def _topology_propose(self, L, H, loads, supports, simp_summary,
                           y_bot, y_top, temperature=0.4, use_mirror=True,
                           prev_layouts=None, verbose=True, design_hint=""):
        """
        Stage 1-A: Topology Agent proposes initial layout.
        This is the 'original proposal' that Critic will review.
        """
        layout = self._decide_node_layout(
            L, H, loads, supports,
            simp_summary=simp_summary,
            y_bot=y_bot, y_top=y_top,
            temperature=temperature,
            use_mirror=use_mirror,
            prev_layouts=prev_layouts,
            design_hint=design_hint
        )
        if verbose and layout:
            print(f"    [Topology Agent] n_intermediate={layout.get('n_intermediate')}, "
                  f"x={layout.get('x_positions')}")
        return layout

    def _critic_review(self, layout, L, H, loads, supports, simp_summary,
                        dy, y_bot, y_top, verbose=True):
        """
        Stage 1-B: Critic Agent reviews the proposal.
        Returns structured critique with severity and specific issues.
        """
        if not layout or layout.get('n_intermediate', 0) == 0:
            return {
                'severity': 'none',
                'issues': [],
                'affected_positions': [],
                'recommendation': 'Direct strut model — no intermediate nodes to review'
            }

        # Compute diagonal angles for Critic's reference
        sorted_supports = sorted(supports, key=lambda s: s[0])
        sorted_loads = sorted(loads, key=lambda l: l[0])

        # Build x-sequence: supports (bot) + loads (top) + intermediate
        x_positions = layout.get('x_positions', [])
        expected_angles = []
        if x_positions:
            sup_left = sorted_supports[0][0]
            for x in x_positions:
                dx = abs(x - sup_left)
                if dx > 0:
                    angle = math.degrees(math.atan2(dy, dx))
                    expected_angles.append((x, round(angle, 1)))

        user_prompt = (
            f"PROPOSED LAYOUT:\n"
            f"  n_intermediate: {layout.get('n_intermediate')}\n"
            f"  x_positions: {layout.get('x_positions')}\n"
            f"  Topology Agent's reasoning: {layout.get('reasoning', '')[:200]}\n\n"
            f"BEAM CONTEXT:\n"
            f"  Beam: {L}mm x {H}mm (L/H = {L/H:.2f})\n"
            f"  Supports: {[s[0] for s in sorted_supports]}\n"
            f"  Loads: {[l[0] for l in sorted_loads]}\n"
            f"  Chord height dy: {dy:.0f}mm\n\n"
            f"COMPUTED DIAGONAL ANGLES (from nearest support to each intermediate x):\n"
        )
        for x, ang in expected_angles:
            user_prompt += f"  x={x}: angle ~{ang}°\n"

        user_prompt += f"\n{simp_summary}\n\n"
        user_prompt += (
            "Critique the proposal. Be specific. Reference actual x-positions and angles.\n"
            "Respond with JSON only: "
            '{"severity": "major|minor|none", "issues": [...], '
            '"affected_positions": [...], "recommendation": "..."}'
        )

        response = self.client.chat(
            model=self.critic_model,
            system=CRITIC_AGENT_PROMPT,
            user=user_prompt,
            temperature=0.2
        )
        critique = parse_json_from_text(response)

        # Fallback structure
        if not critique:
            critique = {
                'severity': 'none',
                'issues': ['Critic could not parse layout'],
                'affected_positions': [],
                'recommendation': 'Proceed with original'
            }

        # Normalize fields
        critique.setdefault('severity', 'minor')
        critique.setdefault('issues', [])
        critique.setdefault('affected_positions', [])
        critique.setdefault('recommendation', '')

        if verbose:
            print(f"    [Critic Agent] severity={critique['severity']}, "
                  f"{len(critique['issues'])} issue(s)")
            for iss in critique['issues'][:3]:
                print(f"      - {iss[:100]}")

        return critique

    def _revision_step(self, original_layout, critique, L, H, loads, supports,
                        simp_summary, y_bot, y_top, use_mirror=True,
                        verbose=True):
        """
        Stage 1-C: Revision Agent produces revised layout addressing Critic's issues.
        If severity is 'none', returns original unchanged.
        """
        # No revision needed
        if critique.get('severity') == 'none':
            if verbose:
                print(f"    [Revision Agent] skipped (severity=none)")
            return dict(original_layout)  # copy

        # Compute x-range for validation (reuse _decide_node_layout logic)
        if y_bot is not None and y_top is not None:
            dy = y_top - y_bot
        else:
            dy = H - 275

        sorted_supports = sorted(supports, key=lambda s: s[0])
        sorted_loads = sorted(loads, key=lambda l: l[0])

        if use_mirror:
            sup_x = sorted_supports[0][0]
            load_x = sorted_loads[0][0]
            x_lo, x_hi = self.compute_x1_range(sup_x, load_x, L, dy)
            max_intermediate = max(1, int(round(L / H)) - 1)
        else:
            sup_left = sorted_supports[0][0]
            sup_right = sorted_supports[-1][0]
            dx_min = dy / math.tan(math.radians(65))
            x_lo = sup_left + max(dx_min, 50)
            x_hi = sup_right - max(dx_min, 50)
            max_intermediate = max(1, int(round(L / H)) * 2 - 2)

        user_prompt = (
            f"ORIGINAL PROPOSAL (from Topology Agent):\n"
            f"  n_intermediate: {original_layout.get('n_intermediate')}\n"
            f"  x_positions: {original_layout.get('x_positions')}\n"
            f"  reasoning: {original_layout.get('reasoning', '')[:200]}\n\n"
            f"CRITIC'S ASSESSMENT:\n"
            f"  severity: {critique.get('severity')}\n"
            f"  issues: {critique.get('issues')}\n"
            f"  affected_positions: {critique.get('affected_positions')}\n"
            f"  recommendation: {critique.get('recommendation')}\n\n"
            f"CONSTRAINTS:\n"
            f"  Beam: {L}mm x {H}mm\n"
            f"  Valid x-range: [{x_lo:.0f}, {x_hi:.0f}]\n"
            f"  Max n_intermediate: {max_intermediate}\n\n"
            f"{simp_summary}\n\n"
            "Produce a revised layout. Keep what worked, fix what the Critic flagged.\n"
            'Respond with JSON only: {"n_intermediate": N, "x_positions": [...], "reasoning": "..."}'
        )

        response = self.client.chat(
            model=self.revision_model,
            system=REVISION_AGENT_PROMPT,
            user=user_prompt,
            temperature=0.4
        )
        revised = parse_json_from_text(response)

        # Fallback
        if not revised or 'x_positions' not in revised:
            if verbose:
                print(f"    [Revision Agent] parse failed, keeping original")
            return dict(original_layout)

        # Validate & clamp
        target_n = int(revised.get('n_intermediate', original_layout.get('n_intermediate', 1)))
        target_n = max(0, min(max_intermediate, target_n))

        x_positions = revised.get('x_positions', [])
        validated = []
        for x in x_positions[:target_n]:
            try:
                x = int(x)
                x = max(int(x_lo), min(int(x_hi), x))
                validated.append(x)
            except (ValueError, TypeError):
                continue

        # Fill missing evenly
        if len(validated) < target_n:
            full_set = [
                int(x_lo + (x_hi - x_lo) * (i + 1) / (target_n + 1))
                for i in range(target_n)
            ]
            for v in full_set:
                if len(validated) >= target_n:
                    break
                if v not in validated:
                    validated.append(v)

        validated = sorted(set(validated))[:target_n]

        revised_layout = {
            'n_intermediate': len(validated),
            'x_positions': validated,
            'reasoning': revised.get('reasoning', 'Revised based on Critic feedback')
        }

        if verbose:
            print(f"    [Revision Agent] revised x={validated}")

        return revised_layout

    def _decision_step(self, original_layout, revised_layout, critique,
                        verbose=True):
        """
        Stage 1-D: Decision Agent chooses between original and revised.
        Returns the chosen layout and the decision record.
        """
        # Identical proposals: skip
        if (original_layout.get('x_positions') == revised_layout.get('x_positions')
            and original_layout.get('n_intermediate') == revised_layout.get('n_intermediate')):
            if verbose:
                print(f"    [Decision Agent] skipped (original == revised)")
            return original_layout, {
                'choice': 'original',
                'justification': 'original and revised are identical'
            }

        # Critic said 'none' → trust original
        if critique.get('severity') == 'none':
            return original_layout, {
                'choice': 'original',
                'justification': 'Critic found no issues'
            }

        user_prompt = (
            f"CRITIC'S ASSESSMENT:\n"
            f"  severity: {critique.get('severity')}\n"
            f"  issues: {critique.get('issues')}\n\n"
            f"ORIGINAL PROPOSAL (Topology Agent):\n"
            f"  x_positions: {original_layout.get('x_positions')}\n"
            f"  reasoning: {original_layout.get('reasoning', '')[:150]}\n\n"
            f"REVISED PROPOSAL (Revision Agent):\n"
            f"  x_positions: {revised_layout.get('x_positions')}\n"
            f"  reasoning: {revised_layout.get('reasoning', '')[:150]}\n\n"
            "Which proposal is better given the Critic's concerns?\n"
            'Respond with JSON only: {"choice": "original"|"revised", "justification": "..."}'
        )

        response = self.client.chat(
            model=self.decision_model,
            system=DECISION_AGENT_PROMPT,
            user=user_prompt,
            temperature=0.1
        )
        decision = parse_json_from_text(response)

        if not decision or decision.get('choice') not in ('original', 'revised'):
            # Heuristic fallback: trust severity
            choice = 'revised' if critique.get('severity') == 'major' else 'original'
            decision = {
                'choice': choice,
                'justification': f'Fallback: severity={critique.get("severity")}'
            }

        if verbose:
            print(f"    [Decision Agent] choice={decision['choice']}")
            print(f"      justification: {decision.get('justification', '')[:100]}")

        chosen = revised_layout if decision['choice'] == 'revised' else original_layout
        return chosen, decision

    def _cooperative_layout(self, L, H, loads, supports, simp_summary,
                             y_bot, y_top, temperature=0.4, use_mirror=True,
                             prev_layouts=None, verbose=True, design_hint=""):
        """
        Orchestrate the 3-agent cooperative flow for one topology call.

        Flow:
            Topology Agent → Critic Agent → Revision Agent
            Both original and revised are returned as candidates.

        Returns:
            (layouts_list, agent_trace)
            layouts_list: [original] or [original, revised] (if different)
        """
        if y_bot is not None and y_top is not None:
            dy = y_top - y_bot
        else:
            dy = H - 275

        if verbose:
            print(f"  ┌── Cooperative Agent Flow ──┐")

        # Stage A: Topology
        topology_layout = self._topology_propose(
            L, H, loads, supports, simp_summary,
            y_bot, y_top, temperature=temperature,
            use_mirror=use_mirror, prev_layouts=prev_layouts,
            verbose=verbose, design_hint=design_hint
        )

        if not topology_layout:
            if verbose:
                print(f"  │ [Topology Agent] failed")
                print(f"  └────────────────────────────┘")
            return [], {
                'topology': None, 'critique': None,
                'revision': None
            }

        # 0-node special case: no critique needed
        if topology_layout.get('n_intermediate', 0) == 0:
            if verbose:
                print(f"  │ [Skipping Critic/Revision — 0 nodes]")
                print(f"  └────────────────────────────┘")
            return [topology_layout], {
                'topology': topology_layout,
                'critique': {'severity': 'none', 'issues': []},
                'revision': None
            }

        # Stage B: Critic
        critique = self._critic_review(
            topology_layout, L, H, loads, supports, simp_summary,
            dy, y_bot, y_top, verbose=verbose
        )

        # Stage C: Revision
        revised_layout = self._revision_step(
            topology_layout, critique, L, H, loads, supports,
            simp_summary, y_bot, y_top, use_mirror=use_mirror,
            verbose=verbose
        )

        # Both original and revised become candidates (no Decision needed)
        layouts = [topology_layout]
        is_same = (topology_layout.get('x_positions') == revised_layout.get('x_positions')
                   and topology_layout.get('n_intermediate') == revised_layout.get('n_intermediate'))

        if not is_same:
            layouts.append(revised_layout)
            if verbose:
                print(f"    [Both kept] original + revised → 2 candidates")
        else:
            if verbose:
                print(f"    [Same] original == revised → 1 candidate")

        if verbose:
            print(f"  └────────────────────────────┘")

        trace = {
            'topology': topology_layout,
            'critique': critique,
            'revision': revised_layout if not is_same else None
        }
        return layouts, trace

    # ═══════════════════════════════════════════════════════════
    # End of cooperative methods
    # ═══════════════════════════════════════════════════════════

    # ═══════════════════════════════════════════════════════════
    # Nelder-Mead x-coordinate optimization
    # ═══════════════════════════════════════════════════════════

    def _build_and_score(self, x_positions, n_intermediate, L, H, b, loads, support_positions,
                          y_bot, y_top, use_mirror):
        """
        Objective function for Nelder-Mead: given x_positions, build STM and return score.
        Returns large penalty if STM is invalid.
        """
        x_sorted = sorted([int(round(x)) for x in x_positions])

        # ── Minimum distance from loads/supports: 200mm ──
        min_dist = 200
        fixed_xs = [lx for lx, _ in loads] + [sx for sx, _ in support_positions]
        for xi in x_sorted:
            for fx in fixed_xs:
                if abs(xi - fx) < min_dist:
                    return 1e15  # Too close to load/support
        # ── Minimum distance between intermediate nodes: 200mm ──
        for i in range(len(x_sorted) - 1):
            if x_sorted[i+1] - x_sorted[i] < min_dist:
                return 1e15  # Nodes too close to each other

        layout = {
            'n_intermediate': n_intermediate,
            'x_positions': x_sorted,
            'reasoning': 'nelder-mead evaluation'
        }

        # Build nodes
        nodes_data = self._build_nodes_from_layout(
            layout, L, H, loads, support_positions, y_bot, y_top, use_mirror=use_mirror
        )

        # Build connections
        fixed, candidates = enumerate_ground_structure(
            nodes_data['nodes'], H, y_bot=y_bot, y_top=y_top
        )
        all_diag_indices = self._remove_crossing_selections(
            candidates, list(range(len(candidates)))
        )
        all_connections = list(fixed)
        for idx in all_diag_indices:
            all_connections.append(candidates[idx]['nodes'])

        stm = {
            'nodes': nodes_data['nodes'],
            'connections': all_connections,
            'supports': nodes_data['supports']
        }

        # Validate
        val = code_validate(stm, L, H, support_positions=support_positions, loads=loads)
        if not val['valid']:
            return 1e15  # penalty (must be larger than any valid raw_TL)

        # Score via truss analysis
        detail = score_stm_detailed(stm, L, H, loads, support_positions)
        if not detail.get('analysis_success'):
            return 1e15

        # Physical validity check
        truss_result = {
            'success': detail.get('analysis_success', False),
            'reactions': detail.get('reactions', {}),
            'member_forces': {k: v['force_N'] for k, v in detail.get('member_forces', {}).items()}
        }
        valid, reason = check_physical_validity(truss_result, stm, loads)
        if not valid:
            return 1e15

        raw_TL = detail.get('raw_TL', 1e15)
        # Zero-force penalty: discourage positions that create unused members
        zero_count = sum(1 for v in detail.get('member_forces', {}).values()
                        if abs(v.get('force_N', 0)) < 1.0)
        return raw_TL * (1 + zero_count * 0.5)

    def _optimize_x_nelder_mead(self, layout, L, H, b, loads, support_positions,
                                  y_bot, y_top, use_mirror=True, verbose=True,
                                  maxiter=20):
        """
        Optimize x-positions using Nelder-Mead (scipy).

        Args:
            layout: LLM's initial layout with n_intermediate and x_positions
            maxiter: max function evaluations

        Returns:
            optimized_layout, optimization_record
        """
        n_intermediate = layout.get('n_intermediate', 0)
        x_init = layout.get('x_positions', [])

        # 0 nodes: nothing to optimize
        if n_intermediate == 0 or len(x_init) == 0:
            return layout, {'skipped': True, 'reason': '0 nodes'}

        try:
            from scipy.optimize import minimize
        except ImportError:
            if verbose:
                print(f"    [Optimizer] scipy not available, skipping")
            return layout, {'skipped': True, 'reason': 'scipy not available'}

        # Compute bounds
        if y_bot is not None and y_top is not None:
            dy = y_top - y_bot
        else:
            dy = H - 275

        sorted_supports = sorted(support_positions, key=lambda s: s[0])
        sorted_loads = sorted(loads, key=lambda l: l[0])

        if use_mirror:
            x_lo, x_hi = self.compute_x1_range(
                sorted_supports[0][0], sorted_loads[0][0], L, dy
            )
        else:
            dx_min = dy / math.tan(math.radians(65))
            x_lo = sorted_supports[0][0] + max(dx_min, 50)
            x_hi = sorted_supports[-1][0] - max(dx_min, 50)

        # Initial score
        init_score = self._build_and_score(
            x_init, n_intermediate, L, H, b, loads, support_positions,
            y_bot, y_top, use_mirror
        )

        if verbose:
            print(f"    [Optimizer] initial x={x_init}, ΣTL={init_score/1e6:.1f} kN·m")

        # Objective wrapper (clamp to bounds)
        eval_count = [0]
        def objective(x_array):
            eval_count[0] += 1
            x_clamped = [max(x_lo, min(x_hi, xi)) for xi in x_array]
            return self._build_and_score(
                x_clamped, n_intermediate, L, H, b, loads, support_positions,
                y_bot, y_top, use_mirror
            )

        x0 = np.array(x_init, dtype=float)

        # Constrained simplex: ±500mm from LLM position (not full range)
        # This respects LLM's structural judgment while allowing fine-tuning
        max_move = 500  # mm
        n_dim = len(x0)
        simplex = np.zeros((n_dim + 1, n_dim))
        simplex[0] = x0  # LLM's suggestion as first vertex

        for i in range(n_dim):
            vertex = x0.copy()
            # Try moving ±max_move, but clamp within valid bounds
            if x0[i] < (x_lo + x_hi) / 2:
                vertex[i] = min(x0[i] + max_move, x_hi)
            else:
                vertex[i] = max(x0[i] - max_move, x_lo)
            simplex[i + 1] = vertex

        result = minimize(
            objective, x0,
            method='Nelder-Mead',
            options={
                'maxfev': maxiter,
                'xatol': 5.0,     # 5mm tolerance
                'fatol': 100.0,   # 100 N·mm tolerance
                'adaptive': True,
                'initial_simplex': simplex
            }
        )

        # Clamp and round final result
        x_opt = sorted([
            int(round(max(x_lo, min(x_hi, xi))))
            for xi in result.x
        ])

        final_score = self._build_and_score(
            x_opt, n_intermediate, L, H, b, loads, support_positions,
            y_bot, y_top, use_mirror
        )

        improved = final_score < init_score

        opt_record = {
            'skipped': False,
            'x_init': x_init,
            'x_optimized': x_opt,
            'score_init': round(init_score / 1e6, 2),
            'score_final': round(final_score / 1e6, 2),
            'improved': improved,
            'n_evals': eval_count[0],
            'converged': result.success
        }

        if verbose:
            status = "improved" if improved else "no improvement"
            print(f"    [Optimizer] final x={x_opt}, ΣTL={final_score/1e6:.1f} kN·m "
                  f"({status}, {eval_count[0]} evals)")

        if improved:
            optimized_layout = {
                'n_intermediate': n_intermediate,
                'x_positions': x_opt,
                'reasoning': f"Nelder-Mead optimized from {x_init} to {x_opt}"
            }
            return optimized_layout, opt_record
        else:
            return layout, opt_record

    # ═══════════════════════════════════════════════════════════
    # End of Nelder-Mead optimizer
    # ═══════════════════════════════════════════════════════════

    # ── Stage 1-B: Code builds nodes from layout (no LLM) ──
    @staticmethod
    def _build_nodes_from_layout(layout, L, H, loads, supports, y_bot, y_top,
                                  use_mirror=True):
        """
        Deterministically build nodes from LLM's layout decision.
        y_bot and y_top are computed from material parameters by compute_chord_positions().

        Args:
            use_mirror: if True, mirror each x-position about L/2 (for symmetric loads).
                        if False, use LLM's x_positions as-is (LLM decides whole beam).
        """
        sorted_supports = sorted(supports, key=lambda s: s[0])
        sorted_loads = sorted(loads, key=lambda l: l[0])

        # Collect all x-positions for bottom and top chords
        bot_xs = set()
        top_xs = set()

        # Fixed: supports (bottom), loads (top)
        for sx, stype in sorted_supports:
            bot_xs.add(sx)
        for lx, lfy in sorted_loads:
            top_xs.add(lx)

        # Intermediate: vertical pairs (both top and bottom)
        for x in layout['x_positions']:
            bot_xs.add(x)
            top_xs.add(x)
            if use_mirror:
                mirror_x = L - x
                bot_xs.add(mirror_x)
                top_xs.add(mirror_x)

        # Sort and assign node IDs
        bot_xs = sorted(bot_xs)
        top_xs = sorted(top_xs)

        nodes = {}
        node_idx = 0

        # Bottom chord (left to right)
        sup_map = {}  # track support node IDs
        for x in bot_xs:
            nid = generate_node_id(node_idx)
            nodes[nid] = [x, y_bot]
            # Check if this x matches a support
            for sx, stype in sorted_supports:
                if abs(x - sx) < 1:
                    sup_map[nid] = stype
            node_idx += 1

        # Top chord (left to right)
        for x in top_xs:
            nid = generate_node_id(node_idx)
            nodes[nid] = [x, y_top]
            node_idx += 1

        supports_dict = {}
        for nid, stype in sup_map.items():
            supports_dict[nid] = stype

        return {'nodes': nodes, 'supports': supports_dict}

    def _remove_crossing_selections(self, candidates, selected):
        """Remove crossing diagonals from selection (keep earlier ones)."""
        final = []
        for idx in selected:
            crosses = False
            for existing in final:
                if existing in candidates[idx].get('crosses_with', []):
                    crosses = True
                    break
            if not crosses:
                final.append(idx)
        return final

    # ── Generate candidates from one topology call ──
    def _generate_one_candidate(self, L, H, b, loads, support_positions,
                                 candidate_id, simp_summary="",
                                 forced_n_intermediate=None, verbose=True,
                                 y_bot=None, y_top=None, temperature=0.4,
                                 use_mirror=True, prev_layouts=None,
                                 design_hint=""):
        """
        One topology call → potentially 2 candidates (original + revised).
        Returns list of (stm, record) tuples.
        """
        cid = candidate_id
        results = []

        nodes_label = f"{forced_n_intermediate} nodes (forced)" if forced_n_intermediate is not None else "free"
        mirror_tag = "M" if use_mirror else "NM"
        if verbose:
            print(f"\n  ┌── Topology Call {cid} [{nodes_label}, {mirror_tag}] ──┐")

        # ── Stage 1-A: Get layouts from cooperative flow ──
        layouts_list = []
        agent_trace = None

        for attempt in range(1, self.max_retries + 1):
            if forced_n_intermediate == 0:
                layout = self._decide_node_layout(L, H, loads, support_positions,
                                                   forced_n_intermediate=0, y_bot=y_bot, y_top=y_top,
                                                   temperature=temperature,
                                                   use_mirror=use_mirror)
                if layout:
                    layouts_list = [layout]
                if verbose:
                    print(f"  │ [Code] 0 nodes -> direct strut model")
                break

            if verbose:
                print(f"  │ [Cooperative Flow] Attempt {attempt}/{self.max_retries} (temp={temperature:.2f})...")

            t0 = time.time()
            layouts_list, agent_trace = self._cooperative_layout(
                L, H, loads, support_positions,
                simp_summary=simp_summary,
                y_bot=y_bot, y_top=y_top,
                temperature=temperature,
                use_mirror=use_mirror,
                prev_layouts=prev_layouts,
                verbose=verbose,
                design_hint=design_hint
            )
            elapsed = time.time() - t0

            if not layouts_list:
                if verbose: print(f"  │   [FAIL] Cooperative flow failed ({elapsed:.1f}s)")
                continue

            if verbose:
                print(f"  │   Total {elapsed:.1f}s -> {len(layouts_list)} layout(s)")
            break

        if not layouts_list:
            if verbose: print(f"  │ [FAIL] No layouts\n  └──────────────────────────┘")
            results.append((None, {'candidate_id': cid, 'result': 'LAYOUT_FAIL'}))
            return results

        # ── Process each layout ──
        for li, layout in enumerate(layouts_list):
            sub_id = f"{cid}" if len(layouts_list) == 1 else f"{cid}{'a' if li == 0 else 'b'}"
            record = {'candidate_id': sub_id, 'result': None, 'layout': layout}
            if agent_trace is not None:
                record['agent_trace'] = agent_trace
            record['layout_source'] = 'topology' if li == 0 else 'revision'

            if verbose:
                src = "Topology" if li == 0 else "Revision"
                print(f"  │ [{src}] n_intermediate={layout['n_intermediate']}, x={layout['x_positions']}")

            # ── Conditional Nelder-Mead: only when LLM position is physically invalid ──
            if layout.get('n_intermediate', 0) > 0:
                # Test LLM's position first
                test_score = self._build_and_score(
                    layout['x_positions'], layout['n_intermediate'],
                    L, H, b, loads, support_positions,
                    y_bot, y_top, use_mirror
                )
                if test_score >= 1e15:
                    # LLM position is invalid → Nelder-Mead tries to fix it
                    if verbose:
                        print(f"    [LLM position invalid] Running Nelder-Mead correction...")
                    optimized_layout, opt_record = self._optimize_x_nelder_mead(
                        layout, L, H, b, loads, support_positions,
                        y_bot, y_top, use_mirror=use_mirror, verbose=verbose
                    )
                    record['optimization'] = opt_record
                    if opt_record.get('improved'):
                        layout = optimized_layout
                        record['layout_optimized'] = layout
                else:
                    if verbose:
                        print(f"    [LLM position valid] ΣTL={test_score/1e6:.1f} kN·m — Nelder-Mead skipped")

            # ── Build nodes ──
            nodes_data = self._build_nodes_from_layout(layout, L, H, loads, support_positions,
                                                        y_bot, y_top, use_mirror=use_mirror)
            n_nodes = len(nodes_data['nodes'])
            if verbose:
                print(f"  │ [Code] Built {n_nodes} nodes")

            plot_nodes_only(nodes_data, L, H, loads=loads, support_positions=support_positions,
                save_path=self._save_path(f'01_C{sub_id}_nodes.png'),
                title=f'C{sub_id} — {n_nodes}n ({layout["n_intermediate"]}p, x={layout["x_positions"]})')

            # ── Build connections ──
            fixed, candidates = enumerate_ground_structure(nodes_data['nodes'], H,
                                                            y_bot=y_bot, y_top=y_top)
            all_diag_indices = self._remove_crossing_selections(candidates, list(range(len(candidates))))
            all_connections = list(fixed)
            for idx in all_diag_indices:
                all_connections.append(candidates[idx]['nodes'])

            if verbose:
                print(f"  │ [Code] Connections: {len(fixed)} fixed + {len(all_diag_indices)} diags = {len(all_connections)} total")
                for c in candidates:
                    incl = "✓" if c['id'] in all_diag_indices else "✗"
                    print(f"  │   [{incl}] {c['description']}")

            stm_candidate = {
                'nodes': nodes_data['nodes'],
                'connections': all_connections,
                'supports': nodes_data['supports'],
                'design_notes': f"layout={layout}, auto_diags={all_diag_indices}"
            }

            val_result = code_validate(stm_candidate, L, H,
                                       support_positions=support_positions, loads=loads)

            if val_result['valid']:
                plot_stm(stm_candidate, L, H, loads=loads, support_positions=support_positions,
                    save_path=self._save_path(f'02_C{sub_id}_PASS.png'),
                    title=f'C{sub_id} [{n_nodes}n, {len(all_connections)}m] [PASS]', status='PASS')
                if verbose:
                    print(f"  │   [PASS]")
                    for w in val_result['warnings']: print(f"  │     Warning: {w}")
                record['result'] = 'PASS'
                results.append((stm_candidate, record))
            else:
                plot_stm(stm_candidate, L, H, loads=loads, support_positions=support_positions,
                    save_path=self._save_path(f'02_C{sub_id}_FAIL.png'),
                    title=f'C{sub_id} [FAIL]', errors=val_result['errors'], status='FAIL')
                if verbose:
                    print(f"  │   [FAIL]")
                    for e in val_result['errors'][:5]: print(f"  │     {e}")
                record['result'] = 'VALIDATE_FAIL'
                results.append((None, record))

        if verbose:
            print(f"  └──────────────────────────┘")

        return results

    # ── Grammar-based STM Generation ──
    def _best_of_n(self, L, H, b, loads, support_positions, simp_summary="",
                    verbose=True, y_bot=None, y_top=None, use_mirror=True):
        if y_bot is not None and y_top is not None:
            dy = y_top - y_bot
        else:
            dy = H - 275

        sorted_supports = sorted(support_positions, key=lambda s: s[0])
        sorted_loads = sorted(loads, key=lambda l: l[0])

        if verbose:
            print(f"\n{'='*55}")
            print(f"  PHASE 1: Grammar-based STM Generation")
            print(f"  Designer: {self.topology_model}")
            print(f"  Calculator: {self.reasoning_model}")
            print(f"  Rules: SPLIT_SPAN, DONE")
            print(f"{'='*55}")

        # ── Initial state: support + load positions ──
        # Bottom chord: support x-positions
        # Top chord: load x-positions
        bottom_xs = [sx for sx, _ in sorted_supports]  # support positions
        top_xs = [lx for lx, _ in sorted_loads]         # load positions

        if verbose:
            print(f"\n  Initial state:")
            print(f"  Bottom (supports): {bottom_xs}")
            print(f"  Top (loads): {top_xs}")

        # ── Grammar loop: LLM decides where to add intermediate nodes ──
        max_steps = 6
        step_log = []

        for step in range(1, max_steps + 1):
            if verbose:
                print(f"\n  -- Step {step} --")

            # Build state description
            lines = []
            lines.append(f"Bottom chord x-positions: {sorted(bottom_xs)}")
            lines.append(f"Top chord x-positions: {sorted(top_xs)}")

            # Show support-to-load spans
            for sx, _ in sorted_supports:
                nearest_load = min(sorted_loads, key=lambda l: abs(l[0] - sx))
                lx = nearest_load[0]
                span = abs(lx - sx)
                angle = math.degrees(math.atan2(dy, span))

                # Check if already split
                has_intermediate = any(
                    min(sx, lx) + 50 < bx < max(sx, lx) - 50
                    for bx in bottom_xs
                )
                if has_intermediate:
                    lines.append(f"  Support(x={sx:.0f})↔Load(x={lx:.0f}): {span:.0f}mm, angle={angle:.1f}° [already split]")
                else:
                    status = "(TOO SHALLOW)" if angle < 25 else ""
                    lines.append(f"  Support(x={sx:.0f})↔Load(x={lx:.0f}): {span:.0f}mm, angle={angle:.1f}° {status}")

            state_desc = "\n".join(lines)
            if verbose:
                for line in lines:
                    print(f"    {line}")

            # Ask LLM
            user_prompt = (
                f"Beam: {L}mm x {H}mm, dy={dy:.0f}mm\n"
                f"{simp_summary}\n\n"
                f"CURRENT STATE:\n{state_desc}\n\n"
                "Should we add an intermediate node to split a support-to-load span?\n"
                "If a span has angle < 25° and is not yet split → SPLIT_SPAN\n"
                "If all spans are acceptable → DONE\n\n"
                'Respond with JSON only: {"rule": "SPLIT_SPAN" or "DONE", "target": "which span", "reasoning": "why"}'
            )

            response = self.client.chat(
                model=self.topology_model,
                system=GRAMMAR_RULES_PROMPT,
                user=user_prompt,
                temperature=0.3
            )

            decision = parse_json_from_text(response) if response else None
            if not decision or 'rule' not in decision:
                if verbose:
                    print(f"    LLM parse failed, retrying...")
                retry_prompt = user_prompt + '\nRespond with ONLY JSON. Example: {"rule": "DONE", "target": "complete", "reasoning": "all spans ok"}'
                response = self.client.chat(
                    model=self.topology_model,
                    system=GRAMMAR_RULES_PROMPT,
                    user=retry_prompt,
                    temperature=0.5
                )
                decision = parse_json_from_text(response) if response else None
                if not decision or 'rule' not in decision:
                    if verbose:
                        print(f"    Retry failed, stopping")
                    break

            rule = decision['rule'].upper().strip()
            if verbose:
                print(f"    LLM: {rule}")
                print(f"    Reason: {decision.get('reasoning', '')[:80]}")

            step_log.append(decision)

            if rule == 'DONE':
                if verbose:
                    print(f"    → Complete")
                break

            elif rule == 'SPLIT_SPAN':
                # Find which support-to-load span needs splitting
                best_pair = None
                shallowest = 90
                for sx, _ in sorted_supports:
                    nearest_load = min(sorted_loads, key=lambda l: abs(l[0] - sx))
                    lx = nearest_load[0]
                    span = abs(lx - sx)
                    angle = math.degrees(math.atan2(dy, span))
                    has_intermediate = any(
                        min(sx, lx) + 50 < bx < max(sx, lx) - 50
                        for bx in bottom_xs
                    )
                    if not has_intermediate and angle < shallowest:
                        shallowest = angle
                        best_pair = (sx, lx)

                if best_pair:
                    sx, lx = best_pair

                    # Ask nemotron to calculate optimal position
                    calc_prompt = (
                        f"Support at x={sx:.0f}mm, Load at x={lx:.0f}mm\n"
                        f"dy = {dy:.0f}mm\n"
                        f"Calculate the optimal x-position for an intermediate node between them.\n"
                        f"Place at midpoint: x = ({sx:.0f} + {lx:.0f}) / 2\n"
                        'Respond with JSON only: {"x": number, "reasoning": "calculation"}'
                    )

                    calc_response = self.client.chat(
                        model=self.reasoning_model,
                        system="You are a calculator. Compute the requested value. Respond with JSON only.",
                        user=calc_prompt,
                        temperature=0.1
                    )

                    calc_result = parse_json_from_text(calc_response) if calc_response else None
                    if calc_result and 'x' in calc_result:
                        x_mid = int(calc_result['x'])
                        if verbose:
                            print(f"    nemotron: x={x_mid} ({calc_result.get('reasoning', '')[:60]})")
                    else:
                        # Fallback: code calculates
                        x_mid = int((sx + lx) / 2)
                        if verbose:
                            print(f"    nemotron failed, code fallback: x={x_mid}")

                    # Clamp to valid range
                    x_mid = max(int(min(sx, lx)) + 100, min(int(max(sx, lx)) - 100, x_mid))

                    # Add to both chords
                    bottom_xs.append(x_mid)
                    top_xs.append(x_mid)

                    if verbose:
                        print(f"    → Added node pair at x={x_mid} (between {sx:.0f} and {lx:.0f})")
                else:
                    if verbose:
                        print(f"    → No span to split")
                    break

        # ── Build STM from node positions ──
        if verbose:
            print(f"\n  -- Building STM --")
            print(f"    Bottom: {sorted(bottom_xs)}")
            print(f"    Top: {sorted(top_xs)}")

        # Create node dict
        nodes = {}
        supports_dict = {}
        nid = 0

        for x in sorted(set(bottom_xs)):
            node_name = chr(65 + nid)
            nodes[node_name] = (float(x), y_bot)
            # Check if this is a support
            for sx, stype in sorted_supports:
                if abs(x - sx) < 50:
                    supports_dict[node_name] = stype
            nid += 1

        for x in sorted(set(top_xs)):
            node_name = chr(65 + nid)
            nodes[node_name] = (float(x), y_top)
            nid += 1

        if verbose:
            print(f"    Nodes ({len(nodes)}): {dict(sorted(nodes.items()))}")

        # Use enumerate_ground_structure for connections (proven, reliable)
        fixed, diag_candidates = enumerate_ground_structure(
            nodes, H, y_bot=y_bot, y_top=y_top
        )
        all_diag_indices = self._remove_crossing_selections(
            diag_candidates, list(range(len(diag_candidates)))
        )
        connections = list(fixed)
        for idx in all_diag_indices:
            connections.append(diag_candidates[idx]['nodes'])

        if verbose:
            print(f"    Connections: {len(fixed)} fixed + {len(all_diag_indices)} diags = {len(connections)} total")
            for c in diag_candidates:
                incl = "V" if c['id'] in all_diag_indices else "X"
                print(f"      [{incl}] {c['description'][:70]}")

        stm = {
            'nodes': nodes,
            'connections': connections,
            'supports': supports_dict
        }

        # Validate
        val_result = code_validate(stm, L, H, support_positions=support_positions, loads=loads)
        status = 'PASS' if val_result['valid'] else 'FAIL'

        if verbose:
            print(f"    Validation: {status}")
            if not val_result['valid']:
                for e in val_result['errors'][:5]:
                    print(f"      {e}")

        plot_stm(stm, L, H, loads=loads, support_positions=support_positions,
            save_path=self._save_path('02_grammar_result.png'),
            title=f'Grammar STM [{status}]', status=status)

        if not val_result['valid']:
            return None, [], []

        sc = score_stm(stm, L, H, loads, support_positions)
        if verbose:
            print(f"  Grammar score: {sc:.4f}")

        # ── Nelder-Mead optimization ──
        support_xs_set = set(int(round(sx)) for sx, _ in sorted_supports)
        load_xs_set = set(int(round(lx)) for lx, _ in sorted_loads)
        fixed_xs = support_xs_set | load_xs_set

        intermediate_xs = sorted([
            int(round(x)) for nid, (x, y) in nodes.items()
            if abs(y - y_bot) < 50 and int(round(x)) not in fixed_xs
        ])

        if intermediate_xs and sc >= 1e6:
            if verbose:
                print(f"\n  -- Nelder-Mead Optimization --")
                print(f"    Intermediate: {intermediate_xs}")

            nm_layout = {
                'n_intermediate': len(intermediate_xs),
                'x_positions': intermediate_xs,
                'reasoning': 'grammar output'
            }
            nm_result, nm_record = self._optimize_x_nelder_mead(
                nm_layout, L, H, b, loads, support_positions,
                y_bot=y_bot, y_top=y_top,
                use_mirror=False, verbose=verbose
            )
            if nm_result and nm_result.get('x_positions'):
                nm_xs = sorted(nm_result['x_positions'])
                if nm_xs != intermediate_xs:
                    nm_layout_full = {
                        'n_intermediate': len(nm_xs),
                        'x_positions': nm_xs,
                        'reasoning': f'NM from {intermediate_xs}'
                    }
                    nm_nodes = self._build_nodes_from_layout(
                        nm_layout_full, L, H, loads, support_positions, y_bot, y_top,
                        use_mirror=False
                    )
                    nm_stm = {'nodes': nm_nodes['nodes'], 'connections': [], 'supports': nm_nodes['supports']}
                    nm_f, nm_d = enumerate_ground_structure(nm_nodes['nodes'], H, y_bot=y_bot, y_top=y_top)
                    nm_di = self._remove_crossing_selections(nm_d, list(range(len(nm_d))))
                    nm_c = list(nm_f)
                    for idx in nm_di:
                        nm_c.append(nm_d[idx]['nodes'])
                    nm_stm['connections'] = nm_c
                    nm_v = code_validate(nm_stm, L, H, support_positions=support_positions, loads=loads)
                    if nm_v['valid']:
                        nm_sc = score_stm(nm_stm, L, H, loads, support_positions)
                        if nm_sc < sc:
                            if verbose:
                                print(f"    NM improved: {sc:.4f} -> {nm_sc:.4f}")
                            stm, sc, nodes = nm_stm, nm_sc, nm_stm['nodes']

        if verbose:
            print(f"  Final score: {sc:.4f}")

        plot_stm(stm, L, H, loads=loads, support_positions=support_positions,
            save_path=self._save_path('03_best_selected.png'),
            title=f'Grammar STM (score={sc:.4f})', status='PASS')

        self.log.append({
            'phase': 'grammar_generation',
            'steps': len(step_log),
            'rules_applied': [s.get('rule') for s in step_log],
            'score': sc
        })

        candidates = [(stm, sc, 'G1')]
        return stm, step_log, candidates

    # ── Phase 2: Engineering Review ──
    def _engineering_review(self, stm_data, L, H, loads, support_positions, verbose=True):
        if verbose:
            print(f"\n{'='*55}")
            print(f"  PHASE 2: Engineering Review ({self.reviewer_model})")
            print(f"{'='*55}")

        t0 = time.time()

        # Build load/support description dynamically
        load_desc = ", ".join(
            f"x={lx:.0f}mm ({abs(lfy):.0f}kN)" for lx, lfy in sorted(loads, key=lambda l: l[0])
        )
        sup_desc = ", ".join(
            f"x={sx:.0f}mm ({stype})" for sx, stype in sorted(support_positions, key=lambda s: s[0])
        )

        response = self.client.chat(
            model=self.reviewer_model, system=ENGINEERING_REVIEWER_PROMPT,
            user=(
                f"Beam: {L}x{H}mm, L/H={L/H:.2f}\n"
                f"Loads: {load_desc}\n"
                f"Supports: {sup_desc}\n\n"
                f"STM:\n{json.dumps(stm_data, indent=2)}\n\n"
                f"Assess using the 4-criteria checklist."
            ), temperature=0.1)
        elapsed = time.time() - t0

        review = parse_json_from_text(response)
        if review and verbose:
            print(f"  Assessment: {review.get('assessment','?')} ({review.get('total_score','?')}/3.0)")
            for k in ['load_path_score','force_flow_score','symmetry_score','completeness_score']:
                print(f"    {k}: {review.get(k,'?')}/3")
            for iss in review.get('issues', []):
                print(f"  Issue: {iss}")

        if review:
            self.log.append({'phase': 'engineering_review', **review})
            return review
        return {'assessment': 'ACCEPTABLE', 'total_score': 0, 'issues': []}

    # ── KDS Verification & Feedback Loop ──
    @staticmethod
    def _run_kds_verification(stm_data, detail, fck, fy, bw, bearing_plate,
                               cover, stirrup_dia, main_bar_dia, beam_height,
                               phi=0.75, verbose=False):
        """
        Run KDS nodal zone verification on an STM.
        Requires nodal_zone_design module to be importable.

        Returns:
            dict with 'design', 'n_fail', 'n_total', 'fail_nodes', 'available'
        """
        try:
            from nodal_zone_design import design_nodal_zones
        except ImportError:
            return {'available': False, 'n_fail': -1, 'n_total': 0,
                    'fail_nodes': [], 'design': None}

        # Convert member forces from detail to the format design_nodal_zones expects
        nodes = stm_data['nodes']
        connections = stm_data['connections']
        supports = stm_data['supports']

        member_forces = {}
        if detail.get('analysis_success'):
            for (n1, n2), info in detail['member_forces'].items():
                member_forces[(n1, n2)] = info['force_N']

        reactions = {}
        if detail.get('reactions'):
            for nid, (rx, ry) in detail['reactions'].items():
                reactions[nid] = (rx, ry)

        design = design_nodal_zones(
            nodes, connections, member_forces, reactions, supports,
            fck=fck, fy=fy, bw=bw, bearing_plate=bearing_plate,
            cover=cover, stirrup_dia=stirrup_dia, main_bar_dia=main_bar_dia,
            phi=phi, beam_height=beam_height, verbose=verbose
        )

        # Count FAIL nodes
        fail_nodes = []
        for nid, nv in design['node_verification'].items():
            if not nv['all_ok']:
                fail_nodes.append({
                    'node': nid,
                    'type': nv['type'],
                    'beta_n': nv['beta_n'],
                    'checks': nv['checks']
                })

        n_total = len(design['node_verification'])
        return {
            'available': True,
            'design': design,
            'n_fail': len(fail_nodes),
            'n_total': n_total,
            'fail_nodes': fail_nodes
        }

    @staticmethod
    def _format_kds_feedback(kds_result, stm_data):
        """
        Convert KDS verification results into feedback text for Node Layout Agent.
        """
        if not kds_result['available'] or kds_result['n_fail'] == 0:
            return ""

        nodes = stm_data['nodes']
        lines = [
            f"KDS NODAL ZONE VERIFICATION: {kds_result['n_fail']}/{kds_result['n_total']} nodes FAILED",
            ""
        ]

        for fn in kds_result['fail_nodes']:
            nid = fn['node']
            nx, ny = nodes[nid]
            lines.append(f"  Node {nid} ({fn['type']}, βn={fn['beta_n']}) at x={nx:.0f} — FAIL:")
            for chk in fn['checks']:
                if not chk['ok']:
                    ratio = chk['w_required'] / max(chk['w_actual'], 1)
                    lines.append(
                        f"    {chk['member']}: w_required={chk['w_required']:.0f}mm "
                        f"> w_actual={chk['w_actual']:.0f}mm (ratio={ratio:.2f})"
                    )

        lines.append("")
        lines.append("ENGINEERING HINT:")
        lines.append("  wsb = lb*sin(θ) + wt*cos(θ), where θ is the strut angle.")
        lines.append("  Moving intermediate nodes changes θ, which changes wsb and force distribution.")
        lines.append("  Try adjusting x-positions to improve strut angles at FAIL nodes.")

        return "\n".join(lines)

    def _kds_feedback_loop(self, best_stm, best_score, L, H, b,
                            loads, support_positions, simp_summary,
                            fck, fy, bearing_plate, cover, stirrup_dia, main_bar_dia,
                            phi, y_bot, y_top, max_loops=2, verbose=True,
                            use_mirror=True):
        """
        KDS-guided feedback loop: if FAIL nodes exist, ask LLM to adjust x-positions.

        Returns:
            final_stm, final_score, kds_result, feedback_log
        """
        feedback_log = []

        # Initial KDS check
        detail = score_stm_detailed(best_stm, L, H, loads, support_positions)
        kds_result = self._run_kds_verification(
            best_stm, detail, fck, fy, b, bearing_plate,
            cover, stirrup_dia, main_bar_dia, H, phi, verbose=verbose
        )

        if not kds_result['available']:
            if verbose:
                print(f"\n  [KDS] nodal_zone_design module not available, skipping feedback loop")
            return best_stm, best_score, kds_result, feedback_log

        if verbose:
            print(f"\n  [KDS] Initial check: {kds_result['n_fail']}/{kds_result['n_total']} FAIL")

        if kds_result['n_fail'] == 0:
            if verbose:
                print(f"  [KDS] All nodes PASS — no feedback needed")
            return best_stm, best_score, kds_result, feedback_log

        current_stm = best_stm
        current_score = best_score
        current_kds = kds_result
        current_detail = detail

        for loop_i in range(1, max_loops + 1):
            if verbose:
                print(f"\n{'='*55}")
                print(f"  PHASE 2: KDS Feedback Loop ({loop_i}/{max_loops})")
                print(f"{'='*55}")

            # Format feedback
            feedback_text = self._format_kds_feedback(current_kds, current_stm)
            if verbose:
                for line in feedback_text.split('\n')[:8]:
                    print(f"    {line}")

            # Ask LLM for new layout with feedback
            layout = self._decide_node_layout(
                L, H, loads, support_positions,
                simp_summary=simp_summary,
                feedback=feedback_text,
                y_bot=y_bot, y_top=y_top,
                temperature=0.5,
                use_mirror=use_mirror
            )

            if layout is None or layout['n_intermediate'] == 0:
                if verbose:
                    print(f"  [KDS] LLM returned no layout or 0 nodes, stopping")
                break

            if verbose:
                print(f"  [KDS] LLM proposed: {layout['n_intermediate']} intermediate, x={layout['x_positions']}")
                if layout.get('reasoning'):
                    print(f"  [KDS] Reason: {layout['reasoning'][:100]}")

            # Build new STM
            nodes_data = self._build_nodes_from_layout(
                layout, L, H, loads, support_positions, y_bot, y_top,
                use_mirror=use_mirror
            )
            fixed, candidates = enumerate_ground_structure(
                nodes_data['nodes'], H, y_bot=y_bot, y_top=y_top
            )
            all_diag_indices = self._remove_crossing_selections(
                candidates, list(range(len(candidates)))
            )
            all_connections = list(fixed)
            for idx in all_diag_indices:
                all_connections.append(candidates[idx]['nodes'])

            new_stm = {
                'nodes': nodes_data['nodes'],
                'connections': all_connections,
                'supports': nodes_data['supports'],
                'design_notes': f"kds_feedback_loop={loop_i}, layout={layout}"
            }

            # Validate
            val_result = code_validate(new_stm, L, H,
                                       support_positions=support_positions, loads=loads)
            if not val_result['valid']:
                if verbose:
                    print(f"  [KDS] New STM failed validation, keeping previous")
                    for e in val_result['errors'][:3]:
                        print(f"    {e}")
                feedback_log.append({
                    'loop': loop_i, 'layout': layout,
                    'result': 'VALIDATE_FAIL', 'errors': val_result['errors']
                })
                continue

            # Score and KDS check
            new_score = score_stm(new_stm, L, H, loads, support_positions)
            new_detail = score_stm_detailed(new_stm, L, H, loads, support_positions)
            new_kds = self._run_kds_verification(
                new_stm, new_detail, fck, fy, b, bearing_plate,
                cover, stirrup_dia, main_bar_dia, H, phi, verbose=verbose
            )

            if verbose:
                print(f"  [KDS] New: {new_kds['n_fail']}/{new_kds['n_total']} FAIL, score={new_score:.4f}")
                print(f"  [KDS] Old: {current_kds['n_fail']}/{current_kds['n_total']} FAIL, score={current_score:.4f}")

            feedback_log.append({
                'loop': loop_i, 'layout': layout,
                'result': 'IMPROVED' if new_kds['n_fail'] < current_kds['n_fail'] else 'NOT_IMPROVED',
                'old_fail': current_kds['n_fail'], 'new_fail': new_kds['n_fail'],
                'old_score': current_score, 'new_score': new_score
            })

            # Adopt if better fail RATIO (primary) or same ratio but better score (secondary)
            new_ratio = new_kds['n_fail'] / max(new_kds['n_total'], 1)
            old_ratio = current_kds['n_fail'] / max(current_kds['n_total'], 1)
            if (new_ratio < old_ratio or
                (new_ratio == old_ratio and new_score < current_score)):
                if verbose:
                    print(f"  [KDS] >>> ADOPTED (improved)")
                current_stm = new_stm
                current_score = new_score
                current_kds = new_kds
                current_detail = new_detail

                # Save improved STM plot
                plot_stm(new_stm, L, H, loads=loads, support_positions=support_positions,
                    save_path=self._save_path(f'04_kds_feedback_{loop_i}.png'),
                    title=f'KDS Feedback Loop {loop_i} (FAIL: {new_kds["n_fail"]}, score={new_score:.4f})',
                    status='PASS')

                if new_kds['n_fail'] == 0:
                    if verbose:
                        print(f"  [KDS] All nodes PASS — stopping feedback loop")
                    break
            else:
                if verbose:
                    print(f"  [KDS] >>> REJECTED (not improved)")

        self.log.append({
            'phase': 'kds_feedback_loop',
            'initial_fail': kds_result['n_fail'],
            'final_fail': current_kds['n_fail'],
            'loops_run': len(feedback_log),
            'feedback_log': feedback_log
        })

        return current_stm, current_score, current_kds, feedback_log

    # ── Main Pipeline ──
    def generate(self, beam_length, beam_height, beam_width,
                 loads, support_positions,
                 cover=40, stirrup_dia=16, main_bar_dia=32,
                 fck=27, fy=400, bearing_plate=450, phi=0.75,
                 max_feedback_loops=2,
                 mirror_mode='auto',
                 verbose=True):
        """
        Run the full MAS-STM pipeline.

        Args:
            beam_length, beam_height, beam_width: mm
            loads: list of (x_mm, Fy_kN) — negative Fy = downward
            support_positions: list of (x_mm, type_str)
            cover, stirrup_dia, main_bar_dia: mm (for chord position calculation)
            fck: concrete compressive strength (MPa)
            fy: steel yield strength (MPa)
            bearing_plate: bearing plate width (mm)
            phi: strength reduction factor (default 0.75)
            max_feedback_loops: max KDS feedback iterations (0 to disable)
            mirror_mode: 'auto' | 'always' | 'never'
                - auto: mirror if loads are symmetric, else LLM decides whole beam
                - always: force mirror (legacy behavior, symmetric STM guaranteed)
                - never: LLM decides whole beam (no auto-mirroring)
        """
        L, H, b = beam_length, beam_height, beam_width

        # Compute chord positions from material parameters + loading
        y_bot, y_top, wt, ws = compute_chord_positions(
            H, cover, stirrup_dia, main_bar_dia,
            b=b, loads=loads, support_positions=support_positions,
            fck=fck, fy=fy
        )

        # Resolve mirror mode
        use_mirror, mirror_reason = resolve_mirror_mode(mirror_mode, loads, L)

        if verbose:
            print("=" * 65)
            print("  MAS-based STM Generation — Cooperative Multi-Agent Version")
            print("=" * 65)
            print(f"  Beam: {L} x {H} x {b} mm  (L/H = {L/H:.2f})")
            print(f"  Loads: {loads}")
            print(f"  Supports: {support_positions}")
            print(f"  Material: cover={cover}, stirrup={stirrup_dia}, bar={main_bar_dia}")
            print(f"  KDS: fck={fck}MPa, fy={fy}MPa, bearing_plate={bearing_plate}mm, φ={phi}")
            print(f"  Chord positions: y_bot={y_bot:.1f}, y_top={y_top:.1f} (wt={wt:.1f}, ws={ws:.1f})")
            print(f"  Mirror mode: {mirror_mode} -> use_mirror={use_mirror} ({mirror_reason})")
            print(f"  Agents:")
            print(f"    Designer:   {self.topology_model} (grammar rule selection)")
            print(f"    Calculator: {self.reasoning_model} (position computation)")
            print(f"    Builder:    code (connections + analysis + KDS)")
            print(f"  Candidates: {self.n_candidates}")
            print(f"  KDS Feedback Loops: {max_feedback_loops}")
            print("=" * 65)

        if not self.client.is_available():
            print("\n[ERROR] Ollama not running!")
            return {'stm_data': None, 'log': self.log}

        import glob
        for old in glob.glob(os.path.join(self.output_dir, '*.png')):
            os.remove(old)

        start_time = time.time()

        # Phase 0: SIMP topology optimization
        if verbose:
            print(f"\n{'='*55}")
            print(f"  PHASE 0: SIMP Topology Optimization")
            print(f"{'='*55}")

        nelx, nely = auto_simp_mesh(L, H)
        if verbose:
            print(f"  Auto mesh: {nelx}x{nely} (from {L}x{H}mm)")

        simp_result = simp_optimize(
            nelx=nelx, nely=nely, volfrac=0.3, penal=3.0, rmin=1.5, maxiter=80,
            loads=loads, supports=support_positions,
            beam_length=L, beam_height=H
        )
        plot_simp_result(simp_result, loads=loads, supports=support_positions,
            save_path=self._save_path('00_simp_result.png'),
            title=f'SIMP Result (vf=0.3, mesh={nelx}x{nely})')

        simp_summary = summarize_simp_for_llm(simp_result, loads, support_positions)
        if verbose:
            print(f"\n  SIMP Summary for LLM:")
            for line in simp_summary.split('\n')[:10]:
                print(f"    {line}")

        # Phase 1: LLM node layout + code connections
        best_stm, all_records, all_candidates = self._best_of_n(
            L, H, b, loads, support_positions,
            simp_summary=simp_summary, verbose=verbose,
            y_bot=y_bot, y_top=y_top, use_mirror=use_mirror
        )
        if best_stm is None:
            if verbose: print(f"\n[FAIL] No valid candidates.")
            return {'stm_data': None, 'log': self.log, 'records': all_records}

        best_score = score_stm(best_stm, L, H, loads, support_positions)
        plot_stm(best_stm, L, H, loads=loads, support_positions=support_positions,
            save_path=self._save_path('03_best_selected.png'),
            title=f'Best Candidate (score={best_score:.4f})', status='PASS')

        # Phase 2: KDS Feedback Loop
        if max_feedback_loops > 0:
            final_stm, final_score_fb, kds_result, feedback_log = self._kds_feedback_loop(
                best_stm, best_score, L, H, b,
                loads, support_positions, simp_summary,
                fck, fy, bearing_plate, cover, stirrup_dia, main_bar_dia,
                phi, y_bot, y_top,
                max_loops=max_feedback_loops, verbose=verbose,
                use_mirror=use_mirror
            )
        else:
            final_stm = best_stm
            kds_result = None

        # Final
        final_val = code_validate(final_stm, L, H, support_positions=support_positions, loads=loads)
        final_score = score_stm(final_stm, L, H, loads, support_positions)
        elapsed = time.time() - start_time

        plot_stm(final_stm, L, H, loads=loads, support_positions=support_positions,
            save_path=self._save_path('06_final_result.png'),
            title=f'FINAL STM (score={final_score:.4f}, time={elapsed:.0f}s)', status='PASS')

        detail = score_stm_detailed(final_stm, L, H, loads, support_positions)

        if verbose:
            print(f"\n{'='*65}")
            print("  FINAL RESULT")
            print(f"{'='*65}")
            self._print_stm_summary(final_stm)
            print(f"\n{'='*55}")
            print(f"  Structural Analysis (Direct Stiffness Method)")
            print(f"{'='*55}")
            print_truss_analysis(detail)
            print(f"\n  Valid: {final_val['valid']}")
            print(f"  Score: {final_score:.4f}")
            print(f"  Time: {elapsed:.1f}s")
            print(f"{'='*65}")

        saved_files = sorted([f for f in os.listdir(self.output_dir) if f.endswith('.png')])

        # ── Build JSON output ──
        # Best candidate members (using shared helper)
        members_json = build_member_json(final_stm, detail)

        # Candidates summary
        candidates_json = []
        for rec in all_records:
            layout = rec.get('layout', {})
            agent_trace = rec.get('agent_trace', {})

            # Summarize agent trace for JSON
            trace_summary = None
            if agent_trace:
                critique = agent_trace.get('critique', {}) or {}
                decision = agent_trace.get('decision', {}) or {}
                topology = agent_trace.get('topology', {}) or {}
                revision = agent_trace.get('revision', {}) or {}
                trace_summary = {
                    'topology_proposal': {
                        'x_positions': topology.get('x_positions'),
                        'n_intermediate': topology.get('n_intermediate')
                    },
                    'critique': {
                        'severity': critique.get('severity'),
                        'issues': critique.get('issues', []),
                        'affected_positions': critique.get('affected_positions', []),
                        'recommendation': critique.get('recommendation', '')
                    },
                    'revision_proposal': {
                        'x_positions': revision.get('x_positions'),
                        'n_intermediate': revision.get('n_intermediate')
                    },
                    'decision': {
                        'choice': decision.get('choice'),
                        'justification': decision.get('justification', '')
                    }
                }

            candidates_json.append({
                'candidate_id': rec.get('candidate_id'),
                'n_intermediate': layout.get('n_intermediate') if layout else None,
                'x_positions': layout.get('x_positions') if layout else None,
                'result': rec.get('result'),
                'reasoning': layout.get('reasoning', '') if layout else '',
                'agent_trace': trace_summary
            })

        # Reactions
        reactions_json = {}
        if detail.get('reactions'):
            for nid, (rx, ry) in detail['reactions'].items():
                reactions_json[nid] = {
                    'Rx_kN': round(rx / 1000, 1),
                    'Ry_kN': round(ry / 1000, 1)
                }

        output_json = {
            'beam': {
                'length_mm': L, 'height_mm': H, 'width_mm': b,
                'L_over_H': round(L / H, 2)
            },
            'chord_positions': {
                'y_bot': round(y_bot, 1),
                'y_top': round(y_top, 1),
                'wt': round(wt, 1),
                'ws': round(ws, 1)
            },
            'material': {
                'cover': cover, 'stirrup_dia': stirrup_dia, 'main_bar_dia': main_bar_dia
            },
            'mirror': {
                'mode': mirror_mode,
                'use_mirror': use_mirror,
                'reason': mirror_reason
            },
            'loads': [{'x_mm': lx, 'Fy_kN': lfy} for lx, lfy in loads],
            'supports': [{'x_mm': sx, 'type': st} for sx, st in support_positions],
            'nodes': {nid: {'x': round(x, 1), 'y': round(y, 1)}
                      for nid, (x, y) in sorted(final_stm['nodes'].items())},
            'support_nodes': final_stm['supports'],
            'members': members_json,
            'score': round(final_score, 4),
            'raw_TL_kNm': round(detail.get('raw_TL', 0) / 1e6, 1),
            'reactions': reactions_json,
            'candidates': candidates_json,
            'elapsed_time_s': round(elapsed, 1),
            'models': {
                'topology_agent': self.topology_model,
                'critic_agent': self.critic_model,
                'revision_agent': self.revision_model,
                'decision_agent': self.decision_model
            }
        }

        # ── All candidates STM data (for nodal_zone_design) ──
        all_stm_json = []
        for cand_stm, cand_score, cand_id in all_candidates:
            # Find matching record by candidate_id
            cand_record = None
            for rec in all_records:
                if rec.get('candidate_id') == cand_id:
                    cand_record = rec
                    break
            cand_layout = cand_record.get('layout', {}) if cand_record else {}
            cand_n_intermediate = cand_layout.get('n_intermediate', 0) if cand_layout else 0

            cand_detail = score_stm_detailed(cand_stm, L, H, loads, support_positions)
            cand_members = build_member_json(cand_stm, cand_detail)

            cand_reactions = {}
            if cand_detail.get('reactions'):
                for nid_r, (rx, ry) in cand_detail['reactions'].items():
                    cand_reactions[nid_r] = {'Rx_kN': round(rx/1000, 1), 'Ry_kN': round(ry/1000, 1)}

            all_stm_json.append({
                'candidate_id': cand_id,
                'n_intermediate': cand_n_intermediate,
                'score': round(cand_score, 4),
                'nodes': {nid: {'x': round(x, 1), 'y': round(y, 1)}
                          for nid, (x, y) in sorted(cand_stm['nodes'].items())},
                'support_nodes': cand_stm['supports'],
                'members': cand_members,
                'reactions': cand_reactions,
                'raw_TL_kNm': round(cand_detail.get('raw_TL', 0) / 1e6, 1),
            })

        output_json['all_stm_candidates'] = all_stm_json

        json_path = self._save_path('stm_result.json')
        try:
            abs_json_path = os.path.abspath(json_path)
            os.makedirs(os.path.dirname(abs_json_path), exist_ok=True)
            with open(abs_json_path, 'w', encoding='utf-8') as f:
                json.dump(output_json, f, indent=2, ensure_ascii=False)
            if verbose:
                print(f"\n  {len(saved_files)} images saved")
                print(f"  STM data saved: {abs_json_path}")
        except Exception as e:
            if verbose:
                print(f"\n  [WARNING] JSON save failed: {e}")
                print(f"  Attempted path: {json_path}")
                backup_path = os.path.join(os.getcwd(), 'stm_result.json')
                try:
                    with open(backup_path, 'w', encoding='utf-8') as f:
                        json.dump(output_json, f, indent=2, ensure_ascii=False)
                    print(f"  Backup saved: {backup_path}")
                except Exception as e2:
                    print(f"  Backup also failed: {e2}")

        return {
            'stm_data': final_stm, 'validation': final_val, 'score': final_score,
            'kds_result': kds_result if max_feedback_loops > 0 else None,
            'log': self.log, 'records': all_records,
            'elapsed_time': elapsed, 'saved_files': saved_files,
            'output_json': output_json
        }

    def _print_stm_summary(self, stm):
        nodes, connections, supports = stm['nodes'], stm['connections'], stm['supports']
        print(f"\n  Nodes ({len(nodes)}):")
        for nid, (x, y) in sorted(nodes.items()):
            role = f" [{supports[nid]}]" if nid in supports else ""
            print(f"    {nid}: ({x:.1f}, {y:.1f}){role}")
        print(f"\n  Members ({len(connections)}):")
        for n1, n2 in connections:
            if n1 not in nodes or n2 not in nodes:
                print(f"    {n1}-{n2}: INVALID"); continue
            x1, y1 = nodes[n1]; x2, y2 = nodes[n2]
            dx, dy = abs(x2-x1), abs(y2-y1)
            length = math.sqrt(dx**2 + dy**2)
            if dy < 50: mtype = "horizontal"
            elif dx < 50: mtype = "vertical"
            else: mtype = f"diagonal ({math.degrees(math.atan2(dy, dx)):.1f} deg)"
            print(f"    {n1}-{n2}: {length:.0f}mm ({mtype})")


# ===========================================================
# MAIN
# ===========================================================

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  STM Cooperative Multi-Agent Pipeline")
    print("=" * 65)

    # ── Beam configuration ──
    beam_length = 6900
    beam_height = 2000
    beam_width = 500

    # Load sign convention: negative Fy = downward
    loads = [(2225, -2000), (4675, -2000)]
    support_positions = [(225, 'pin'), (6675, 'roller')]

    # Material parameters
    cover = 40
    stirrup_dia = 16
    main_bar_dia = 32

    # KDS verification parameters
    fck = 27
    fy = 400
    bearing_plate = 450

    # Mirror mode: 'auto' (recommended), 'always', 'never'
    MIRROR_MODE = 'auto'
    output_dir = f"plots_cooper_{MIRROR_MODE}"

    # ── Agent models ──
    # Grammar-based: LLM selects rules, code executes them
    mas = MASSTMGenerator(
        topology_model="qwen3:32b",           # Designer (rule selection)
        reviewer_model="qwen3:32b",           # legacy
        critic_model="qwen3:32b",             # legacy
        revision_model="qwen3:32b",           # legacy
        decision_model="qwen3:32b",           # legacy
        reasoning_model="nemotron-3-nano",    # legacy
        max_retries=5,
        n_candidates=3,
        output_dir=output_dir
    )

    result = mas.generate(
        beam_length=beam_length,
        beam_height=beam_height,
        beam_width=beam_width,
        loads=loads,
        support_positions=support_positions,
        cover=cover,
        stirrup_dia=stirrup_dia,
        main_bar_dia=main_bar_dia,
        fck=fck,
        fy=fy,
        bearing_plate=bearing_plate,
        max_feedback_loops=2,
        mirror_mode=MIRROR_MODE,
        verbose=True
    )

    if result['stm_data']:
        print("\n-- Pipeline Log --")
        for entry in result['log']:
            phase = entry.get('phase', '?')
            print(f"  {phase}: {json.dumps({k:v for k,v in entry.items() if k!='phase'}, default=str)[:120]}")

        # Print agent trace summary for each candidate
        print("\n-- Agent Trace Summary --")
        for cand in result['output_json'].get('candidates', []):
            trace = cand.get('agent_trace')
            if trace:
                print(f"\n  C{cand['candidate_id']}:")
                print(f"    Topology proposed:  x={trace['topology_proposal']['x_positions']}")
                print(f"    Critic severity:    {trace['critique']['severity']}")
                if trace['critique']['issues']:
                    print(f"    Critic issues:      {trace['critique']['issues'][0][:80]}")
                print(f"    Revision proposed:  x={trace['revision_proposal']['x_positions']}")
                print(f"    Decision:           {trace['decision']['choice']} "
                      f"({trace['decision']['justification'][:80]})")
    else:
        print("\n>> Generation failed.")
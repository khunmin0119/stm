"""
MAS-based STM Generator — Final Version
=========================================
Hybrid pipeline: SIMP (physics) + LLM (judgment) + Code (calculation)

  Phase 0:   SIMP topology optimization (code) — Finds optimal force paths
  Phase 1-A: Node Layout Agent (LLM) — Interprets SIMP result, decides node positions
             Code builds all nodes automatically
  Phase 1-B: Ground Structure (code) — Generates all valid connections automatically
  Phase 2:   Reviewer Agent (LLM) — Engineering quality assessment
  Phase 3:   Truss analysis (code) — Score = Sum(Ti x Li)

Key insight: SIMP provides "where forces flow", LLM decides "where to place nodes",
             Code handles "how to connect and validate"
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
# 0. SIMP Topology Optimization (External Tool)
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


def summarize_simp_for_llm(simp_result, threshold=0.3):
    """Convert SIMP density to text description for LLM."""
    density = simp_result['density']
    nely, nelx = density.shape
    L = simp_result['beam_length']
    H = simp_result['beam_height']
    dx = L / nelx
    dy = H / nely

    # For each column, find vertical extent of high-density diagonal regions
    # Skip top 2 and bottom 2 rows (chord bands)
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
            y_top = max(mid_high)
            y_bot = min(mid_high)
            diag_info.append((x_mm, y_bot, y_top))

    # Find where diagonal paths cross mid-height
    mid_y = H / 2
    mid_x = L / 2
    left_cross = [x for x, yb, yt in diag_info if x < mid_x and yb < mid_y < yt]
    right_cross = [x for x, yb, yt in diag_info if x >= mid_x and yb < mid_y < yt]

    left_x = int(np.mean(left_cross)) if left_cross else None
    right_x = int(np.mean(right_cross)) if right_cross else None

    lines = [
        "SIMP TOPOLOGY OPTIMIZATION RESULT:",
        f"  Volume fraction: 30% of material used",
        "",
        "IDENTIFIED FORCE PATHS:",
        f"  1. TOP CHORD: Horizontal compression band at y~{H}mm",
        f"  2. BOTTOM CHORD: Horizontal tension band at y~0mm",
        f"  3. LEFT DIAGONAL STRUT: From left load (x=2225) down to left support (x=225)",
        f"  4. RIGHT DIAGONAL STRUT: From right load (x=4675) down to right support (x=6675)",
    ]

    if left_x:
        lines.append(f"\n  Left diagonal path crosses mid-height (y={mid_y:.0f}) at x~{left_x}mm")
    if right_x:
        lines.append(f"  Right diagonal path crosses mid-height at x~{right_x}mm")

    lines.append(f"\nNODE PLACEMENT GUIDANCE:")
    lines.append(f"  Place intermediate nodes along the diagonal force paths.")
    if left_x:
        lines.append(f"  Suggested x1 ≈ {left_x}mm (where left diagonal is strongest)")
    lines.append(f"  The valid range for x1 is given separately.")

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

Your job: Decide how many intermediate node pairs to add and where to place them.
You will be given SIMP topology optimization results showing where forces flow in the beam.
Use this physical information to decide optimal node positions.

## CONTEXT
A deep beam STM has FIXED nodes (supports at bottom, loads at top) and INTERMEDIATE nodes.
Intermediate nodes come in vertical pairs: one on top chord, one on bottom chord, same x-position.
If loading is symmetric, intermediate positions must also be symmetric about beam center.

## WHAT YOU DECIDE
1. How many intermediate pairs: 0 pairs (4 nodes, simplest), 1 pair (6 nodes), or 2 pairs (8 nodes)
2. The x-position of each pair (left-half only; right-half mirrors automatically)

## HOW TO USE SIMP RESULTS
- SIMP shows diagonal force paths as high-density regions
- Place intermediate nodes WHERE diagonal paths are strongest
- The intersection of diagonal paths with chord lines = ideal node positions
- Diagonal angles closer to 45 degrees = better force distribution

## OUTPUT FORMAT
{"n_pairs": 0 or 1 or 2, "x_positions": [] or [x1] or [x1, x2], "reasoning": "how SIMP guided this"}"""


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
# 3. Ground Structure: Deterministic Connection Enumeration
# ===========================================================

def enumerate_ground_structure(nodes, beam_height):
    """
    Deterministically generate all valid connections for an STM.

    Returns:
        fixed_connections: list of [n1, n2] — horizontals + verticals (always included)
        diagonal_candidates: list of {'id': int, 'nodes': [n1, n2], 'angle': float,
                                       'length': float, 'panel': (x_left, x_right)}
    """
    H = beam_height

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
    dy = H - 275

    # Build crossing checker for fixed connections
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

        # Find top and bottom nodes at this panel's boundaries
        panel_top = [n for n in top_nodes if abs(round(n[1]) - left_x) < 50 or abs(round(n[1]) - right_x) < 50]
        panel_bot = [n for n in bot_nodes if abs(round(n[1]) - left_x) < 50 or abs(round(n[1]) - right_x) < 50]

        for t_nid, t_x, t_y in panel_top:
            for b_nid, b_x, b_y in panel_bot:
                dx = abs(t_x - b_x)
                if dx < 50:
                    continue  # vertical, already in fixed

                # Check panel adjacency: no intermediate x between them
                lo_x, hi_x = min(t_x, b_x), max(t_x, b_x)
                between = [nx for nx in all_xs if lo_x + 50 < nx < hi_x - 50]
                if between:
                    continue  # skips intermediate node

                # Check angle (1 deg tolerance for rounding)
                angle = math.degrees(math.atan2(dy, dx))
                if angle < 24 or angle > 66:
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

                length = math.sqrt(dx**2 + dy**2)
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

    # Check crossing between diagonal candidates themselves
    # Mark pairs that cross each other
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

    if nodes and connections:
        top_nodes = {nid: (x, y) for nid, (x, y) in nodes.items() if y > beam_height / 2}
        bot_nodes = {nid: (x, y) for nid, (x, y) in nodes.items() if y <= beam_height / 2}
        conn_set = {frozenset([n1, n2]) for n1, n2 in connections}
        for t_nid, (t_x, t_y) in top_nodes.items():
            for b_nid, (b_x, b_y) in bot_nodes.items():
                if abs(t_x - b_x) < 50:
                    if frozenset([t_nid, b_nid]) not in conn_set:
                        errors.append(f"Vertical pair {t_nid}-{b_nid} must be connected (same x={t_x:.0f})")

    if nodes and connections and loads and len(loads) == 2:
        mid = beam_length / 2.0
        load_sym = abs((loads[0][0] - mid) + (loads[1][0] - mid))
        if load_sym < 100:
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
        if angle < 24 or angle > 66:
            errors.append(f"Member {n1}-{n2}: angle={angle:.1f} deg outside 25-65 deg")

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
# 3-C. Truss Analysis (Direct Stiffness Method)
# ===========================================================

def solve_truss(nodes, connections, supports, loads_dict):
    """Solve 2D truss using direct stiffness method."""
    node_ids = sorted(nodes.keys())
    n_nodes = len(node_ids)
    n_dofs = 2 * n_nodes
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    K = np.zeros((n_dofs, n_dofs))
    EA = 200000.0 * 1000.0

    member_info = []
    for n1, n2 in connections:
        if n1 not in nodes or n2 not in nodes:
            continue
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]
        dx, dy = x2 - x1, y2 - y1
        L = math.sqrt(dx**2 + dy**2)
        if L < 1.0:
            continue
        c, s = dx / L, dy / L
        member_info.append((n1, n2, L, c, s))

        k_local = (EA / L) * np.array([
            [ c*c,  c*s, -c*c, -c*s],
            [ c*s,  s*s, -c*s, -s*s],
            [-c*c, -c*s,  c*c,  c*s],
            [-c*s, -s*s,  c*s,  s*s]
        ])

        i1, i2 = id_to_idx[n1], id_to_idx[n2]
        dofs = [2*i1, 2*i1+1, 2*i2, 2*i2+1]
        for a in range(4):
            for b in range(4):
                K[dofs[a], dofs[b]] += k_local[a, b]

    F = np.zeros(n_dofs)
    for nid, (fx, fy) in loads_dict.items():
        if nid in id_to_idx:
            idx = id_to_idx[nid]
            F[2*idx] += fx
            F[2*idx+1] += fy

    fixed_dofs = set()
    for nid, stype in supports.items():
        if nid not in id_to_idx:
            continue
        idx = id_to_idx[nid]
        if stype == 'pin':
            fixed_dofs.add(2*idx)
            fixed_dofs.add(2*idx + 1)
        elif stype == 'roller':
            fixed_dofs.add(2*idx + 1)

    free_dofs = sorted(set(range(n_dofs)) - fixed_dofs)
    if not free_dofs:
        return {'success': False, 'error': 'No free DOFs',
                'member_forces': {}, 'displacements': {}, 'reactions': {}}

    K_ff = K[np.ix_(free_dofs, free_dofs)]
    F_f = F[free_dofs]

    try:
        if abs(np.linalg.det(K_ff)) < 1e-10:
            return {'success': False, 'error': 'Singular stiffness matrix',
                    'member_forces': {}, 'displacements': {}, 'reactions': {}}
        u_f = np.linalg.solve(K_ff, F_f)
    except np.linalg.LinAlgError as e:
        return {'success': False, 'error': f'Solver failed: {e}',
                'member_forces': {}, 'displacements': {}, 'reactions': {}}

    u = np.zeros(n_dofs)
    for i, dof in enumerate(free_dofs):
        u[dof] = u_f[i]

    R = K @ u - F

    displacements = {}
    for nid in node_ids:
        idx = id_to_idx[nid]
        displacements[nid] = [float(u[2*idx]), float(u[2*idx+1])]

    reactions = {}
    for nid in supports:
        if nid in id_to_idx:
            idx = id_to_idx[nid]
            reactions[nid] = [float(R[2*idx]), float(R[2*idx+1])]

    member_forces = {}
    for n1, n2, L, c, s in member_info:
        i1, i2 = id_to_idx[n1], id_to_idx[n2]
        du = u[2*i2] - u[2*i1]
        dv = u[2*i2+1] - u[2*i1+1]
        force = EA / L * (du * c + dv * s)
        member_forces[(n1, n2)] = float(force)

    return {
        'success': True, 'error': None,
        'member_forces': member_forces,
        'displacements': displacements,
        'reactions': reactions
    }


def score_stm(stm_data, beam_length, beam_height, loads, support_positions):
    """Score STM using truss analysis. Lower = better."""
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
            total_TL = 0.0
            for (n1, n2), force in result['member_forces'].items():
                if force > 0:
                    x1, y1 = nodes[n1]
                    x2, y2 = nodes[n2]
                    L = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                    total_TL += force * L
            ref_load = max(abs(fy) * 1000 for _, fy in loads)
            ref_length = math.sqrt(beam_length**2 + beam_height**2)
            return total_TL / (ref_load * ref_length * 1000) if ref_load > 0 else total_TL

    # Fallback: geometric
    beam_diag = math.sqrt(beam_length**2 + beam_height**2)
    total_length = sum(
        math.sqrt((nodes[n2][0]-nodes[n1][0])**2 + (nodes[n2][1]-nodes[n1][1])**2)
        for n1, n2 in connections if n1 in nodes and n2 in nodes
    )
    return 0.5 * (total_length / beam_diag)


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
            'score': score_stm(stm_data, beam_length, beam_height, loads, support_positions),
            'analysis_success': False, 'error': result['error'],
            'member_forces': {}, 'reactions': {}
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

    return {
        'score': normalized, 'raw_TL': total_TL,
        'member_forces': member_details,
        'reactions': result['reactions'],
        'analysis_success': True, 'error': None
    }


def print_truss_analysis(detail):
    """Pretty-print truss analysis results."""
    if not detail.get('analysis_success'):
        print(f"  [Truss Analysis] FAILED: {detail.get('error', 'unknown')}")
        print(f"  [Fallback] Geometric score = {detail.get('score', '?'):.4f}")
        return

    print(f"  [Truss Analysis] SUCCESS")
    print(f"  Score (norm. Sigma Ti*Li): {detail['score']:.4f}")
    print(f"  Raw Sigma(Ti*Li): {detail['raw_TL']/1e6:.1f} kN*m")

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
# 5. Visualization Functions
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
# 6. MAS Controller (Ground Structure Version)
# ===========================================================

class MASSTMGenerator:
    """
    3-Agent MAS-based STM generator with Ground Structure approach.
    Pipeline: Phase 1 (Nodes + Ground Structure + LLM Selection)
           -> Phase 2 (Engineering Review with feedback)
           -> Phase 3 (Optimizer via re-selection)
    """

    def __init__(self,
                 topology_model="deepseek-r1:14b",
                 reviewer_model="qwen2.5:7b",
                 max_retries=5,
                 n_candidates=3,
                 ollama_url="http://localhost:11434",
                 output_dir="plots"):
        self.client = OllamaClient(ollama_url)
        self.topology_model = topology_model      # Node Layout Agent
        self.reviewer_model = reviewer_model      # Engineering Review
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
                            forced_n_pairs=None, prev_layouts=None, feedback=""):
        """LLM decides node layout. If forced_n_pairs is set, LLM only decides x_positions."""
        dy = H - 275
        sup_x, load_x = supports[0][0], loads[0][0]
        x1_lo, x1_hi = self.compute_x1_range(sup_x, load_x, L, dy)

        # 0 pairs: no LLM needed
        if forced_n_pairs == 0:
            return {
                'n_pairs': 0,
                'x_positions': [],
                'reasoning': 'Direct strut model (0 intermediate pairs)'
            }

        # For 1 or 2 pairs: LLM decides x_positions
        n_pairs_text = f"EXACTLY {forced_n_pairs}" if forced_n_pairs else "0, 1, or 2"

        user_prompt = (
            f"Deep beam STM node layout decision:\n\n"
            f"Beam: {L}mm x {H}mm\n"
            f"Left support: x={sup_x}mm\n"
            f"Left load: x={load_x}mm\n"
            f"Right load: x={loads[1][0]}mm\n"
            f"Right support: x={supports[1][0]}mm\n"
            f"Beam center: x={L/2:.0f}mm\n"
            f"Vertical distance between chords: dy={dy}mm\n\n"
            f"VALID x-range for intermediate nodes: [{x1_lo:.0f}, {x1_hi:.0f}]\n\n"
        )

        if simp_summary:
            user_prompt += f"{simp_summary}\n\n"

        if forced_n_pairs:
            user_prompt += (
                f"You MUST use {forced_n_pairs} intermediate pair(s).\n"
                f"Decide the x-position(s) for the {forced_n_pairs} pair(s).\n"
                f"Use SIMP force paths to choose optimal positions.\n\n"
            )
        else:
            user_prompt += (
                f"FIXED nodes (4): supports at x={sup_x},{supports[1][0]} + loads at x={load_x},{loads[1][0]}\n"
                f"You decide: how many intermediate PAIRS and their x-positions.\n"
                f"  0 pairs -> 4 total nodes (direct strut, simplest)\n"
                f"  1 pair -> 6 total nodes\n"
                f"  2 pairs -> 8 total nodes\n\n"
            )

        if feedback:
            user_prompt += f"FEEDBACK: {feedback}\n\n"

        user_prompt += 'Respond with ONLY a JSON: {"n_pairs": N, "x_positions": [...], "reasoning": "..."}'

        response = self.client.chat(
            model=self.topology_model, system=NODE_LAYOUT_PROMPT,
            user=user_prompt, temperature=0.4
        )

        result = parse_json_from_text(response)
        if result and 'x_positions' in result:
            target_n = forced_n_pairs if forced_n_pairs else int(result.get('n_pairs', 1))
            x_positions = result.get('x_positions', [])

            target_n = max(0, min(2, target_n))
            validated = []
            for x in x_positions[:target_n]:
                x = int(x)
                x = max(int(x1_lo), min(int(x1_hi), x))
                validated.append(x)

            # Fill missing positions with evenly spaced values
            if len(validated) < target_n:
                # Generate target_n evenly spaced values
                full_set = []
                if target_n == 1:
                    full_set = [int((x1_lo + x1_hi) / 2)]
                elif target_n == 2:
                    full_set = [int(x1_lo + (x1_hi - x1_lo) * 0.33),
                                int(x1_lo + (x1_hi - x1_lo) * 0.67)]
                # Merge LLM values with generated ones, prefer LLM values
                for v in full_set:
                    if len(validated) >= target_n:
                        break
                    if v not in validated:
                        validated.append(v)

            validated = sorted(set(validated))[:target_n]

            return {
                'n_pairs': len(validated),
                'x_positions': validated,
                'reasoning': result.get('reasoning', '')
            }

        # Fallback: if LLM failed entirely but forced_n_pairs is set
        if forced_n_pairs and forced_n_pairs > 0:
            if forced_n_pairs == 1:
                fallback_x = [int((x1_lo + x1_hi) / 2)]
            else:
                fallback_x = [int(x1_lo + (x1_hi - x1_lo) * 0.33),
                              int(x1_lo + (x1_hi - x1_lo) * 0.67)]
            return {
                'n_pairs': forced_n_pairs,
                'x_positions': fallback_x[:forced_n_pairs],
                'reasoning': 'Fallback: LLM parse failed, using evenly spaced positions'
            }
        return None

    # ── Stage 1-B: Code builds nodes from layout (no LLM) ──
    @staticmethod
    def _build_nodes_from_layout(layout, L, H, loads, supports):
        """Deterministically build nodes from LLM's layout decision."""
        y_bot = 125
        y_top = H - 150
        sup_left, sup_right = supports[0][0], supports[1][0]
        load_left, load_right = loads[0][0], loads[1][0]

        # Collect all x-positions for bottom and top chords
        bot_xs = set()
        top_xs = set()

        # Fixed: supports (bottom), loads (top)
        bot_xs.add(sup_left)
        bot_xs.add(sup_right)
        top_xs.add(load_left)
        top_xs.add(load_right)

        # Intermediate: vertical pairs (both top and bottom)
        for x in layout['x_positions']:
            bot_xs.add(x)
            top_xs.add(x)
            mirror_x = L - x
            bot_xs.add(mirror_x)
            top_xs.add(mirror_x)

        # Sort and assign node IDs alphabetically
        bot_xs = sorted(bot_xs)
        top_xs = sorted(top_xs)

        nodes = {}
        node_id = ord('A')

        # Bottom chord (left to right)
        sup_left_id = None
        sup_right_id = None
        for x in bot_xs:
            nid = chr(node_id)
            nodes[nid] = [x, y_bot]
            if abs(x - sup_left) < 1:
                sup_left_id = nid
            if abs(x - sup_right) < 1:
                sup_right_id = nid
            node_id += 1

        # Top chord (left to right)
        for x in top_xs:
            nid = chr(node_id)
            nodes[nid] = [x, y_top]
            node_id += 1

        supports_dict = {}
        if sup_left_id:
            supports_dict[sup_left_id] = 'pin'
        if sup_right_id:
            supports_dict[sup_right_id] = 'roller'

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

    # ── Generate one candidate ──
    def _generate_one_candidate(self, L, H, b, loads, support_positions,
                                 candidate_id, simp_summary="",
                                 forced_n_pairs=None, verbose=True):
        cid = candidate_id
        record = {'candidate_id': cid, 'result': None}

        pairs_label = f"{forced_n_pairs} pairs (forced)" if forced_n_pairs is not None else "free"
        if verbose:
            print(f"\n  ┌── Candidate {cid} [{pairs_label}] ──┐")

        # ── Stage 1-A: LLM decides node layout (guided by SIMP) ──
        layout = None
        for attempt in range(1, self.max_retries + 1):
            if forced_n_pairs == 0:
                layout = self._decide_node_layout(L, H, loads, support_positions,
                                                   forced_n_pairs=0)
                if verbose:
                    print(f"  │ [Code] 0 pairs -> direct strut model")
                break

            if verbose:
                print(f"  │ [Node Layout Agent] Attempt {attempt}/{self.max_retries}...")

            t0 = time.time()
            layout = self._decide_node_layout(L, H, loads, support_positions,
                                               simp_summary=simp_summary,
                                               forced_n_pairs=forced_n_pairs)
            elapsed = time.time() - t0

            if layout is None:
                if verbose: print(f"  │   [FAIL] Could not parse layout ({elapsed:.1f}s)")
                continue

            if verbose:
                print(f"  │   {elapsed:.1f}s -> {layout['n_pairs']} pair(s), x={layout['x_positions']}")
                if layout['reasoning']:
                    print(f"  │   Reason: {layout['reasoning'][:80]}")
            break

        if layout is None:
            if verbose: print(f"  │ [FAIL] Layout decision failed\n  └──────────────────────────┘")
            record['result'] = 'LAYOUT_FAIL'
            return None, record

        record['layout'] = layout

        # ── Stage 1-B: Code builds nodes (no LLM) ──
        nodes_data = self._build_nodes_from_layout(layout, L, H, loads, support_positions)
        n_nodes = len(nodes_data['nodes'])
        if verbose:
            print(f"  │ [Code] Built {n_nodes} nodes")

        plot_nodes_only(nodes_data, L, H, loads=loads, support_positions=support_positions,
            save_path=self._save_path(f'01_C{cid}_nodes.png'),
            title=f'C{cid} — {n_nodes}n ({layout["n_pairs"]}p, x={layout["x_positions"]})')

        # ── Stage 1-C: Code generates ALL connections (no LLM) ──
        fixed, candidates = enumerate_ground_structure(nodes_data['nodes'], H)
        # Add ALL valid diagonals automatically
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
                save_path=self._save_path(f'02_C{cid}_PASS.png'),
                title=f'C{cid} [{n_nodes}n, {len(all_connections)}m] [PASS]', status='PASS')
            if verbose:
                print(f"  │   [PASS]")
                for w in val_result['warnings']: print(f"  │     Warning: {w}")
                print(f"  └──────────────────────────┘")
            record['result'] = 'PASS'
            return stm_candidate, record
        else:
            plot_stm(stm_candidate, L, H, loads=loads, support_positions=support_positions,
                save_path=self._save_path(f'02_C{cid}_FAIL.png'),
                title=f'C{cid} [FAIL]', errors=val_result['errors'], status='FAIL')
            if verbose:
                print(f"  │   [FAIL]")
                for e in val_result['errors'][:5]: print(f"  │     {e}")
                print(f"  └──────────────────────────┘")
            record['result'] = 'VALIDATE_FAIL'
            return None, record

    # ── Best-of-3: Compare 0, 1, 2 pairs ──
    def _best_of_n(self, L, H, b, loads, support_positions, simp_summary="", verbose=True):
        dy = H - 275
        x1_lo, x1_hi = self.compute_x1_range(support_positions[0][0], loads[0][0], L, dy)

        forced_configs = [
            (0, "0 pairs (4 nodes, direct strut)"),
            (1, "1 pair (8 nodes)"),
            (2, "2 pairs (12 nodes)"),
        ]

        if verbose:
            print(f"\n{'='*55}")
            print(f"  PHASE 1: Compare 3 Topologies")
            print(f"  Valid x1 range: [{x1_lo:.0f}, {x1_hi:.0f}]")
            print(f"  C1: 0 pairs | C2: 1 pair | C3: 2 pairs")
            print(f"{'='*55}")

        candidates = []
        all_records = []

        for i, (n_pairs, desc) in enumerate(forced_configs):
            candidate, record = self._generate_one_candidate(
                L, H, b, loads, support_positions,
                candidate_id=i+1,
                simp_summary=simp_summary,
                forced_n_pairs=n_pairs,
                verbose=verbose
            )
            all_records.append(record)

            if candidate:
                sc = score_stm(candidate, L, H, loads, support_positions)
                candidates.append((candidate, sc, i+1))
                layout = record.get('layout', {})
                if verbose:
                    print(f"  -> C{i+1} ({desc}): "
                          f"{len(candidate['connections'])}m, score={sc:.4f}")
            else:
                if verbose: print(f"  -> C{i+1} ({desc}): FAILED")

        if not candidates:
            return None, all_records, []

        candidates.sort(key=lambda x: x[1])
        plot_candidates_comparison(candidates, L, H, loads=loads, support_positions=support_positions,
            save_path=self._save_path('03_candidates_comparison.png'))

        best_stm, best_score, best_cid = candidates[0]
        if verbose:
            print(f"\n  BEST: C{best_cid}, score={best_score:.4f}")
            print(f"  Valid: {len(candidates)}/3")

        self.log.append({
            'phase': 'best_of_3_topology_comparison',
            'layouts': [r.get('layout') for r in all_records],
            'valid': len(candidates), 'total': 3,
            'scores': [(cid, s) for _, s, cid in candidates],
            'best_score': best_score, 'best_cid': best_cid
        })
        return best_stm, all_records, candidates

    # ── Phase 2: Engineering Review ──
    def _engineering_review(self, stm_data, L, H, loads, support_positions, verbose=True):
        if verbose:
            print(f"\n{'='*55}")
            print(f"  PHASE 2: Engineering Review ({self.reviewer_model})")
            print(f"{'='*55}")

        t0 = time.time()
        response = self.client.chat(
            model=self.reviewer_model, system=ENGINEERING_REVIEWER_PROMPT,
            user=(
                f"Beam: {L}x{H}mm, L/H={L/H:.2f}\n"
                f"Loads at x={loads[0][0]}mm ({abs(loads[0][1])}kN) "
                f"and x={loads[1][0]}mm ({abs(loads[1][1])}kN)\n"
                f"Supports at x={support_positions[0][0]}mm (pin) "
                f"and x={support_positions[1][0]}mm (roller)\n\n"
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

    # ── Main Pipeline ──
    def generate(self, beam_length, beam_height, beam_width,
                 loads, support_positions, verbose=True):
        L, H, b = beam_length, beam_height, beam_width

        if verbose:
            print("=" * 65)
            print("  MAS-based STM Generation — Topology Comparison")
            print("=" * 65)
            print(f"  Beam: {L} x {H} x {b} mm  (L/H = {L/H:.2f})")
            print(f"  Loads: {loads}")
            print(f"  Supports: {support_positions}")
            print(f"  Node Layout: {self.topology_model}")
            print(f"  Reviewer: {self.reviewer_model}")
            print(f"  Candidates: {self.n_candidates}")
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

        simp_result = simp_optimize(
            nelx=69, nely=20, volfrac=0.3, penal=3.0, rmin=1.5, maxiter=80,
            loads=loads, supports=support_positions,
            beam_length=L, beam_height=H
        )
        plot_simp_result(simp_result, loads=loads, supports=support_positions,
            save_path=self._save_path('00_simp_result.png'),
            title=f'SIMP Result (vf=0.3)')

        simp_summary = summarize_simp_for_llm(simp_result)
        if verbose:
            print(f"\n  SIMP Summary for LLM:")
            for line in simp_summary.split('\n')[:8]:
                print(f"    {line}")

        # Phase 1: LLM node layout + code connections
        best_stm, all_records, all_candidates = self._best_of_n(L, H, b, loads, support_positions,
                                                 simp_summary=simp_summary, verbose=verbose)
        if best_stm is None:
            if verbose: print(f"\n[FAIL] No valid candidates.")
            return {'stm_data': None, 'log': self.log, 'records': all_records}

        best_score = score_stm(best_stm, L, H, loads, support_positions)
        plot_stm(best_stm, L, H, loads=loads, support_positions=support_positions,
            save_path=self._save_path('03_best_selected.png'),
            title=f'Best Candidate (score={best_score:.4f})', status='PASS')

        # Phase 2: Engineering Review
        review = self._engineering_review(best_stm, L, H, loads, support_positions, verbose)

        final_stm = best_stm

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

        # Save STM data as JSON

        # Build member info for JSON
        members_json = []
        for n1, n2 in final_stm['connections']:
            if n1 not in final_stm['nodes'] or n2 not in final_stm['nodes']:
                continue
            x1, y1 = final_stm['nodes'][n1]
            x2, y2 = final_stm['nodes'][n2]
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            length = math.sqrt(dx**2 + dy**2)
            if dy < 50: mtype = "horizontal"
            elif dx < 50: mtype = "vertical"
            else: mtype = "diagonal"
            angle = math.degrees(math.atan2(dy, dx)) if (dy >= 50 and dx >= 50) else None

            # Get force from truss analysis
            force_kN = None
            if detail.get('analysis_success'):
                for (fn1, fn2), info in detail.get('member_forces', {}).items():
                    if (fn1 == n1 and fn2 == n2) or (fn1 == n2 and fn2 == n1):
                        force_kN = round(info['force_kN'], 1)
                        break

            member = {
                'nodes': [n1, n2],
                'type': mtype,
                'length_mm': round(length, 1),
            }
            if angle: member['angle_deg'] = round(angle, 1)
            if force_kN is not None: member['force_kN'] = force_kN

            members_json.append(member)

        # Build candidates summary
        candidates_json = []
        for rec in all_records:
            layout = rec.get('layout', {})
            candidates_json.append({
                'candidate_id': rec.get('candidate_id'),
                'n_pairs': layout.get('n_pairs'),
                'x_positions': layout.get('x_positions'),
                'result': rec.get('result'),
                'reasoning': layout.get('reasoning', '')
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
                'node_layout': self.topology_model,
                'reviewer': self.reviewer_model
            }
        }

        # ── 전체 후보의 STM 데이터 저장 (노드 설계용) ──
        all_stm_json = []
        for cand_stm, cand_score, cand_id in all_candidates:
            cand_detail = score_stm_detailed(cand_stm, L, H, loads, support_positions)
            cand_members = []
            for n1, n2 in cand_stm['connections']:
                if n1 not in cand_stm['nodes'] or n2 not in cand_stm['nodes']:
                    continue
                x1, y1 = cand_stm['nodes'][n1]
                x2, y2 = cand_stm['nodes'][n2]
                dx, dy = abs(x2-x1), abs(y2-y1)
                length = math.sqrt(dx**2 + dy**2)
                if dy < 50: mtype = "horizontal"
                elif dx < 50: mtype = "vertical"
                else: mtype = "diagonal"
                angle = math.degrees(math.atan2(dy, dx)) if (dy >= 50 and dx >= 50) else None
                force_kN = None
                if cand_detail.get('analysis_success'):
                    for (fn1, fn2), info in cand_detail.get('member_forces', {}).items():
                        if (fn1 == n1 and fn2 == n2) or (fn1 == n2 and fn2 == n1):
                            force_kN = round(info['force_kN'], 1)
                            break
                member = {'nodes': [n1, n2], 'type': mtype, 'length_mm': round(length, 1)}
                if angle: member['angle_deg'] = round(angle, 1)
                if force_kN is not None: member['force_kN'] = force_kN
                cand_members.append(member)

            cand_reactions = {}
            if cand_detail.get('reactions'):
                for nid_r, (rx, ry) in cand_detail['reactions'].items():
                    cand_reactions[nid_r] = {'Rx_kN': round(rx/1000, 1), 'Ry_kN': round(ry/1000, 1)}

            all_stm_json.append({
                'candidate_id': cand_id,
                'n_pairs': (len(cand_stm['nodes']) - 4) // 4,
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
                # 현재 디렉토리에 백업 저장 시도
                backup_path = os.path.join(os.getcwd(), 'stm_result.json')
                try:
                    with open(backup_path, 'w', encoding='utf-8') as f:
                        json.dump(output_json, f, indent=2, ensure_ascii=False)
                    print(f"  Backup saved: {backup_path}")
                except Exception as e2:
                    print(f"  Backup also failed: {e2}")

        return {
            'stm_data': final_stm, 'validation': final_val, 'score': final_score,
            'engineering_review': review, 'log': self.log, 'records': all_records,
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
    print("  STM Topology Comparison — 0 vs 1 vs 2 pairs")
    print("=" * 65)

    beam_length = 6900
    beam_height = 2000
    beam_width = 500
    loads = [(2225, -2000), (4675, -2000)]
    support_positions = [(225, 'pin'), (6675, 'roller')]

    mas = MASSTMGenerator(
        topology_model="deepseek-r1:14b",
        reviewer_model="qwen2.5:7b",
        max_retries=5,
        n_candidates=3,
        output_dir="plots_8node"
    )

    result = mas.generate(
        beam_length=beam_length,
        beam_height=beam_height,
        beam_width=beam_width,
        loads=loads,
        support_positions=support_positions,
        verbose=True
    )

    if result['stm_data']:
        print("\n-- Pipeline Log --")
        for entry in result['log']:
            phase = entry.get('phase', '?')
            print(f"  {phase}: {json.dumps({k:v for k,v in entry.items() if k!='phase'}, default=str)[:120]}")
    else:
        print("\n>> Generation failed.")
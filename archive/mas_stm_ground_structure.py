"""
MAS-based STM Generator — Ground Structure Version
====================================================
3-Agent pipeline with Ground Structure approach for connection generation.

Architecture:
  Phase 1: Node Placer (deepseek-r1:14b) — Node placement via JSON
           Ground Structure (deterministic code) — Enumerate all valid connections
           Connection Selector (qwen2.5-coder:14b) — Select diagonal combination from candidates
  Phase 2: Reviewer Agent (qwen2.5:7b) — Engineering quality assessment with feedback loop
  Phase 3: Optimizer Agent (qwen2.5-coder:14b) — Try different diagonal combination

Key difference from previous versions:
  - No LLM code generation for connections (eliminates code parse errors)
  - Code deterministically generates: horizontals + verticals + all valid diagonal candidates
  - LLM only selects which diagonals to include (JSON response, not code)
  - Truss analysis (direct stiffness method) for scoring: Sum(Ti x Li)

Image outputs:
  01_C{n}_nodes_A{a}_{PASS|FAIL}.png
  02_C{n}_diag_selection.png
  03_candidates_comparison.png
  04_best_selected.png
  05_optimizer_comparison.png
  06_final_result.png
"""

import json
import math
import re
import time
import urllib.request
import os
from copy import deepcopy
import numpy as np


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

NODE_PLACER_PROMPT = """You are the NODE PLACER AGENT for Strut-and-Tie Model (STM) design.

Your ONLY job: Decide where to place nodes for a deep beam STM.
You do NOT decide connections — another agent handles that.

## RULES
1. COORDINATE SYSTEM: Origin (0,0) at bottom-left corner of beam.
2. BOTTOM CHORD NODES: y = 125mm (center of assumed bottom reinforcement).
3. TOP CHORD NODES: y = H - 150mm (center of assumed top compression zone).
4. Support positions → bottom chord nodes (y=125).
5. Load positions → top chord nodes (y=H-150).
6. You may add intermediate nodes at y=125 (bottom) or y=H-150 (top) as specified.

## ANGLE CONSTRAINT (critical for later connection)
The vertical distance between chords is: dy = H - 275 mm.
Any diagonal will span this dy. For the diagonal angle to be 25-65 degrees:
  - Horizontal distance must be between dy*0.466 and dy*2.145.
  - So adjacent top-bottom node pairs should be within this range horizontally.

## OUTPUT FORMAT
You MUST respond with ONLY a valid JSON object. No thinking, no explanation, no markdown.
All coordinates must be plain numbers (NOT arithmetic like 6675-75).
Start your response directly with the opening brace {

{"nodes":{"A":[x,y],"B":[x,y],...},"supports":{"A":"pin","F":"roller"},"node_roles":{"A":"support","B":"load",...}}"""


TOPOLOGY_8NODE = {
    "name": "8-node symmetric with vertical pairs",
    "description": (
        "Place exactly 8 nodes: 4 on top chord, 4 on bottom chord.\n"
        "- 4 FIXED nodes: 2 supports (bottom) + 2 loads (top)\n"
        "- 4 INTERMEDIATE nodes: 2 vertical pairs between each support-load section\n\n"
        "RULE 1 (Vertical pairs): Each intermediate pair shares the same x-coordinate.\n"
        "  - Left pair: (x1, 125) and (x1, H-150)\n"
        "  - Right pair: (x2, 125) and (x2, H-150)\n\n"
        "RULE 2 (Symmetry): If loading is symmetric about beam center (x_mid = L/2),\n"
        "  then x2 = L - x1. You only need to decide x1."
    ),
    "strategy": (
        "Step 1: Place support nodes at bottom (y=125) at given x-positions.\n"
        "Step 2: Place load nodes at top (y=H-150) at given x-positions.\n"
        "Step 3: Place x1 at the SPECIFIED position (given below).\n"
        "Step 4: Calculate x2 = L - x1 (symmetry).\n"
        "Step 5: Place (x1, 125), (x1, H-150), (x2, 125), (x2, H-150).\n"
        "Step 6: Verify angles are within 25-65 degrees."
    )
}


CONNECTION_SELECTOR_PROMPT = """You are the CONNECTION SELECTOR AGENT for Strut-and-Tie Model (STM) design.

The horizontal, vertical, and diagonal CANDIDATE connections have already been computed by code.
Your job: select which diagonal candidates to INCLUDE in the final STM.

## SELECTION CRITERIA (in priority order)
1. EVERY load point node should have at least one diagonal connection
2. EVERY support node should have at least one diagonal connection
3. Prefer symmetric diagonal patterns (if loading is symmetric)
4. Prefer diagonals closer to 45 degrees for efficient force transfer
5. Include enough diagonals so the truss is structurally stable

## RESPONSE FORMAT
Respond with ONLY a valid JSON object:
{
  "selected_indices": [0, 1, 2, 3],
  "reasoning": "brief explanation of selection"
}

The indices refer to the diagonal candidate list provided to you.
Select ALL diagonals that improve the structural model. More is usually better than fewer."""


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


OPTIMIZER_SELECTOR_PROMPT = """You are the OPTIMIZER AGENT for Strut-and-Tie Model (STM) design.

You are given the current diagonal selection and ALL available diagonal candidates.
Your job: suggest a BETTER combination of diagonals that reduces total reinforcement.

## GOALS
1. Reduce the sum of (tension force x length) in tie members
2. Keep structural stability (enough diagonals for load transfer)
3. Maintain symmetry if loading is symmetric
4. Every load and support node should have diagonal connections

## RESPONSE FORMAT
Respond with ONLY a valid JSON object:
{
  "selected_indices": [0, 1, 2, 3],
  "reasoning": "what was changed and why"
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

                # Check angle
                angle = math.degrees(math.atan2(dy, dx))
                if angle < 25 or angle > 65:
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
        if angle < 25 or angle > 65:
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


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

    legend_elements = [
        Line2D([0], [0], color='#2563eb', linewidth=2, label='Horizontal'),
        Line2D([0], [0], color='#dc2626', linewidth=2.5, label='Diagonal'),
        Line2D([0], [0], color='#16a34a', linewidth=2, label='Vertical'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
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
                 selector_model="qwen2.5-coder:14b",
                 reviewer_model="qwen2.5:7b",
                 max_retries=5,
                 n_candidates=3,
                 ollama_url="http://localhost:11434",
                 output_dir="plots"):
        self.client = OllamaClient(ollama_url)
        self.topology_model = topology_model      # Node Placer
        self.selector_model = selector_model      # Connection Selector + Optimizer
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

    # ── Stage 1: Node Placement (LLM) ──
    def _place_nodes(self, L, H, b, loads, supports, assigned_x1,
                     feedback="", prev_output=None):
        dy = H - 275
        sup_left_x, sup_right_x = supports[0][0], supports[1][0]
        load_left_x, load_right_x = loads[0][0], loads[1][0]
        x2 = L - assigned_x1

        user_prompt = (
            f"Place nodes for this deep beam STM:\n\n"
            f"Beam: {L}mm x {H}mm x {b}mm\n"
            f"dy = {dy}mm\n\n"
            f"Load 1: x={loads[0][0]}mm -> top node at ({loads[0][0]}, {H-150})\n"
            f"Load 2: x={loads[1][0]}mm -> top node at ({loads[1][0]}, {H-150})\n"
            f"Pin: x={supports[0][0]}mm -> bottom node at ({supports[0][0]}, 125)\n"
            f"Roller: x={supports[1][0]}mm -> bottom node at ({supports[1][0]}, 125)\n\n"
            f"## TARGET: {TOPOLOGY_8NODE['name']}\n"
            f"Place EXACTLY 8 nodes: 4 top (y={H-150}), 4 bottom (y=125).\n\n"
            f"{TOPOLOGY_8NODE['description']}\n\n"
            f"## ASSIGNED x1 = {assigned_x1}\n"
            f"x2 = {L} - {assigned_x1} = {x2}\n\n"
            f"YOUR 8 NODES:\n"
            f"  Bottom: ({sup_left_x},125), ({assigned_x1},125), ({x2},125), ({sup_right_x},125)\n"
            f"  Top: ({assigned_x1},{H-150}), ({load_left_x},{H-150}), ({load_right_x},{H-150}), ({x2},{H-150})\n\n"
            f"Respond with ONLY a JSON object."
        )
        if prev_output and feedback:
            user_prompt += f"\n\n## ERRORS:\n{feedback}\nFix them."
        elif feedback:
            user_prompt += f"\n\n## FEEDBACK:\n{feedback}"

        response = self.client.chat(
            model=self.topology_model, system=NODE_PLACER_PROMPT,
            user=user_prompt, temperature=0.3
        )
        return parse_json_from_text(response)

    # ── Stage 2: Connection Selection (Ground Structure + LLM) ──
    def _select_connections(self, nodes_data, L, H, loads, support_positions,
                            feedback="", prev_selection=None):
        nodes = nodes_data['nodes']
        supports = nodes_data['supports']

        # Deterministic: enumerate all valid connections
        fixed, candidates = enumerate_ground_structure(nodes, H)

        if not candidates:
            # No diagonal candidates — just use fixed
            return {
                'connections': fixed,
                'fixed': fixed,
                'candidates': candidates,
                'selected_indices': [],
                'design_notes': 'No diagonal candidates available'
            }

        # Build prompt for LLM to select diagonals
        candidate_list = "\n".join(
            f"  [{c['id']}] {c['description']}"
            + (f" (crosses with: {c['crosses_with']})" if c['crosses_with'] else "")
            for c in candidates
        )

        user_prompt = (
            f"Beam: {L}mm x {H}mm\n"
            f"Loads at x={loads[0][0]}mm and x={loads[1][0]}mm\n"
            f"Supports at x={support_positions[0][0]}mm (pin) and x={support_positions[1][0]}mm (roller)\n\n"
            f"FIXED connections (always included): {len(fixed)} members\n"
            f"  Horizontals + Verticals are already placed.\n\n"
            f"DIAGONAL CANDIDATES (select which to include):\n"
            f"{candidate_list}\n\n"
            f"Total candidates: {len(candidates)}\n"
            f"Select by index number. Avoid selecting diagonals that cross each other.\n"
        )

        if prev_selection is not None and feedback:
            user_prompt += (
                f"\n## PREVIOUS SELECTION: {prev_selection}\n"
                f"## FEEDBACK: {feedback}\n"
                f"Improve your selection.\n"
            )
        elif feedback:
            user_prompt += f"\n## FEEDBACK:\n{feedback}\n"

        user_prompt += "\nRespond with ONLY a JSON object."

        response = self.client.chat(
            model=self.selector_model, system=CONNECTION_SELECTOR_PROMPT,
            user=user_prompt, temperature=0.3
        )

        result = parse_json_from_text(response)
        if result is None:
            # Fallback: select all non-crossing candidates
            selected = list(range(len(candidates)))
            return self._build_selection_result(fixed, candidates, selected, nodes, supports,
                                                 "Fallback: all candidates selected")

        selected = result.get('selected_indices', result.get('selected', []))

        # Validate selected indices
        if not isinstance(selected, list):
            selected = list(range(len(candidates)))

        selected = [idx for idx in selected if isinstance(idx, int) and 0 <= idx < len(candidates)]

        # Remove crossing pairs
        selected = self._remove_crossing_selections(candidates, selected)

        return self._build_selection_result(
            fixed, candidates, selected, nodes, supports,
            result.get('reasoning', '')
        )

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

    def _build_selection_result(self, fixed, candidates, selected, nodes, supports, notes):
        connections = list(fixed)
        for idx in selected:
            connections.append(candidates[idx]['nodes'])
        return {
            'connections': connections,
            'fixed': fixed,
            'candidates': candidates,
            'selected_indices': selected,
            'design_notes': notes,
            'nodes': nodes,
            'supports': supports
        }

    # ── Generate one candidate ──
    def _generate_one_candidate(self, L, H, b, loads, support_positions,
                                 candidate_id, assigned_x1, verbose=True):
        cid = candidate_id
        record = {'candidate_id': cid, 'assigned_x1': assigned_x1, 'result': None}

        if verbose:
            print(f"\n  ┌── Candidate {cid} (x1={assigned_x1}) ──┐")

        # ── Stage 1: Place nodes ──
        nodes_data = None
        node_feedback = ""
        prev_node_output = None

        for attempt in range(1, self.max_retries + 1):
            if verbose:
                print(f"  │ [Node Placer] Attempt {attempt}/{self.max_retries}...")

            t0 = time.time()
            nodes_data = self._place_nodes(L, H, b, loads, support_positions,
                                            assigned_x1=assigned_x1,
                                            feedback=node_feedback, prev_output=prev_node_output)
            elapsed = time.time() - t0
            if verbose:
                print(f"  │   Response: {elapsed:.1f}s")

            if nodes_data is None:
                node_feedback = "Response was not valid JSON."
                if verbose: print(f"  │   [FAIL] JSON parse error")
                continue

            nodes = nodes_data.get('nodes', {})
            sup = nodes_data.get('supports', {})
            errs = []

            if len(nodes) != 8:
                errs.append(f"Need 8 nodes, got {len(nodes)}")
            for nid, (x, y) in nodes.items():
                if x < 0 or x > L: errs.append(f"Node {nid}: x={x:.0f} outside beam")
                if y < 0 or y > H: errs.append(f"Node {nid}: y={y:.0f} outside beam")
            if not any(t == 'pin' for t in sup.values()): errs.append("Missing pin")
            if not any(t == 'roller' for t in sup.values()): errs.append("Missing roller")

            if len(nodes) == 8:
                fixed_xs = {round(support_positions[0][0]), round(support_positions[1][0]),
                            round(loads[0][0]), round(loads[1][0])}
                intermediate = [(nid, x, y) for nid, (x, y) in nodes.items() if round(x) not in fixed_xs]
                inter_top = sorted([x for _, x, y in intermediate if y > H/2])
                inter_bot = sorted([x for _, x, y in intermediate if y <= H/2])
                if inter_top != inter_bot:
                    errs.append(f"Intermediate not vertically paired: top={inter_top}, bot={inter_bot}")
                if len(inter_top) == 2:
                    if abs((inter_top[0] + inter_top[1]) - L) > 100:
                        errs.append(f"Not symmetric: x1+x2={inter_top[0]+inter_top[1]:.0f} != L={L}")

            status = 'FAIL' if errs else 'PASS'
            plot_nodes_only(nodes_data, L, H, loads=loads, support_positions=support_positions,
                save_path=self._save_path(f'01_C{cid}_nodes_A{attempt}_{status}.png'),
                title=f'Candidate {cid} - Nodes (Attempt {attempt}) [{status}] x1={assigned_x1}')

            if errs:
                node_feedback = "\n".join(f"- {e}" for e in errs)
                prev_node_output = nodes_data
                if verbose:
                    for e in errs: print(f"  │   Error: {e}")
                continue

            if verbose: print(f"  │   Nodes OK: {len(nodes)} nodes")
            break

        if nodes_data is None or not nodes_data.get('nodes'):
            if verbose: print(f"  │ [FAIL] Node placement failed\n  └──────────────────────────┘")
            record['result'] = 'NODE_FAIL'
            return None, record

        # ── Stage 2: Ground Structure + LLM Selection ──
        if verbose:
            print(f"  │ [Ground Structure] Enumerating connections...")

        fixed, candidates = enumerate_ground_structure(nodes_data['nodes'], H)
        if verbose:
            print(f"  │   Fixed: {len(fixed)} (horizontals + verticals)")
            print(f"  │   Diagonal candidates: {len(candidates)}")
            for c in candidates:
                print(f"  │     [{c['id']}] {c['description']}")

        conn_feedback = ""
        prev_selection = None

        for attempt in range(1, self.max_retries + 1):
            if verbose:
                print(f"  │ [Connection Selector] Attempt {attempt}/{self.max_retries}...")

            t0 = time.time()
            sel_result = self._select_connections(
                nodes_data, L, H, loads, support_positions,
                feedback=conn_feedback, prev_selection=prev_selection
            )
            elapsed = time.time() - t0
            if verbose:
                selected = sel_result['selected_indices']
                print(f"  │   Response: {elapsed:.1f}s")
                print(f"  │   Selected diagonals: {selected} ({len(selected)} of {len(candidates)})")

            stm_candidate = {
                'nodes': nodes_data['nodes'],
                'connections': sel_result['connections'],
                'supports': nodes_data['supports'],
                'design_notes': sel_result['design_notes']
            }

            val_result = code_validate(stm_candidate, L, H,
                                       support_positions=support_positions, loads=loads)

            if not val_result['valid']:
                plot_stm(stm_candidate, L, H, loads=loads, support_positions=support_positions,
                    save_path=self._save_path(f'02_C{cid}_conn_A{attempt}_FAIL.png'),
                    title=f'C{cid} Connection (Attempt {attempt}) [FAIL]',
                    errors=val_result['errors'], status='FAIL')
                error_list = "\n".join(f"- {e}" for e in val_result['errors'])
                conn_feedback = f"ERRORS:\n{error_list}\nSelect more/different diagonals."
                prev_selection = sel_result['selected_indices']
                if verbose:
                    print(f"  │   [Validator] FAIL")
                    for e in val_result['errors'][:5]: print(f"  │     {e}")
                continue

            # PASS
            plot_stm(stm_candidate, L, H, loads=loads, support_positions=support_positions,
                save_path=self._save_path(f'02_C{cid}_conn_A{attempt}_PASS.png'),
                title=f'C{cid} Connection (Attempt {attempt}) [PASS] x1={assigned_x1}',
                status='PASS')
            if verbose:
                print(f"  │   [Validator] PASS")
                for w in val_result['warnings']: print(f"  │     Warning: {w}")
                print(f"  └──────────────────────────┘")
            record['result'] = 'PASS'
            return stm_candidate, record

        if verbose: print(f"  │ [FAIL] Connection selection failed\n  └──────────────────────────┘")
        record['result'] = 'CONN_FAIL'
        return None, record

    # ── Best-of-N ──
    def _best_of_n(self, L, H, b, loads, support_positions, verbose=True):
        dy = H - 275
        x1_values = self._distribute_x1_values(support_positions[0][0], loads[0][0], L, dy, self.n_candidates)
        x1_lo, x1_hi = self.compute_x1_range(support_positions[0][0], loads[0][0], L, dy)

        if verbose:
            print(f"\n{'='*55}")
            print(f"  PHASE 1: Best-of-{self.n_candidates} (Ground Structure)")
            print(f"  Valid x1 range: [{x1_lo:.0f}, {x1_hi:.0f}]")
            print(f"  Assigned x1: {x1_values}")
            print(f"{'='*55}")

        candidates = []
        all_records = []
        for i in range(self.n_candidates):
            candidate, record = self._generate_one_candidate(
                L, H, b, loads, support_positions,
                candidate_id=i+1, assigned_x1=x1_values[i], verbose=verbose)
            all_records.append(record)
            if candidate:
                sc = score_stm(candidate, L, H, loads, support_positions)
                candidates.append((candidate, sc, i+1))
                if verbose:
                    print(f"  -> C{i+1} (x1={x1_values[i]}): {len(candidate['connections'])}m, score={sc:.4f}")
            else:
                if verbose: print(f"  -> C{i+1} (x1={x1_values[i]}): FAILED")

        if not candidates:
            return None, all_records

        candidates.sort(key=lambda x: x[1])
        plot_candidates_comparison(candidates, L, H, loads=loads, support_positions=support_positions,
            save_path=self._save_path('03_candidates_comparison.png'))

        best_stm, best_score, best_cid = candidates[0]
        if verbose:
            print(f"\n  BEST: C{best_cid}, score={best_score:.4f}")
            print(f"  Valid: {len(candidates)}/{self.n_candidates}")

        self.log.append({
            'phase': 'best_of_n', 'x1_values': x1_values,
            'valid': len(candidates), 'total': self.n_candidates,
            'scores': [(cid, s) for _, s, cid in candidates],
            'best_score': best_score, 'best_cid': best_cid
        })
        return best_stm, all_records

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

    # ── Phase 2.5: Retry with Reviewer feedback ──
    def _retry_with_review_feedback(self, stm_data, review, L, H, loads, support_positions, verbose=True):
        issues = review.get('issues', [])
        suggestion = review.get('suggestion', '')
        feedback = "REVIEWER ISSUES:\n" + "\n".join(f"- {i}" for i in issues)
        if suggestion: feedback += f"\nSuggestion: {suggestion}"

        nodes_data = {'nodes': stm_data['nodes'], 'supports': stm_data['supports']}
        old_score = score_stm(stm_data, L, H, loads, support_positions)

        for attempt in range(1, 3):
            if verbose: print(f"  │ [Reviewer -> Selector] Attempt {attempt}/2...")
            t0 = time.time()
            sel_result = self._select_connections(nodes_data, L, H, loads, support_positions,
                                                   feedback=feedback)
            elapsed = time.time() - t0
            if verbose: print(f"  │   Response: {elapsed:.1f}s")

            candidate = {
                'nodes': stm_data['nodes'],
                'connections': sel_result['connections'],
                'supports': stm_data['supports']
            }
            val = code_validate(candidate, L, H, support_positions=support_positions, loads=loads)
            if val['valid']:
                new_score = score_stm(candidate, L, H, loads, support_positions)
                if verbose: print(f"  │   PASS (score: {old_score:.4f} -> {new_score:.4f})")
                if new_score < old_score:
                    return candidate
                else:
                    if verbose: print(f"  │   Score not improved, keeping original")
            else:
                if verbose:
                    print(f"  │   FAIL")
                    for e in val['errors'][:3]: print(f"  │     {e}")
        return None

    # ── Phase 3: Optimizer (re-selection) ──
    def _optimize(self, stm_data, L, H, loads, support_positions, verbose=True):
        if verbose:
            print(f"\n{'='*55}")
            print(f"  PHASE 3: Optimization ({self.selector_model})")
            print(f"{'='*55}")

        before_score = score_stm(stm_data, L, H, loads, support_positions)
        nodes = stm_data['nodes']
        fixed, candidates = enumerate_ground_structure(nodes, H)

        if not candidates:
            if verbose: print(f"  No diagonal candidates to optimize")
            self.log.append({'phase': 'optimizer', 'status': 'NO_CANDIDATES',
                            'before_score': before_score, 'after_score': before_score})
            return stm_data

        after_stm = stm_data
        after_score = before_score
        status = 'NO_OUTPUT'

        candidate_list = "\n".join(f"  [{c['id']}] {c['description']}" for c in candidates)

        for attempt in range(1, 4):
            if verbose: print(f"  [Optimizer] Attempt {attempt}/3...")

            user_prompt = (
                f"Beam: {L}x{H}mm\n"
                f"Current score (lower=better): {before_score:.4f}\n\n"
                f"Current connections: {json.dumps(stm_data['connections'])}\n\n"
                f"ALL diagonal candidates:\n{candidate_list}\n\n"
                f"Select a DIFFERENT combination to reduce total reinforcement.\n"
            )

            t0 = time.time()
            response = self.client.chat(
                model=self.selector_model, system=OPTIMIZER_SELECTOR_PROMPT,
                user=user_prompt, temperature=0.3)
            elapsed = time.time() - t0
            if verbose: print(f"    Response: {elapsed:.1f}s")

            result = parse_json_from_text(response)
            if not result or 'selected_indices' not in result:
                if verbose: print(f"    [FAIL] No valid selection")
                continue

            selected = [idx for idx in result['selected_indices']
                       if isinstance(idx, int) and 0 <= idx < len(candidates)]
            selected = self._remove_crossing_selections(candidates, selected)

            opt_stm = build_stm_from_selection(nodes, stm_data['supports'], fixed, candidates, selected)
            opt_val = code_validate(opt_stm, L, H, support_positions=support_positions, loads=loads)

            if not opt_val['valid']:
                status = 'REJECTED'
                if verbose:
                    print(f"    [FAIL] Validation errors")
                    for e in opt_val['errors'][:3]: print(f"      {e}")
                continue

            new_score = score_stm(opt_stm, L, H, loads, support_positions)
            if new_score < before_score:
                after_stm = opt_stm
                after_score = new_score
                status = 'IMPROVED'
                if verbose: print(f"    Improved! {before_score:.4f} -> {new_score:.4f}")
                break
            else:
                status = 'NO_IMPROVEMENT'
                if verbose: print(f"    No improvement ({before_score:.4f} -> {new_score:.4f})")

        plot_optimizer_comparison(stm_data, after_stm, before_score, after_score,
            L, H, loads=loads, support_positions=support_positions,
            save_path=self._save_path('05_optimizer_comparison.png'))
        self.log.append({'phase': 'optimizer', 'status': status,
                        'before_score': before_score, 'after_score': after_score})
        return after_stm

    # ── Main Pipeline ──
    def generate(self, beam_length, beam_height, beam_width,
                 loads, support_positions, verbose=True):
        L, H, b = beam_length, beam_height, beam_width

        if verbose:
            print("=" * 65)
            print("  MAS-based STM Generation — Ground Structure Version")
            print("=" * 65)
            print(f"  Beam: {L} x {H} x {b} mm  (L/H = {L/H:.2f})")
            print(f"  Loads: {loads}")
            print(f"  Supports: {support_positions}")
            print(f"  Topology: {self.topology_model}")
            print(f"  Selector: {self.selector_model}")
            print(f"  Reviewer: {self.reviewer_model}")
            print(f"  Candidates: {self.n_candidates}, Retries: {self.max_retries}")
            print("=" * 65)

        if not self.client.is_available():
            print("\n[ERROR] Ollama not running!")
            return {'stm_data': None, 'log': self.log}

        import glob
        for old in glob.glob(os.path.join(self.output_dir, '*.png')):
            os.remove(old)

        start_time = time.time()

        # Phase 1
        best_stm, all_records = self._best_of_n(L, H, b, loads, support_positions, verbose)
        if best_stm is None:
            if verbose: print(f"\n[FAIL] No valid candidates.")
            return {'stm_data': None, 'log': self.log, 'records': all_records}

        best_score = score_stm(best_stm, L, H, loads, support_positions)
        plot_stm(best_stm, L, H, loads=loads, support_positions=support_positions,
            save_path=self._save_path('04_best_selected.png'),
            title=f'Best Candidate (score={best_score:.4f})', status='PASS')

        # Phase 2
        review = self._engineering_review(best_stm, L, H, loads, support_positions, verbose)
        if review.get('assessment') == 'QUESTIONABLE' and review.get('issues'):
            if verbose: print(f"\n  [Reviewer -> Selector] Fixing issues...")
            improved = self._retry_with_review_feedback(best_stm, review, L, H, loads, support_positions, verbose)
            if improved:
                best_stm = improved
                if verbose: print(f"  [Reviewer Loop] Improved!")
                self._engineering_review(best_stm, L, H, loads, support_positions, verbose)
            else:
                if verbose: print(f"  [Reviewer Loop] Could not improve, keeping original")

        # Phase 3
        final_stm = self._optimize(best_stm, L, H, loads, support_positions, verbose)

        # Final
        final_val = code_validate(final_stm, L, H, support_positions=support_positions, loads=loads)
        final_score = score_stm(final_stm, L, H, loads, support_positions)
        elapsed = time.time() - start_time

        plot_stm(final_stm, L, H, loads=loads, support_positions=support_positions,
            save_path=self._save_path('06_final_result.png'),
            title=f'FINAL STM (score={final_score:.4f}, time={elapsed:.0f}s)', status='PASS')

        if verbose:
            print(f"\n{'='*65}")
            print("  FINAL RESULT")
            print(f"{'='*65}")
            self._print_stm_summary(final_stm)
            print(f"\n{'='*55}")
            print(f"  Structural Analysis (Direct Stiffness Method)")
            print(f"{'='*55}")
            detail = score_stm_detailed(final_stm, L, H, loads, support_positions)
            print_truss_analysis(detail)
            print(f"\n  Valid: {final_val['valid']}")
            print(f"  Score: {final_score:.4f}")
            print(f"  Time: {elapsed:.1f}s")
            print(f"{'='*65}")

        saved_files = sorted([f for f in os.listdir(self.output_dir) if f.endswith('.png')])
        if verbose:
            print(f"\n  {len(saved_files)} images saved")

        return {
            'stm_data': final_stm, 'validation': final_val, 'score': final_score,
            'engineering_review': review, 'log': self.log, 'records': all_records,
            'elapsed_time': elapsed, 'saved_files': saved_files
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
    print("  MAS STM Generator — Ground Structure Version")
    print("=" * 65)

    beam_length = 6900
    beam_height = 2000
    beam_width = 500
    loads = [(2225, -2000), (4675, -2000)]
    support_positions = [(225, 'pin'), (6675, 'roller')]

    mas = MASSTMGenerator(
        topology_model="deepseek-r1:14b",
        selector_model="qwen2.5-coder:14b",
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

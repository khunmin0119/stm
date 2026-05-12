"""
MAS-based STM Generator v3 — 3-Agent Architecture Restored
============================================================
3-Agent pipeline:
  Phase 1: Topology Agent (deepseek-r1:14b) — Node Placement + Connection Planning
  Phase 2: Reviewer Agent (qwen2.5:7b) — Engineering quality assessment
  Phase 3: Optimizer Agent (deepseek-r1:14b) — Connection improvement via code generation

Improvements over v1:
  - Per-candidate x1 assignment for structural diversity
  - Panel adjacency validation (no skipping intermediate nodes)
  - max_retries=5 for more feedback-correction cycles
  - Improved Reviewer with specific engineering checklist
  - Improved Optimizer using code generation (not raw JSON)

Image outputs:
  01_C{n}_nodes_A{a}_{PASS|FAIL}.png  — Node placement
  02_C{n}_conn_A{a}_{PASS|FAIL}.png   — Connection attempt
  03_candidates_comparison.png          — All PASS candidates side-by-side
  04_best_selected.png                  — Best candidate before optimization
  05_optimizer_comparison.png           — Before vs After optimization
  06_final_result.png                   — Final output
"""

import json
import math
import re
import time
import urllib.request
import os
from copy import deepcopy


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


CONNECTION_PLANNER_PROMPT = """You are the CONNECTION PLANNER AGENT for Strut-and-Tie Model (STM) design.

You are given a set of nodes already placed. Your job: write a Python function that generates the connections.

## WHY CODE?
Instead of listing connections directly, you write a Python function that COMPUTES the connections.
This ensures angle calculations, crossing checks, and symmetry are handled precisely by code.

## HELPER FUNCTIONS (already available — DO NOT redefine them, just CALL them)

```
def calc_angle(nodes, n1, n2):
    # Returns angle in degrees between horizontal and member n1-n2
    # Returns 0 for horizontal, 90 for vertical
    x1, y1 = nodes[n1][0], nodes[n1][1]
    x2, y2 = nodes[n2][0], nodes[n2][1]
    dx, dy = abs(x2 - x1), abs(y2 - y1)
    if dx < 1: return 90.0
    return math.degrees(math.atan2(dy, dx))

def segments_cross(nodes, a, b, c, d):
    # Returns True if member a-b crosses member c-d (ignoring shared endpoints)
    p1, p2 = nodes[a], nodes[b]
    p3, p4 = nodes[c], nodes[d]
    def cp(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    d1, d2 = cp(p3, p4, p1), cp(p3, p4, p2)
    d3, d4 = cp(p1, p2, p3), cp(p1, p2, p4)
    return ((d1>0 and d2<0) or (d1<0 and d2>0)) and ((d3>0 and d4<0) or (d3<0 and d4>0))

def has_crossing(nodes, connections, new_a, new_b):
    # Check if adding member new_a-new_b would cross any existing connection
    for c1, c2 in connections:
        if c1 == new_a or c1 == new_b or c2 == new_a or c2 == new_b:
            continue
        if segments_cross(nodes, new_a, new_b, c1, c2):
            return True
    return False
```

## RULES YOUR CODE MUST ENFORCE (IN THIS ORDER)
1. Top chord: connect adjacent top nodes horizontally (left to right).
2. Bottom chord: connect adjacent bottom nodes horizontally (left to right).
3. VERTICALS (MANDATORY): If a top node and a bottom node share the same x-coordinate,
   they MUST be connected vertically. This is NOT optional. Skip this = FAIL.
4. Diagonals: try connecting top to bottom nodes diagonally.
   - PANEL ADJACENCY RULE: Only connect nodes in ADJACENT panels.
     A diagonal from node A to node B is only valid if there is NO other node
     whose x-coordinate falls between A.x and B.x. Skipping panels = FAIL.
   - Only add if 25 <= calc_angle(nodes, n1, n2) <= 65
   - Only add if not has_crossing(nodes, connections, n1, n2)
   - Prefer diagonals from support nodes to nearest top nodes first.
5. EVERY NODE must have at least 2 connections.
6. Number of MEMBERS must be >= number of NODES.

## PYTHON CODE CONSTRAINTS
- ONLY `import math` is allowed. No other imports.
- Use basic Python: for loops, if statements, list/dict operations.
- The function receives `nodes` dict and THREE helper functions as arguments.
- Function signature: def generate_connections(nodes, calc_angle, segments_cross, has_crossing)

## OUTPUT FORMAT
You MUST respond with ONLY a valid JSON object. No markdown, no explanation.

{"code": "def generate_connections(nodes, calc_angle, segments_cross, has_crossing):\\n    import math\\n    connections = []\\n    # Step 1: separate top and bottom\\n    ...\\n    return connections", "design_notes": "brief explanation"}"""


# ── Reviewer Agent Prompt (improved with specific checklist) ──
ENGINEERING_REVIEWER_PROMPT = """You are an ENGINEERING REVIEWER for Strut-and-Tie Models (STM).

The STM has already passed all structural rule checks by code.
Your job: assess ENGINEERING QUALITY with specific criteria.

## CHECKLIST (score each 1-3, then average)
1. LOAD PATH: Does EVERY load point have at least one diagonal strut going toward a support?
   - 3: All loads have direct diagonal paths to supports
   - 2: Most loads have diagonal paths
   - 1: Some loads rely only on horizontal transfer (no diagonal)

2. FORCE FLOW: Do struts follow natural compression paths (roughly 45° from loads to supports)?
   - 3: Diagonals are 35-55° (near optimal)
   - 2: Diagonals are 25-35° or 55-65° (acceptable but not ideal)
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


# ── Optimizer Agent Prompt (code generation approach) ──
OPTIMIZER_AGENT_PROMPT = """You are the OPTIMIZER AGENT for Strut-and-Tie Model (STM) design.

Given a VALID STM, write a Python function to IMPROVE its connections.
The nodes stay EXACTLY the same — you only change which nodes are connected.

## IMPROVEMENT GOALS (in priority order)
1. Ensure every load point has a diagonal strut toward a support
2. Make diagonal angles closer to 45 degrees
3. Ensure symmetry if loading is symmetric
4. Minimize total member length while keeping structural completeness

## CONSTRAINTS YOUR CODE MUST ENFORCE
- Keep ALL nodes exactly the same (do not add or remove nodes)
- Horizontals: connect adjacent top nodes and adjacent bottom nodes
- Verticals: connect all vertically aligned pairs (same x-coordinate)
- Diagonals: only between adjacent panels (no skipping intermediate nodes)
- Diagonal angles must be 25-65 degrees
- No member crossing
- Every node >= 2 connections
- Members >= nodes

## HELPER FUNCTIONS (passed as arguments, DO NOT redefine)
- calc_angle(nodes, n1, n2) → angle in degrees
- has_crossing(nodes, connections, n1, n2) → True if crossing
- segments_cross(nodes, a, b, c, d) → True if a-b crosses c-d

## FUNCTION SIGNATURE
def optimize_connections(nodes, calc_angle, segments_cross, has_crossing):
    # Return improved connections list
    return connections

## OUTPUT FORMAT
{"code": "def optimize_connections(nodes, calc_angle, segments_cross, has_crossing):\\n    ...", "design_notes": "what was improved"}"""


# ===========================================================
# 3. Deterministic Structural Checks
# ===========================================================

def code_validate(stm_data, beam_length, beam_height, support_positions=None, loads=None):
    """Validate STM against structural rules. Returns dict with 'valid', 'errors', 'warnings'."""
    errors = []
    warnings = []

    nodes = stm_data.get('nodes', {})
    connections = stm_data.get('connections', [])
    supports = stm_data.get('supports', {})

    # ── Basic counts ──
    if len(nodes) < 4:
        errors.append(f"Min 4 nodes required (got {len(nodes)})")
    if len(connections) < len(nodes):
        errors.append(f"Too few members: {len(connections)} for {len(nodes)} nodes (need >= {len(nodes)})")

    # ── Node degree ──
    if nodes and connections:
        degree = {nid: 0 for nid in nodes}
        for n1, n2 in connections:
            if n1 in degree: degree[n1] += 1
            if n2 in degree: degree[n2] += 1
        for nid, d in degree.items():
            if d < 2:
                errors.append(f"Node {nid} has only {d} connection(s) (need >= 2)")

    # ── Bounds ──
    for nid, (x, y) in nodes.items():
        if x < 0 or x > beam_length:
            errors.append(f"Node {nid}: x={x:.0f} outside beam (0~{beam_length:.0f})")
        if y < 0 or y > beam_height:
            errors.append(f"Node {nid}: y={y:.0f} outside beam (0~{beam_height:.0f})")

    # ── Support coordinate verification ──
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

    # ── Vertical member check ──
    if nodes and connections:
        top_nodes = {nid: (x, y) for nid, (x, y) in nodes.items() if y > beam_height / 2}
        bot_nodes = {nid: (x, y) for nid, (x, y) in nodes.items() if y <= beam_height / 2}
        conn_set = {frozenset([n1, n2]) for n1, n2 in connections}

        for t_nid, (t_x, t_y) in top_nodes.items():
            for b_nid, (b_x, b_y) in bot_nodes.items():
                if abs(t_x - b_x) < 50:
                    if frozenset([t_nid, b_nid]) not in conn_set:
                        errors.append(
                            f"Vertical pair {t_nid}({t_x:.0f},{t_y:.0f})-{b_nid}({b_x:.0f},{b_y:.0f}) "
                            f"must be connected (same x={t_x:.0f})"
                        )

    # ── Connection symmetry check ──
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

    # ── Diagonal angle check ──
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
            errors.append(f"Member {n1}-{n2}: angle={angle:.1f}° outside 25-65°")

    # ── Panel adjacency check (no skipping intermediate nodes) ──
    if nodes and connections:
        all_xs = sorted(set(round(x) for _, (x, y) in nodes.items()))
        for n1, n2 in connections:
            if n1 not in nodes or n2 not in nodes:
                continue
            x1, y1 = nodes[n1]
            x2, y2 = nodes[n2]
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            # Only check diagonals (not horizontal or vertical)
            if dy < 50 or dx < 50:
                continue
            # Find nodes whose x falls strictly between x1 and x2
            lo_x, hi_x = min(x1, x2), max(x1, x2)
            between = [nx for nx in all_xs if lo_x + 50 < nx < hi_x - 50]
            if between:
                errors.append(
                    f"Diagonal {n1}-{n2} skips intermediate x={between}: "
                    f"diagonals must connect adjacent panels only"
                )

    # ── Crossing check ──
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

    # ── Connectivity ──
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

    # ── Supports ──
    has_pin = any(t == 'pin' for t in supports.values())
    has_roller = any(t == 'roller' for t in supports.values())
    if not has_pin: errors.append("Missing pin support")
    if not has_roller: errors.append("Missing roller support")

    # ── Determinacy ──
    m = len(connections)
    n = len(nodes)
    r = sum(2 if t == 'pin' else 1 for t in supports.values())
    if m + r < 2 * n:
        warnings.append(f"Underdetermined: m({m})+r({r})={m+r} < 2n={2*n}")

    return {'valid': len(errors) == 0, 'errors': errors, 'warnings': warnings}


def score_stm(stm_data, beam_length, beam_height, loads, support_positions):
    """Score STM: lower = better. Based on total member length + symmetry error."""
    nodes = stm_data['nodes']
    connections = stm_data['connections']
    score = 0.0

    beam_diag = math.sqrt(beam_length**2 + beam_height**2)
    total_length = 0.0
    for n1, n2 in connections:
        if n1 in nodes and n2 in nodes:
            x1, y1 = nodes[n1]
            x2, y2 = nodes[n2]
            total_length += math.sqrt((x2-x1)**2 + (y2-y1)**2)
    score += 0.5 * (total_length / beam_diag)

    if len(loads) == 2:
        mid = beam_length / 2.0
        load_sym = abs((loads[0][0] - mid) + (loads[1][0] - mid))
        if load_sym < 100:
            xs = sorted([nodes[n][0] for n in nodes])
            sym_err = 0.0
            for x in xs:
                mirror = beam_length - x
                closest = min(xs, key=lambda xx: abs(xx - mirror))
                sym_err += abs(closest - mirror)
            score += 0.01 * sym_err

    return score


# ===========================================================
# 4. JSON Parser (Enhanced)
# ===========================================================

def parse_json_from_text(text):
    """Extract JSON from LLM response with multi-strategy fallback."""
    if text is None:
        return None

    # Remove <think> blocks
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

    # Strategy 1: Direct parse
    result = try_parse(text)
    if result: return result

    # Strategy 2: Markdown code block
    try:
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            result = try_parse(match.group(1))
            if result: return result
    except: pass

    # Strategy 3: Brace matching
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

    # Strategy 4: Repair truncated
    try:
        start = text.index('{')
        repaired = repair_truncated_json(preprocess(text[start:]))
        if repaired:
            result = json.loads(repaired)
            if result: return result
    except: pass

    # Strategy 5: Any JSON-like substring
    try:
        matches = list(re.finditer(r'\{[^{}]*\}', preprocess(text)))
        if matches:
            longest = max(matches, key=lambda m: len(m.group(0)))
            result = json.loads(longest.group(0))
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
                tri = plt.Polygon(
                    [(sx-80, -50), (sx+80, -50), (sx, 0)],
                    closed=True, facecolor='#fef3c7', edgecolor='#d97706', linewidth=1.5
                )
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
    """Plot node positions only (no connections)."""
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
        ax.annotate(
            f'{nid}\n({x:.0f},{y:.0f})', (x, y + offset_y),
            fontsize=9, fontweight='bold', ha='center', va='center',
            color='#1e293b',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#dbeafe',
                      edgecolor='#3b82f6', alpha=0.9)
        )

    ax.set_title(title or 'Node Placement', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"  >> Saved: {save_path}")
    plt.close()


def plot_stm(stm_data, beam_length, beam_height,
             loads=None, support_positions=None,
             save_path=None, title=None,
             errors=None, status=None):
    """Plot STM with optional error annotations."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    L, H = beam_length, beam_height
    _draw_beam(ax, L, H, loads, support_positions)

    nodes = stm_data.get('nodes', {})
    connections = stm_data.get('connections', [])
    supports = stm_data.get('supports', {})

    for n1_id, n2_id in connections:
        if n1_id not in nodes or n2_id not in nodes:
            continue
        x1, y1 = nodes[n1_id]
        x2, y2 = nodes[n2_id]
        dx, dy = abs(x2-x1), abs(y2-y1)

        member_has_error = False
        if errors:
            for e in errors:
                if f"{n1_id}-{n2_id}" in e or f"{n2_id}-{n1_id}" in e:
                    member_has_error = True
                    break

        if member_has_error:
            color, lw, ls = '#ef4444', 3.0, '--'
        elif dy < 50:
            color, lw, ls = '#2563eb', 2.0, '-'
        elif dx < 50:
            color, lw, ls = '#16a34a', 2.0, '-'
        else:
            color, lw, ls = '#dc2626', 2.5, '-'

        ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, linestyle=ls, zorder=2)

        mx, my = (x1+x2)/2, (y1+y2)/2
        if dy >= 50 and dx >= 50:
            angle = math.degrees(math.atan2(dy, dx))
            label = f'{n1_id}-{n2_id}\n{angle:.1f}°'
        else:
            label = f'{n1_id}-{n2_id}'
        ax.annotate(label, (mx, my), fontsize=7, ha='center', va='bottom',
                    color='#475569', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor='none', alpha=0.85))

    for nid, (x, y) in nodes.items():
        node_has_error = errors and any(f"Node {nid}" in e for e in errors)

        if node_has_error:
            ax.plot(x, y, 'X', markersize=14, color='#ef4444',
                    markeredgecolor='#991b1b', markeredgewidth=2, zorder=5)
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
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#fef3c7',
                              edgecolor='#f59e0b', alpha=0.9))

    if status:
        badge_color = '#16a34a' if status == 'PASS' else '#ef4444'
        ax.text(0.02, 0.95, status, transform=ax.transAxes,
                fontsize=16, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.4', facecolor=badge_color, alpha=0.9),
                verticalalignment='top', zorder=10)

    if errors:
        error_text = "ERRORS:\n" + "\n".join(f"• {e}" for e in errors[:5])
        if len(errors) > 5:
            error_text += f"\n  ...+{len(errors)-5} more"
        ax.text(0.98, 0.95, error_text, transform=ax.transAxes,
                fontsize=8, color='#991b1b', verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#fef2f2',
                          edgecolor='#fca5a5', alpha=0.95), zorder=10)

    legend_elements = [
        Line2D([0], [0], color='#2563eb', linewidth=2, label='Horizontal'),
        Line2D([0], [0], color='#dc2626', linewidth=2.5, label='Diagonal'),
        Line2D([0], [0], color='#16a34a', linewidth=2, label='Vertical'),
    ]
    if errors:
        legend_elements.append(
            Line2D([0], [0], color='#ef4444', linewidth=3, linestyle='--', label='Error member'))
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    ax.set_title(title or 'STM', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"  >> Saved: {save_path}")
    plt.close()
    return fig


def plot_candidates_comparison(candidates_info, beam_length, beam_height,
                               loads=None, support_positions=None,
                               save_path=None):
    """Plot all PASS candidates side by side."""
    n = len(candidates_info)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(7*n, 6))
    if n == 1:
        axes = [axes]

    for idx, (stm, sc, cid) in enumerate(candidates_info):
        ax = axes[idx]
        _draw_beam(ax, beam_length, beam_height, loads, support_positions)

        nodes = stm['nodes']
        connections = stm['connections']
        supports = stm['supports']

        for n1_id, n2_id in connections:
            if n1_id not in nodes or n2_id not in nodes: continue
            x1, y1 = nodes[n1_id]
            x2, y2 = nodes[n2_id]
            dx, dy = abs(x2-x1), abs(y2-y1)
            if dy < 50: color = '#2563eb'
            elif dx < 50: color = '#16a34a'
            else: color = '#dc2626'
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=2, zorder=2)

        for nid, (x, y) in nodes.items():
            if nid in supports:
                marker = '^' if supports[nid] == 'pin' else 'o'
                ax.plot(x, y, marker=marker, markersize=12, color='#f59e0b',
                        markeredgecolor='#d97706', markeredgewidth=2, zorder=4)
            else:
                ax.plot(x, y, 'o', markersize=8, color='#1e293b', zorder=4)
            offset_y = 60 if y > beam_height / 2 else -80
            ax.annotate(nid, (x, y + offset_y), fontsize=9, fontweight='bold',
                        ha='center', va='center')

        is_best = (idx == 0)
        border_color = '#16a34a' if is_best else '#94a3b8'
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3 if is_best else 1)

        badge = "★ BEST" if is_best else ""
        ax.set_title(
            f'Candidate {cid}  score={sc:.2f}  {badge}\n'
            f'{len(nodes)} nodes, {len(connections)} members',
            fontsize=11, fontweight='bold', pad=10
        )

    fig.suptitle('Candidate Comparison (lower score = better)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"  >> Saved: {save_path}")
    plt.close()


def plot_optimizer_comparison(before_stm, after_stm, before_score, after_score,
                              beam_length, beam_height,
                              loads=None, support_positions=None,
                              save_path=None):
    """Plot before/after optimization side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    L, H = beam_length, beam_height

    for ax, stm, label, sc in [(ax1, before_stm, 'Before Optimization', before_score),
                                 (ax2, after_stm, 'After Optimization', after_score)]:
        _draw_beam(ax, L, H, loads, support_positions)
        nodes = stm['nodes']
        connections = stm['connections']
        supports = stm['supports']

        for n1_id, n2_id in connections:
            if n1_id not in nodes or n2_id not in nodes: continue
            x1, y1 = nodes[n1_id]
            x2, y2 = nodes[n2_id]
            dx, dy = abs(x2-x1), abs(y2-y1)
            if dy < 50: color = '#2563eb'
            elif dx < 50: color = '#16a34a'
            else: color = '#dc2626'
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=2, zorder=2)

            mx, my = (x1+x2)/2, (y1+y2)/2
            if dy >= 50 and dx >= 50:
                angle = math.degrees(math.atan2(dy, dx))
                ax.annotate(f'{angle:.1f}°', (mx, my), fontsize=8, ha='center',
                           color='#475569', fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85))

        for nid, (x, y) in nodes.items():
            if nid in supports:
                marker = '^' if supports[nid] == 'pin' else 'o'
                ax.plot(x, y, marker=marker, markersize=12, color='#f59e0b',
                        markeredgecolor='#d97706', markeredgewidth=2, zorder=4)
            else:
                ax.plot(x, y, 'o', markersize=8, color='#1e293b', zorder=4)
            offset_y = 60 if y > H/2 else -80
            ax.annotate(nid, (x, y + offset_y), fontsize=10, fontweight='bold', ha='center')

        ax.set_title(f'{label}\nscore={sc:.2f} | {len(nodes)} nodes, {len(connections)} members',
                     fontsize=11, fontweight='bold', pad=10)

    if after_score < before_score:
        improvement = ((before_score - after_score) / before_score) * 100
        fig.suptitle(f'Optimizer: score {before_score:.2f} → {after_score:.2f} '
                     f'({improvement:.1f}% improvement)', fontsize=14, fontweight='bold', y=1.02)
    else:
        fig.suptitle('Optimizer: No improvement (original kept)',
                     fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"  >> Saved: {save_path}")
    plt.close()


# ===========================================================
# 6. MAS Controller (Refactored)
# ===========================================================

class MASSTMGenerator:
    """
    3-Agent MAS-based STM generator.
    Pipeline: Phase 1 (Best-of-N) -> Phase 2 (Engineering Review) -> Phase 3 (Optimization)
    Each candidate receives a different x1 value to ensure structural diversity.
    """

    def __init__(self,
                 topology_model="deepseek-r1:14b",
                 reviewer_model="qwen2.5:7b",
                 optimizer_model="deepseek-r1:14b",
                 max_retries=5,
                 n_candidates=3,
                 ollama_url="http://localhost:11434",
                 output_dir="plots"):
        self.client = OllamaClient(ollama_url)
        self.topology_model = topology_model
        self.reviewer_model = reviewer_model
        self.optimizer_model = optimizer_model
        self.max_retries = max_retries
        self.n_candidates = n_candidates
        self.output_dir = output_dir
        self.log = []
        os.makedirs(output_dir, exist_ok=True)

    def _save_path(self, filename):
        return os.path.join(self.output_dir, filename)

    # ── Compute angle-valid x1 range ──
    @staticmethod
    def compute_x1_range(sup_x, load_x, beam_length, dy):
        """Return (x1_lo, x1_hi) ensuring diagonals from support-to-x1 and x1-to-load stay within 25-65 deg."""
        dx_min = dy / math.tan(math.radians(65))
        x1_lo = sup_x + dx_min
        x1_hi = load_x - dx_min
        x1_lo = max(x1_lo, sup_x + 50)
        x1_hi = min(x1_hi, load_x - 50)
        return x1_lo, x1_hi

    # ── Distribute x1 values across candidates ──
    def _distribute_x1_values(self, sup_x, load_x, beam_length, dy, n):
        """Return n distinct x1 values evenly spaced across the valid range (with 5% inset margin)."""
        x1_lo, x1_hi = self.compute_x1_range(sup_x, load_x, beam_length, dy)
        if x1_hi <= x1_lo:
            return [int((sup_x + load_x) / 2)] * n
        # Inset 5% from each end to avoid boundary angle violations from int() truncation
        margin = (x1_hi - x1_lo) * 0.05
        x1_lo_safe = x1_lo + margin
        x1_hi_safe = x1_hi - margin
        if n == 1:
            return [int((x1_lo_safe + x1_hi_safe) / 2)]
        values = []
        for i in range(n):
            t = i / (n - 1)  # 0.0, 0.5, 1.0 for n=3
            values.append(int(x1_lo_safe + t * (x1_hi_safe - x1_lo_safe)))
        return values

    # ── Stage 1: Node Placement (with assigned x1) ──
    def _place_nodes(self, L, H, b, loads, supports, assigned_x1,
                     feedback="", prev_output=None):
        dy = H - 275
        dx_min = dy * 0.466
        dx_max = dy * 2.145

        sup_left_x = supports[0][0]
        sup_right_x = supports[1][0]
        load_left_x = loads[0][0]
        load_right_x = loads[1][0]
        mid_x = L / 2.0
        x2 = L - assigned_x1

        user_prompt = (
            f"Place nodes for this deep beam STM:\n\n"
            f"Beam: {L}mm x {H}mm x {b}mm\n"
            f"Vertical distance between chords: dy = {dy}mm\n"
            f"Min horizontal for diagonal: {dx_min:.0f}mm (65°)\n"
            f"Max horizontal for diagonal: {dx_max:.0f}mm (25°)\n\n"
            f"Load 1: x={loads[0][0]}mm → top chord node at ({loads[0][0]}, {H-150})\n"
            f"Load 2: x={loads[1][0]}mm → top chord node at ({loads[1][0]}, {H-150})\n\n"
            f"Pin support: x={supports[0][0]}mm → bottom chord node at ({supports[0][0]}, 125)\n"
            f"Roller support: x={supports[1][0]}mm → bottom chord node at ({supports[1][0]}, 125)\n\n"
            f"## TARGET TOPOLOGY: {TOPOLOGY_8NODE['name']}\n"
            f"You MUST place EXACTLY 8 nodes: 4 on top chord (y={H-150}) and 4 on bottom chord (y=125).\n\n"
            f"## NODE PLACEMENT GUIDE\n"
            f"{TOPOLOGY_8NODE['description']}\n\n"
            f"## STRATEGY\n"
            f"{TOPOLOGY_8NODE['strategy']}\n\n"
            f"## ASSIGNED x1 VALUE\n"
            f"Use x1 = {assigned_x1} for this candidate.\n"
            f"x2 = {L} - {assigned_x1} = {x2}\n\n"
            f"YOUR 8 NODES:\n"
            f"  Bottom: ({sup_left_x}, 125), ({assigned_x1}, 125), ({x2}, 125), ({sup_right_x}, 125)\n"
            f"  Top:    ({assigned_x1}, {H-150}), ({load_left_x}, {H-150}), ({load_right_x}, {H-150}), ({x2}, {H-150})\n\n"
            f"Respond with ONLY a JSON object."
        )

        if prev_output and feedback:
            user_prompt += (
                f"\n\n## YOUR PREVIOUS ATTEMPT (had errors):\n"
                f"{json.dumps(prev_output, indent=2)}\n\n"
                f"## ERRORS FOUND:\n{feedback}\n\n"
                f"Fix ONLY the errors above. Keep correct parts the same."
            )
        elif feedback:
            user_prompt += f"\n\n## FEEDBACK:\n{feedback}"

        response = self.client.chat(
            model=self.topology_model, system=NODE_PLACER_PROMPT,
            user=user_prompt, temperature=0.3
        )
        return parse_json_from_text(response)

    # ── Stage 2: Connection Planning (via Code Generation) ──
    def _plan_connections(self, nodes_data, L, H,
                          feedback="", prev_output=None):
        nodes = nodes_data.get('nodes', {})

        top_nodes = sorted(
            [(nid, x, y) for nid, (x, y) in nodes.items() if y > H/2],
            key=lambda t: t[1]
        )
        bot_nodes = sorted(
            [(nid, x, y) for nid, (x, y) in nodes.items() if y <= H/2],
            key=lambda t: t[1]
        )

        dy = H - 275
        nodes_str = json.dumps(nodes, indent=2)

        user_prompt = (
            f"Write a Python function to connect these STM nodes:\n\n"
            f"Beam: {L}mm x {H}mm\n"
            f"dy (vertical between chords) = {dy}mm\n\n"
            f"Nodes dict:\n{nodes_str}\n\n"
            f"TOP CHORD (y={H-150}): {', '.join(f'{n}({x:.0f})' for n,x,_ in top_nodes)}\n"
            f"BOTTOM CHORD (y=125): {', '.join(f'{n}({x:.0f})' for n,x,_ in bot_nodes)}\n\n"
        )

        # Identify vertical pairs
        vert_pairs = []
        for t_nid, t_x, t_y in top_nodes:
            for b_nid, b_x, b_y in bot_nodes:
                if abs(t_x - b_x) < 50:
                    vert_pairs.append((t_nid, b_nid, t_x))

        if vert_pairs:
            user_prompt += "VERTICAL PAIRS (same x, MUST be connected):\n"
            for t, b, x in vert_pairs:
                user_prompt += f"  {t} ↔ {b} at x={x:.0f}\n"
            user_prompt += "\n"

        user_prompt += (
            f"FUNCTION SIGNATURE:\n"
            f"  def generate_connections(nodes, calc_angle, segments_cross, has_crossing):\n\n"
            f"THREE HELPER FUNCTIONS are passed as arguments. USE THEM:\n"
            f"  - calc_angle(nodes, n1, n2) → angle in degrees (0=horiz, 90=vert)\n"
            f"  - has_crossing(nodes, connections, n1, n2) → True if adding n1-n2 crosses existing\n"
            f"  - segments_cross(nodes, a, b, c, d) → True if a-b crosses c-d\n\n"
            f"ALGORITHM (IN THIS EXACT ORDER):\n"
            f"1. Separate top (y>{H/2}) and bottom (y<={H/2}), sort by x\n"
            f"2. Connect adjacent top nodes horizontally\n"
            f"3. Connect adjacent bottom nodes horizontally\n"
            f"4. MANDATORY VERTICALS:\n"
        )
        for t, b, x in vert_pairs:
            user_prompt += f"   connections.append(['{t}', '{b}'])  # x={x:.0f}\n"
        user_prompt += (
            f"   Skipping verticals = VALIDATION FAILURE\n"
            f"5. DIAGONALS — one per panel:\n"
            f"   a) Collect all unique x-positions from all nodes, sort them.\n"
            f"   b) Each consecutive pair of x-positions defines a PANEL.\n"
            f"   c) For EACH panel, find the top and bottom nodes at the panel's two x-boundaries.\n"
            f"   d) Try to add ONE diagonal in that panel (top-left↔bottom-right or top-right↔bottom-left).\n"
            f"   e) Only add if 25 <= calc_angle(...) <= 65 and not has_crossing(...).\n"
            f"   f) PANEL ADJACENCY RULE: diagonals must stay within their panel.\n"
            f"      No connecting nodes that skip over intermediate x-positions.\n"
            f"6. Verify every node has >= 2 connections (add more diagonals if needed)\n"
            f"7. Return connections\n\n"
        )

        if prev_output and feedback:
            user_prompt += (
                f"## YOUR PREVIOUS CODE HAD ERRORS:\n{feedback}\n\n"
                f"## YOUR PREVIOUS CODE:\n{prev_output}\n\n"
                f"Fix the errors.\n\n"
            )
        elif feedback:
            user_prompt += f"## FEEDBACK:\n{feedback}\n\n"

        user_prompt += "Respond with ONLY a JSON object containing the code string."

        response = self.client.chat(
            model=self.topology_model, system=CONNECTION_PLANNER_PROMPT,
            user=user_prompt, temperature=0.3
        )

        result = parse_json_from_text(response)
        if result is None:
            return None

        code_str = result.get('code', '')
        if not code_str:
            return None

        connections = self._execute_connection_code(code_str, nodes)
        if connections is None:
            return None

        return {
            'connections': connections,
            'design_notes': result.get('design_notes', ''),
            'generated_code': code_str
        }

    def _execute_connection_code(self, code_str, nodes):
        """Safely execute LLM-generated connection code with helper functions."""
        try:
            code_str = code_str.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"')

            import math as _math

            def _calc_angle(nodes, n1, n2):
                x1, y1 = nodes[n1][0], nodes[n1][1]
                x2, y2 = nodes[n2][0], nodes[n2][1]
                dx, dy = abs(x2 - x1), abs(y2 - y1)
                if dx < 1: return 90.0
                return _math.degrees(_math.atan2(dy, dx))

            def _segments_cross(nodes, a, b, c, d):
                p1, p2, p3, p4 = nodes[a], nodes[b], nodes[c], nodes[d]
                def cp(o, aa, bb):
                    return (aa[0]-o[0])*(bb[1]-o[1]) - (aa[1]-o[1])*(bb[0]-o[0])
                d1, d2 = cp(p3, p4, p1), cp(p3, p4, p2)
                d3, d4 = cp(p1, p2, p3), cp(p1, p2, p4)
                return ((d1>0 and d2<0) or (d1<0 and d2>0)) and \
                       ((d3>0 and d4<0) or (d3<0 and d4>0))

            def _has_crossing(nodes, connections, new_a, new_b):
                for c1, c2 in connections:
                    if c1 == new_a or c1 == new_b or c2 == new_a or c2 == new_b:
                        continue
                    if _segments_cross(nodes, new_a, new_b, c1, c2):
                        return True
                return False

            _allowed = {'math': _math}
            def _safe_import(name, *args, **kwargs):
                if name in _allowed: return _allowed[name]
                raise ImportError(f"Import of '{name}' is not allowed")

            safe_globals = {
                '__builtins__': {
                    '__import__': _safe_import,
                    'range': range, 'len': len, 'abs': abs,
                    'sorted': sorted, 'list': list, 'dict': dict,
                    'set': set, 'tuple': tuple, 'int': int, 'float': float,
                    'str': str, 'bool': bool, 'min': min, 'max': max,
                    'enumerate': enumerate, 'zip': zip, 'round': round,
                    'print': print, 'isinstance': isinstance,
                    'True': True, 'False': False, 'None': None,
                    'map': map, 'filter': filter, 'any': any, 'all': all,
                    'sum': sum, 'reversed': reversed,
                    'frozenset': frozenset, 'hasattr': hasattr, 'getattr': getattr,
                },
                'math': _math
            }
            safe_locals = {}

            exec(code_str, safe_globals, safe_locals)

            gen_func = safe_locals.get('generate_connections') or safe_globals.get('generate_connections')
            if gen_func is None:
                print(f"  │   [CODE ERROR] No 'generate_connections' function found")
                return None

            try:
                connections = gen_func(nodes, _calc_angle, _segments_cross, _has_crossing)
            except TypeError:
                try:
                    connections = gen_func(nodes)
                except Exception as e2:
                    print(f"  │   [CODE ERROR] Function call failed: {e2}")
                    return None

            if not isinstance(connections, list):
                print(f"  │   [CODE ERROR] Function returned {type(connections)}, expected list")
                return None

            cleaned = []
            seen = set()
            for item in connections:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    n1, n2 = str(item[0]), str(item[1])
                    if n1 in nodes and n2 in nodes and n1 != n2:
                        key = tuple(sorted([n1, n2]))
                        if key not in seen:
                            seen.add(key)
                            cleaned.append([n1, n2])

            if not cleaned:
                print(f"  │   [CODE ERROR] No valid connections from function")
                return None

            return cleaned

        except Exception as e:
            print(f"  │   [CODE ERROR] Execution failed: {e}")
            lines = code_str.split('\n')[:5]
            for line in lines:
                print(f"  │     {line}")
            return None

    # ── Generate one candidate ──
    def _generate_one_candidate(self, L, H, b, loads, support_positions,
                                 candidate_id, assigned_x1, verbose=True):
        cid = candidate_id
        record = {'candidate_id': cid, 'assigned_x1': assigned_x1,
                  'node_attempts': [], 'conn_attempts': [], 'result': None}

        if verbose:
            print(f"\n  ┌─ Candidate {cid} (x1={assigned_x1}) ─┐")

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
                node_feedback = "Response was not valid JSON. Return ONLY a JSON object."
                record['node_attempts'].append({'attempt': attempt, 'status': 'PARSE_ERROR', 'time': elapsed})
                if verbose:
                    print(f"  │   [FAIL] JSON parse error")
                continue

            nodes = nodes_data.get('nodes', {})
            supports = nodes_data.get('supports', {})
            node_errors = []

            if len(nodes) < 4:
                node_errors.append(f"Need at least 4 nodes, got {len(nodes)}")
            if len(nodes) != 8:
                node_errors.append(f"Need exactly 8 nodes, got {len(nodes)}")
            for nid, (x, y) in nodes.items():
                if x < 0 or x > L:
                    node_errors.append(f"Node {nid}: x={x:.0f} outside beam")
                if y < 0 or y > H:
                    node_errors.append(f"Node {nid}: y={y:.0f} outside beam")
            if not any(t == 'pin' for t in supports.values()):
                node_errors.append("Missing pin support")
            if not any(t == 'roller' for t in supports.values()):
                node_errors.append("Missing roller support")

            # 8-node validation
            if len(nodes) == 8:
                sup_left_x = support_positions[0][0]
                sup_right_x = support_positions[1][0]
                load_left_x = loads[0][0]
                load_right_x = loads[1][0]

                fixed_xs = {round(sup_left_x), round(sup_right_x),
                            round(load_left_x), round(load_right_x)}
                intermediate = [(nid, x, y) for nid, (x, y) in nodes.items()
                                if round(x) not in fixed_xs]

                for nid, x, y in intermediate:
                    in_left = sup_left_x < x < load_left_x
                    in_right = load_right_x < x < sup_right_x
                    if not (in_left or in_right):
                        node_errors.append(
                            f"Node {nid}: x={x:.0f} not between support-load pair")

                inter_top_xs = sorted([x for _, x, y in intermediate if y > H/2])
                inter_bot_xs = sorted([x for _, x, y in intermediate if y <= H/2])
                if inter_top_xs != inter_bot_xs:
                    node_errors.append(
                        f"Intermediate nodes not vertically paired: "
                        f"top_xs={inter_top_xs}, bot_xs={inter_bot_xs}")

                mid_x = L / 2.0
                if len(inter_top_xs) == 2:
                    x1, x2 = inter_top_xs[0], inter_top_xs[1]
                    sym_error = abs((x1 + x2) - L)
                    if sym_error > 100:
                        node_errors.append(
                            f"Not symmetric: x1={x1:.0f}, x2={x2:.0f}, "
                            f"x1+x2={x1+x2:.0f} ≠ L={L}")

            status = 'FAIL' if node_errors else 'PASS'
            plot_nodes_only(
                nodes_data, L, H, loads=loads, support_positions=support_positions,
                save_path=self._save_path(f'01_C{cid}_nodes_A{attempt}_{status}.png'),
                title=f'Candidate {cid} — Node Placement (Attempt {attempt}) [{status}] x1={assigned_x1}'
            )

            record['node_attempts'].append({
                'attempt': attempt, 'status': status,
                'data': deepcopy(nodes_data), 'errors': node_errors, 'time': elapsed
            })

            if node_errors:
                node_feedback = "\n".join(f"- {e}" for e in node_errors)
                prev_node_output = nodes_data
                if verbose:
                    for e in node_errors:
                        print(f"  │   Error: {e}")
                continue

            if verbose:
                print(f"  │   Nodes OK: {len(nodes)} nodes placed")
            break

        if nodes_data is None or not nodes_data.get('nodes'):
            if verbose:
                print(f"  │ [FAIL] Node placement failed")
                print(f"  └──────────────────────────┘")
            record['result'] = 'NODE_FAIL'
            return None, record

        # ── Stage 2: Plan connections ──
        conn_data = None
        conn_feedback = ""
        prev_conn_output = None

        for attempt in range(1, self.max_retries + 1):
            if verbose:
                print(f"  │ [Connection Planner] Attempt {attempt}/{self.max_retries}...")

            t0 = time.time()
            conn_data = self._plan_connections(nodes_data, L, H,
                                               feedback=conn_feedback, prev_output=prev_conn_output)
            elapsed = time.time() - t0
            if verbose:
                print(f"  │   Response: {elapsed:.1f}s")

            if conn_data is None:
                conn_feedback = (
                    "Your response could not be parsed or the code failed to execute.\n"
                    "RULES:\n"
                    "- Only 'import math' allowed.\n"
                    "- Function: def generate_connections(nodes, calc_angle, segments_cross, has_crossing)\n"
                    "- Return list of [node1_id, node2_id] pairs.\n"
                    "- JSON format: {\"code\": \"def generate_connections(...):\\n    ...\", \"design_notes\": \"...\"}\n"
                )
                record['conn_attempts'].append({'attempt': attempt, 'status': 'CODE_ERROR', 'time': elapsed})
                if verbose:
                    print(f"  │   [FAIL] Code parse/execution error")
                continue

            connections = conn_data.get('connections', [])
            generated_code = conn_data.get('generated_code', '')

            if not connections:
                conn_feedback = "Function returned no valid connections. Fix it."
                prev_conn_output = generated_code
                record['conn_attempts'].append({'attempt': attempt, 'status': 'NO_CONNECTIONS', 'time': elapsed})
                if verbose:
                    print(f"  │   [FAIL] No connections from code")
                continue

            if verbose:
                print(f"  │   Code generated {len(connections)} connections")

            stm_candidate = {
                'nodes': nodes_data['nodes'],
                'connections': connections,
                'supports': nodes_data['supports'],
                'design_notes': conn_data.get('design_notes', '')
            }

            val_result = code_validate(stm_candidate, L, H,
                                       support_positions=support_positions, loads=loads)

            if not val_result['valid']:
                plot_stm(
                    stm_candidate, L, H, loads=loads, support_positions=support_positions,
                    save_path=self._save_path(f'02_C{cid}_conn_A{attempt}_FAIL.png'),
                    title=f'Candidate {cid} — Connection (Attempt {attempt}) [FAIL]',
                    errors=val_result['errors'], status='FAIL'
                )
                error_list = "\n".join(f"- {e}" for e in val_result['errors'])
                conn_feedback = f"ERRORS:\n{error_list}\n\nFix your function."
                prev_conn_output = generated_code
                record['conn_attempts'].append({
                    'attempt': attempt, 'status': 'FAIL',
                    'data': deepcopy(stm_candidate), 'errors': val_result['errors'], 'time': elapsed
                })
                if verbose:
                    print(f"  │   [Validator] FAIL")
                    for e in val_result['errors']:
                        print(f"  │     {e}")
                continue

            # PASS
            plot_stm(
                stm_candidate, L, H, loads=loads, support_positions=support_positions,
                save_path=self._save_path(f'02_C{cid}_conn_A{attempt}_PASS.png'),
                title=f'Candidate {cid} — Connection (Attempt {attempt}) [PASS] x1={assigned_x1}',
                status='PASS'
            )
            record['conn_attempts'].append({
                'attempt': attempt, 'status': 'PASS',
                'data': deepcopy(stm_candidate), 'time': elapsed,
                'generated_code': generated_code
            })
            if verbose:
                print(f"  │   [Validator] PASS ✓")
                for w in val_result['warnings']:
                    print(f"  │     Warning: {w}")
                print(f"  └──────────────────────────┘")
            record['result'] = 'PASS'
            return stm_candidate, record

        if verbose:
            print(f"  │ [FAIL] Connection planning failed")
            print(f"  └──────────────────────────┘")
        record['result'] = 'CONN_FAIL'
        return None, record

    # ── Best-of-N with per-candidate x1 ──
    def _best_of_n(self, L, H, b, loads, support_positions, verbose=True):
        dy = H - 275
        sup_left_x = support_positions[0][0]
        load_left_x = loads[0][0]

        # Distribute x1 values
        x1_values = self._distribute_x1_values(sup_left_x, load_left_x, L, dy, self.n_candidates)

        if verbose:
            x1_lo, x1_hi = self.compute_x1_range(sup_left_x, load_left_x, L, dy)
            print(f"\n{'─' * 55}")
            print(f"  PHASE 1: Best-of-{self.n_candidates} Candidate Generation")
            print(f"  Valid x1 range: [{x1_lo:.0f}, {x1_hi:.0f}]")
            print(f"  Assigned x1 values: {x1_values}")
            print(f"{'─' * 55}")

        candidates = []
        all_records = []

        for i in range(self.n_candidates):
            candidate, record = self._generate_one_candidate(
                L, H, b, loads, support_positions,
                candidate_id=i + 1, assigned_x1=x1_values[i],
                verbose=verbose
            )
            all_records.append(record)

            if candidate is not None:
                sc = score_stm(candidate, L, H, loads, support_positions)
                candidates.append((candidate, sc, i + 1))
                if verbose:
                    print(f"  → Candidate {i+1} (x1={x1_values[i]}): "
                          f"{len(candidate['nodes'])} nodes, "
                          f"{len(candidate['connections'])} members, score={sc:.2f}")
            else:
                if verbose:
                    print(f"  → Candidate {i+1} (x1={x1_values[i]}): FAILED")

        if not candidates:
            return None, all_records

        candidates.sort(key=lambda x: x[1])

        if len(candidates) >= 1:
            plot_candidates_comparison(
                candidates, L, H, loads=loads, support_positions=support_positions,
                save_path=self._save_path('03_candidates_comparison.png')
            )

        best_stm, best_score, best_cid = candidates[0]
        if verbose:
            print(f"\n  ★ Best: C{best_cid}, score={best_score:.2f}")
            print(f"  Valid: {len(candidates)}/{self.n_candidates}")

        self.log.append({
            'phase': 'best_of_n',
            'x1_values': x1_values,
            'total_candidates': self.n_candidates,
            'valid_candidates': len(candidates),
            'scores': [(cid, s) for _, s, cid in candidates],
            'best_score': best_score,
            'best_candidate': best_cid
        })

        return best_stm, all_records

    # ── Phase 2: Engineering Review ──
    def _engineering_review(self, stm_data, L, H, loads, support_positions, verbose=True):
        if verbose:
            print(f"\n{'─' * 55}")
            print(f"  PHASE 2: Engineering Review ({self.reviewer_model})")
            print(f"{'─' * 55}")
            print(f"  [Reviewer Agent] Assessing...")

        t0 = time.time()
        response = self.client.chat(
            model=self.reviewer_model, system=ENGINEERING_REVIEWER_PROMPT,
            user=(
                f"Beam: {L}x{H}mm, L/H={L/H:.2f}\n"
                f"Loads at x={loads[0][0]}mm ({abs(loads[0][1])}kN) "
                f"and x={loads[1][0]}mm ({abs(loads[1][1])}kN)\n"
                f"Supports at x={support_positions[0][0]}mm (pin) "
                f"and x={support_positions[1][0]}mm (roller)\n\n"
                f"STM (passed all code validation):\n{json.dumps(stm_data, indent=2)}\n\n"
                f"Assess using the 4-criteria checklist."
            ),
            temperature=0.1
        )
        elapsed = time.time() - t0
        if verbose:
            print(f"  Response: {elapsed:.1f}s")

        review = parse_json_from_text(response)
        if review:
            if verbose:
                print(f"  Assessment: {review.get('assessment', '?')}")
                print(f"  Total score: {review.get('total_score', '?')}/3.0")
                for key in ['load_path_score', 'force_flow_score', 'symmetry_score', 'completeness_score']:
                    print(f"    {key}: {review.get(key, '?')}/3")
                issues = review.get('issues', [])
                if issues:
                    for iss in issues:
                        print(f"  Issue: {iss}")
            self.log.append({'phase': 'engineering_review', **review})
            return review
        else:
            if verbose:
                print(f"  [WARNING] Could not parse review")
            return {'assessment': 'ACCEPTABLE', 'total_score': 0, 'issues': []}

    # ── Phase 2.5: Retry connections with Reviewer feedback ──
    def _retry_with_review_feedback(self, stm_data, review, L, H, loads, support_positions, verbose=True):
        """Regenerate connections using Reviewer's issues as feedback. Nodes stay the same."""
        issues = review.get('issues', [])
        suggestion = review.get('suggestion', '')

        feedback = "ENGINEERING REVIEWER found these problems:\n"
        for iss in issues:
            feedback += f"- {iss}\n"
        if suggestion:
            feedback += f"\nSuggestion: {suggestion}\n"
        feedback += "\nFix these issues in your connection code."

        nodes_data = {
            'nodes': stm_data['nodes'],
            'supports': stm_data['supports']
        }

        for attempt in range(1, 3):  # 2 attempts to fix
            if verbose:
                print(f"  │ [Reviewer Feedback → Connection Planner] Attempt {attempt}/2...")

            t0 = time.time()
            conn_data = self._plan_connections(nodes_data, L, H,
                                               feedback=feedback, prev_output=None)
            elapsed = time.time() - t0
            if verbose:
                print(f"  │   Response: {elapsed:.1f}s")

            if conn_data is None:
                if verbose:
                    print(f"  │   [FAIL] Code parse/execution error")
                continue

            connections = conn_data.get('connections', [])
            if not connections:
                if verbose:
                    print(f"  │   [FAIL] No connections")
                continue

            candidate = {
                'nodes': stm_data['nodes'],
                'connections': connections,
                'supports': stm_data['supports'],
                'design_notes': conn_data.get('design_notes', '')
            }

            val_result = code_validate(candidate, L, H,
                                       support_positions=support_positions, loads=loads)
            if val_result['valid']:
                new_score = score_stm(candidate, L, H, loads, support_positions)
                old_score = score_stm(stm_data, L, H, loads, support_positions)
                if verbose:
                    print(f"  │   [Validator] PASS ✓ (score: {old_score:.2f} → {new_score:.2f})")
                return candidate
            else:
                if verbose:
                    print(f"  │   [Validator] FAIL")
                    for e in val_result['errors'][:3]:
                        print(f"  │     {e}")
                error_list = "\n".join(f"- {e}" for e in val_result['errors'])
                feedback = f"Previous issues:\n{feedback}\n\nNew errors:\n{error_list}\n\nFix all."

        return None

    # ── Phase 3: Optimization (via code generation with retry) ──
    def _optimize(self, stm_data, L, H, loads, support_positions, verbose=True):
        if verbose:
            print(f"\n{'─' * 55}")
            print(f"  PHASE 3: Optimization ({self.optimizer_model})")
            print(f"{'─' * 55}")

        before_score = score_stm(stm_data, L, H, loads, support_positions)
        nodes = stm_data['nodes']
        after_stm = stm_data
        after_score = before_score
        status = 'NO_OUTPUT'
        opt_feedback = ""

        for attempt in range(1, 4):  # 3 attempts
            if verbose:
                print(f"  [Optimizer Agent] Attempt {attempt}/3...")

            user_prompt = (
                f"Beam: {L}x{H}mm, L/H={L/H:.2f}\n"
                f"Loads at x={loads[0][0]}mm and x={loads[1][0]}mm\n"
                f"Supports at x={support_positions[0][0]}mm (pin) "
                f"and x={support_positions[1][0]}mm (roller)\n\n"
                f"Current STM nodes (DO NOT CHANGE):\n{json.dumps(nodes, indent=2)}\n\n"
                f"Current connections:\n{json.dumps(stm_data['connections'])}\n"
                f"Current supports:\n{json.dumps(stm_data['supports'])}\n\n"
                f"Write optimize_connections() to improve the connections.\n"
                f"The function signature is the same as generate_connections:\n"
                f"  def optimize_connections(nodes, calc_angle, segments_cross, has_crossing):\n"
            )

            if opt_feedback:
                user_prompt += f"\n## PREVIOUS ATTEMPT FAILED:\n{opt_feedback}\n"

            t0 = time.time()
            response = self.client.chat(
                model=self.optimizer_model, system=OPTIMIZER_AGENT_PROMPT,
                user=user_prompt,
                temperature=0.3,
                force_json=True
            )
            elapsed = time.time() - t0
            if verbose:
                print(f"    Response: {elapsed:.1f}s")

            result = parse_json_from_text(response)
            if not result or not result.get('code'):
                opt_feedback = (
                    "Could not parse your response. Return a JSON with:\n"
                    '{"code": "def optimize_connections(nodes, calc_angle, segments_cross, has_crossing):\\n    ...", '
                    '"design_notes": "..."}'
                )
                if verbose:
                    print(f"    [FAIL] No valid code in response")
                continue

            # Execute — rename function to generate_connections for compatibility
            code_str = result['code']
            code_str = code_str.replace('optimize_connections', 'generate_connections')
            opt_connections = self._execute_connection_code(code_str, nodes)

            if not opt_connections:
                opt_feedback = "Your code failed to execute. Only 'import math' is allowed."
                if verbose:
                    print(f"    [FAIL] Code execution error")
                continue

            opt_stm = {
                'nodes': nodes,
                'connections': opt_connections,
                'supports': stm_data['supports'],
                'design_notes': result.get('design_notes', '')
            }
            opt_val = code_validate(opt_stm, L, H,
                                    support_positions=support_positions, loads=loads)

            if not opt_val['valid']:
                error_list = "\n".join(f"- {e}" for e in opt_val['errors'])
                opt_feedback = f"Your code produced invalid connections:\n{error_list}\nFix these errors."
                status = 'REJECTED'
                if verbose:
                    print(f"    [FAIL] Validation errors:")
                    for e in opt_val['errors'][:5]:
                        print(f"      - {e}")
                continue

            after_score = score_stm(opt_stm, L, H, loads, support_positions)
            if after_score < before_score:
                after_stm = opt_stm
                status = 'IMPROVED'
                if verbose:
                    print(f"    Improved! score {before_score:.2f} → {after_score:.2f}")
                break
            else:
                status = 'NO_IMPROVEMENT'
                opt_feedback = (
                    f"Your code produced a valid STM but score={after_score:.2f} "
                    f"is not better than {before_score:.2f}. Try harder to minimize total member length."
                )
                if verbose:
                    print(f"    No improvement ({before_score:.2f} → {after_score:.2f})")

        plot_optimizer_comparison(
            stm_data, after_stm, before_score, after_score,
            L, H, loads=loads, support_positions=support_positions,
            save_path=self._save_path('05_optimizer_comparison.png')
        )

        self.log.append({
            'phase': 'optimizer', 'status': status,
            'before_score': before_score, 'after_score': after_score
        })

        return after_stm

    # ── Main Pipeline ──
    def generate(self, beam_length, beam_height, beam_width,
                 loads, support_positions, verbose=True):
        L, H, b = beam_length, beam_height, beam_width

        if verbose:
            print("=" * 65)
            print("  MAS-based STM Generation v3 (3-Agent Architecture)")
            print("=" * 65)
            print(f"  Beam: {L} x {H} x {b} mm  (L/H = {L/H:.2f})")
            print(f"  Loads: {loads}")
            print(f"  Supports: {support_positions}")
            print(f"  Topology: {self.topology_model}")
            print(f"  Reviewer: {self.reviewer_model}")
            print(f"  Optimizer: {self.optimizer_model}")
            print(f"  Candidates: {self.n_candidates}, Retries: {self.max_retries}")
            print(f"  Output dir: {self.output_dir}")
            print("=" * 65)

        if not self.client.is_available():
            print("\n[ERROR] Ollama server is not running!")
            return {'stm_data': None, 'log': self.log}

        import glob
        for old in glob.glob(os.path.join(self.output_dir, '*.png')):
            os.remove(old)

        start_time = time.time()

        # Phase 1: Best-of-N
        best_stm, all_records = self._best_of_n(L, H, b, loads, support_positions, verbose)

        if best_stm is None:
            if verbose:
                print(f"\n[FAIL] No valid candidates generated.")
            return {'stm_data': None, 'log': self.log, 'records': all_records}

        # Save best selected
        best_score = score_stm(best_stm, L, H, loads, support_positions)
        plot_stm(
            best_stm, L, H, loads=loads, support_positions=support_positions,
            save_path=self._save_path('04_best_selected.png'),
            title=f'★ Best Candidate (score={best_score:.2f})',
            status='PASS'
        )

        # Phase 2: Engineering Review → Feedback Loop
        review = self._engineering_review(best_stm, L, H, loads, support_positions, verbose)

        if review.get('assessment') == 'QUESTIONABLE' and review.get('issues'):
            if verbose:
                print(f"\n  [Reviewer → Connection Planner] Attempting to fix issues...")

            # Try to regenerate connections using reviewer feedback
            improved_stm = self._retry_with_review_feedback(
                best_stm, review, L, H, loads, support_positions, verbose
            )
            if improved_stm:
                best_stm = improved_stm
                best_score = score_stm(best_stm, L, H, loads, support_positions)
                if verbose:
                    print(f"  [Reviewer Loop] Improved STM accepted (score={best_score:.2f})")

                # Re-review the improved STM
                review2 = self._engineering_review(best_stm, L, H, loads, support_positions, verbose)
            else:
                if verbose:
                    print(f"  [Reviewer Loop] Could not improve, keeping original")

        # Phase 3: Optimization
        final_stm = self._optimize(best_stm, L, H, loads, support_positions, verbose)

        # Final
        final_val = code_validate(final_stm, L, H,
                                  support_positions=support_positions, loads=loads)
        final_score = score_stm(final_stm, L, H, loads, support_positions)
        elapsed = time.time() - start_time

        plot_stm(
            final_stm, L, H, loads=loads, support_positions=support_positions,
            save_path=self._save_path('06_final_result.png'),
            title=f'FINAL STM (score={final_score:.2f}, time={elapsed:.0f}s)',
            status='PASS'
        )

        if verbose:
            print(f"\n{'=' * 65}")
            print("  FINAL RESULT")
            print(f"{'=' * 65}")
            self._print_stm_summary(final_stm)
            print(f"\n  Valid: {final_val['valid']}")
            print(f"  Score: {final_score:.2f}")
            print(f"  Total time: {elapsed:.1f}s")
            print(f"  Images saved to: {self.output_dir}/")
            print(f"{'=' * 65}")

        saved_files = sorted([f for f in os.listdir(self.output_dir) if f.endswith('.png')])
        if verbose:
            print(f"\n  Saved {len(saved_files)} images:")
            for f in saved_files:
                print(f"    {f}")

        return {
            'stm_data': final_stm,
            'validation': final_val,
            'score': final_score,
            'engineering_review': review,
            'log': self.log,
            'records': all_records,
            'elapsed_time': elapsed,
            'saved_files': saved_files
        }

    def _print_stm_summary(self, stm):
        nodes = stm['nodes']
        connections = stm['connections']
        supports = stm['supports']

        print(f"\n  Nodes ({len(nodes)}):")
        for nid, (x, y) in sorted(nodes.items()):
            role = f" [{supports[nid]}]" if nid in supports else ""
            print(f"    {nid}: ({x:.1f}, {y:.1f}){role}")

        print(f"\n  Members ({len(connections)}):")
        for n1, n2 in connections:
            if n1 not in nodes or n2 not in nodes:
                print(f"    {n1}-{n2}: INVALID")
                continue
            x1, y1 = nodes[n1]
            x2, y2 = nodes[n2]
            dx, dy = abs(x2-x1), abs(y2-y1)
            length = math.sqrt(dx**2 + dy**2)
            if dy < 50: mtype = "horizontal"
            elif dx < 50: mtype = "vertical"
            else:
                angle = math.degrees(math.atan2(dy, dx))
                mtype = f"diagonal ({angle:.1f}°)"
            print(f"    {n1}-{n2}: {length:.0f}mm ({mtype})")

        if stm.get('design_notes'):
            print(f"\n  Notes: {stm['design_notes']}")


# ===========================================================
# MAIN
# ===========================================================

if __name__ == "__main__":

    print("\n" + "=" * 65)
    print("  MAS STM Generator v3 — 3-Agent Architecture")
    print("=" * 65)

    # ── Beam conditions (KDS Example 10.2) ──
    beam_length = 6900
    beam_height = 2000
    beam_width = 500
    loads = [(2225, -2000), (4675, -2000)]
    support_positions = [(225, 'pin'), (6675, 'roller')]

    mas = MASSTMGenerator(
        topology_model="deepseek-r1:14b",
        reviewer_model="qwen2.5:7b",
        optimizer_model="deepseek-r1:14b",
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
        print("\n── Pipeline Log ──")
        for entry in result['log']:
            phase = entry.get('phase', '?')
            print(f"  {phase}: {json.dumps({k: v for k, v in entry.items() if k != 'phase'}, default=str)[:120]}")
    else:
        print("\n>> Generation failed.")
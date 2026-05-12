"""
ENHANCED Multi-Agent System (MAS) based STM Auto-Generation
=============================================================
Improvements over baseline:
  1. Feedback with previous output (LLM sees what it got wrong)
  2. Best-of-N sampling (generate multiple candidates, pick best)
  3. Two-stage decomposition (Node Placer → Connection Planner)
  4. LLM Validator focuses on engineering reasonableness only
     (structural rule checking handled deterministically by code)

Agents:
  - Node Placer Agent (deepseek-r1:14b): Place STM nodes
  - Connection Planner Agent (deepseek-r1:14b): Connect nodes into truss
  - Code Validator: Deterministic KDS rule checking
  - Engineering Reviewer Agent (qwen2.5:7b): Engineering reasonableness
  - Optimizer Agent (deepseek-r1:14b): Suggest improvements

Pipeline:
  Input → [Node Placer → Connection Planner] × N candidates
        → Code Validation (filter)
        → Scoring (select best)
        → Engineering Review
        → Optimizer → Final Validation → Output
"""

import json
import math
import re
import time
import urllib.request
from copy import deepcopy


# ===========================================================
# 1. Ollama API Client
# ===========================================================

class OllamaClient:
    """Local Ollama API client"""
    
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
    
    def chat(self, model: str, system: str, user: str, temperature: float = 0.3) -> str:
        """Call Ollama chat API"""
        url = f"{self.base_url}/api/chat"
        
        payload = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 4096
            }
        })
        
        req = urllib.request.Request(
            url, 
            data=payload.encode('utf-8'),
            headers={"Content-Type": "application/json"}
        )
        
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                result = json.loads(resp.read().decode('utf-8'))
                return result['message']['content']
        except Exception as e:
            print(f"  [Ollama Error] {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if Ollama server is running"""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except:
            return False


# ===========================================================
# 2. Agent Prompts (Decomposed)
# ===========================================================

# ── Stage 1: Node Placement ──
NODE_PLACER_PROMPT = """You are the NODE PLACER AGENT for Strut-and-Tie Model (STM) design.

Your ONLY job: Decide where to place nodes for a deep beam STM.
You do NOT decide connections — another agent handles that.

## RULES
1. COORDINATE SYSTEM: Origin (0,0) at bottom-left corner of beam.
2. BOTTOM CHORD NODES: y = 125mm (center of assumed bottom reinforcement).
3. TOP CHORD NODES: y = H - 150mm (center of assumed top compression zone).
4. Support positions → bottom chord nodes (y=125).
5. Load positions → top chord nodes (y=H-150).
6. You may add intermediate nodes at y=125 (bottom) or y=H-150 (top) if needed.

## ANGLE CONSTRAINT (critical for later connection)
The vertical distance between chords is: dy = H - 275 mm.
Any diagonal will span this dy. For the diagonal angle to be 25-65 degrees:
  - Horizontal distance must be between dy*0.466 and dy*2.145.
  - So adjacent top-bottom node pairs should be within this range horizontally.

## STRATEGY
- Start with support and load nodes (minimum required).
- If any support-to-load horizontal distance exceeds dy*2.145, add intermediate nodes to break the span.
- If the distance is less than dy*0.466, the angle will be too steep — adjust by placing intermediate nodes.
- Aim for roughly even panel spacing.

## OUTPUT FORMAT
Respond with ONLY a JSON object. No explanation, no markdown.
All coordinates must be plain numbers (NOT arithmetic like 6675-75).

{"nodes":{"A":[x,y],"B":[x,y],...},"supports":{"A":"pin","F":"roller"},"node_roles":{"A":"support","B":"load",...}}"""


# ── Stage 2: Connection Planning ──
CONNECTION_PLANNER_PROMPT = """You are the CONNECTION PLANNER AGENT for Strut-and-Tie Model (STM) design.

You are given a set of nodes already placed. Your ONLY job: decide how to connect them with members.

## RULES
1. DIAGONAL ANGLES: All diagonal members must be 25-65 degrees from horizontal.
2. NO CROSSING: No two members may intersect except at shared endpoint nodes.
3. EVERY NODE must have at least 2 connections.
4. Number of MEMBERS must be >= number of NODES.
5. Top chord: connect adjacent top nodes horizontally (left to right).
6. Bottom chord: connect adjacent bottom nodes horizontally (left to right).
7. Diagonals: connect top nodes to bottom nodes. Use consistent direction to avoid crossing.
8. Load path: every load node must connect (directly or through chain) to a support node.

## AVOIDING MEMBER CROSSING
- Sort nodes left to right.
- For each panel (between two adjacent vertical lines of nodes), use ONE diagonal direction only.
- Never create an X-pattern within a single panel.

## OUTPUT FORMAT
Respond with ONLY a JSON object. No explanation, no markdown.

{"connections":[["A","B"],["B","C"],...], "design_notes":"brief explanation"}"""


# ── Engineering Reviewer (not structural rule checker) ──
ENGINEERING_REVIEWER_PROMPT = """You are an ENGINEERING REVIEWER for Strut-and-Tie Models (STM).

The STM has already passed all structural rule checks (angles, connectivity, no crossing, etc.) by code.
You do NOT need to re-check those rules.

Your job is to assess ENGINEERING REASONABLENESS only:

1. LOAD PATH DIRECTNESS: Do the struts and ties form a reasonable, direct path from loads to supports?
   A good STM has diagonal struts going from load points toward supports, not zigzagging.

2. SYMMETRY: If the loading is symmetric, is the STM roughly symmetric?

3. EFFICIENCY: Are there redundant nodes or members that don't contribute to load transfer?

4. PRACTICAL CONSTRUCTABILITY: Would this truss layout make sense for actual reinforcement placement?

Respond with ONLY a JSON object:
{
  "assessment": "ACCEPTABLE" or "QUESTIONABLE",
  "score": 1-10 (10 = excellent engineering judgment),
  "comments": "brief engineering assessment",
  "suggestion": "if QUESTIONABLE, what specifically should change (or empty string)"
}"""


# ── Optimizer ──
OPTIMIZER_AGENT_PROMPT = """You are the OPTIMIZER AGENT for Strut-and-Tie Model (STM) design.

Given a VALID STM that passes all rules, suggest improvements:
1. Make diagonal angles closer to 45 degrees (optimal force distribution)
2. Improve symmetry if loading is symmetric
3. Reduce total number of members if possible while keeping all constraints
4. Make load paths more direct

CONSTRAINTS you MUST preserve:
- Keep ALL support node positions exactly the same
- Keep ALL load node x-positions the same (y stays at H-150)
- Keep angles between 25-65 degrees
- No member crossing
- Minimum 4 nodes
- Members >= nodes
- Every node must have >= 2 connections
- Do NOT add or remove support or load nodes

Respond with ONLY a JSON object:
{
  "nodes": {"A": [x, y], ...},
  "connections": [["A", "B"], ...],
  "supports": {"A": "pin", ...},
  "design_notes": "what was improved and why"
}

If no improvement is possible, return the original STM unchanged."""


# ===========================================================
# 3. Deterministic Structural Checks
# ===========================================================

def code_validate(stm_data, beam_length, beam_height):
    """Deterministic structural validation — the authoritative checker"""
    errors = []
    warnings = []
    
    nodes = stm_data.get('nodes', {})
    connections = stm_data.get('connections', [])
    supports = stm_data.get('supports', {})
    
    # 1. Min 4 nodes
    if len(nodes) < 4:
        errors.append(f"Min 4 nodes required (got {len(nodes)})")
    
    # 1b. Min members
    if len(connections) < len(nodes):
        errors.append(f"Too few members: {len(connections)} for {len(nodes)} nodes (need >= {len(nodes)})")
    
    # 1c. Every node must have >= 2 connections
    if nodes and connections:
        degree = {nid: 0 for nid in nodes}
        for n1, n2 in connections:
            if n1 in degree:
                degree[n1] += 1
            if n2 in degree:
                degree[n2] += 1
        for nid, d in degree.items():
            if d < 2:
                errors.append(f"Node {nid} has only {d} connection(s) (need >= 2)")
    
    # 2. Node positions within beam
    for nid, (x, y) in nodes.items():
        if x < 0 or x > beam_length:
            errors.append(f"Node {nid}: x={x:.0f} outside beam (0~{beam_length:.0f})")
        if y < 0 or y > beam_height:
            errors.append(f"Node {nid}: y={y:.0f} outside beam (0~{beam_height:.0f})")
    
    # 3. Diagonal angles
    for n1, n2 in connections:
        if n1 not in nodes or n2 not in nodes:
            errors.append(f"Connection {n1}-{n2} references undefined node")
            continue
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        if dy < 50 or dx < 50:  # horizontal or vertical
            continue
        angle = math.degrees(math.atan2(dy, dx))
        if angle < 25 or angle > 65:
            errors.append(f"Member {n1}-{n2}: angle={angle:.1f}° outside 25-65° range")
    
    # 4. Member crossing
    def segments_cross(p1, p2, p3, p4):
        def cp(o, a, b):
            return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
        d1, d2 = cp(p3, p4, p1), cp(p3, p4, p2)
        d3, d4 = cp(p1, p2, p3), cp(p1, p2, p4)
        return ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
               ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0))
    
    for i in range(len(connections)):
        n1, n2 = connections[i]
        if n1 not in nodes or n2 not in nodes:
            continue
        for j in range(i + 1, len(connections)):
            n3, n4 = connections[j]
            if n3 not in nodes or n4 not in nodes:
                continue
            if n1 == n3 or n1 == n4 or n2 == n3 or n2 == n4:
                continue
            if segments_cross(nodes[n1], nodes[n2], nodes[n3], nodes[n4]):
                errors.append(f"Members {n1}-{n2} and {n3}-{n4} cross")
    
    # 5. Connectivity (BFS)
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
    
    # 6. Supports
    has_pin = any(t == 'pin' for t in supports.values())
    has_roller = any(t == 'roller' for t in supports.values())
    if not has_pin:
        errors.append("Missing pin support")
    if not has_roller:
        errors.append("Missing roller support")
    
    # 7. Duplicate connections
    seen = set()
    for n1, n2 in connections:
        key = tuple(sorted([n1, n2]))
        if key in seen:
            warnings.append(f"Duplicate member: {n1}-{n2}")
        seen.add(key)
    
    # 8. Static determinacy check
    m = len(connections)
    n = len(nodes)
    r = sum(2 if t == 'pin' else 1 for t in supports.values())
    if m + r < 2 * n:
        warnings.append(f"Underdetermined: m({m})+r({r})={m+r} < 2n={2*n}")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }


def score_stm(stm_data, beam_length, beam_height, loads, support_positions):
    """Score a valid STM — lower is better.
    Criteria:
      - Diagonal angles closer to 45° → better
      - Symmetry (if loading is symmetric)
      - Fewer total members (simpler model)
      - Total member length (material efficiency)
    """
    nodes = stm_data['nodes']
    connections = stm_data['connections']
    
    score = 0.0
    
    # 1. Angle deviation from 45° (sum of squared deviations)
    angle_penalty = 0.0
    n_diag = 0
    for n1, n2 in connections:
        if n1 not in nodes or n2 not in nodes:
            continue
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        if dy < 50 or dx < 50:
            continue
        angle = math.degrees(math.atan2(dy, dx))
        angle_penalty += (angle - 45.0) ** 2
        n_diag += 1
    
    if n_diag > 0:
        score += (angle_penalty / n_diag)  # avg squared deviation
    
    # 2. Total member length (normalized by beam diagonal)
    beam_diag = math.sqrt(beam_length**2 + beam_height**2)
    total_length = 0.0
    for n1, n2 in connections:
        if n1 in nodes and n2 in nodes:
            x1, y1 = nodes[n1]
            x2, y2 = nodes[n2]
            total_length += math.sqrt((x2-x1)**2 + (y2-y1)**2)
    score += 0.5 * (total_length / beam_diag)
    
    # 3. Symmetry penalty (if loading is symmetric)
    if len(loads) == 2:
        mid = beam_length / 2.0
        load_sym = abs((loads[0][0] - mid) + (loads[1][0] - mid))
        if load_sym < 100:  # approximately symmetric loading
            # Check node symmetry
            xs = sorted([nodes[n][0] for n in nodes])
            sym_err = 0.0
            for x in xs:
                mirror = beam_length - x
                closest = min(xs, key=lambda xx: abs(xx - mirror))
                sym_err += abs(closest - mirror)
            score += 0.01 * sym_err
    
    # 4. Complexity penalty (fewer nodes/members = simpler)
    score += 2.0 * len(connections)
    
    return score


# ===========================================================
# 4. JSON Parser
# ===========================================================

def parse_json_from_text(text):
    """Extract JSON from LLM response (handles markdown, extra text, arithmetic)"""
    if text is None:
        return None
    
    # Remove <think>...</think> blocks (deepseek-r1 reasoning)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    def fix_arithmetic(match):
        expr = match.group(0)
        try:
            result = eval(expr)
            return str(result)
        except:
            return expr
    
    def preprocess_json(json_str):
        fixed = re.sub(r'\d+\s*[\+\-\*/]\s*\d+', fix_arithmetic, json_str)
        return fixed
    
    # Try direct parse
    try:
        return json.loads(text.strip())
    except:
        pass
    
    # Try with arithmetic fix
    try:
        return json.loads(preprocess_json(text.strip()))
    except:
        pass
    
    # Try markdown code block
    try:
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            fixed = preprocess_json(match.group(1))
            return json.loads(fixed)
    except:
        pass
    
    # Try finding JSON object by brace matching
    try:
        start = text.index('{')
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    raw = text[start:i+1]
                    fixed = preprocess_json(raw)
                    return json.loads(fixed)
    except:
        pass
    
    return None


# ===========================================================
# 5. Enhanced MAS Controller
# ===========================================================

class EnhancedMASSTMGenerator:
    """Enhanced Multi-Agent System for STM Generation
    
    Improvements:
      1. Two-stage decomposition: Node Placer → Connection Planner
      2. Best-of-N sampling: generate N candidates, pick best
      3. Feedback includes previous output
      4. LLM reviewer only checks engineering reasonableness
    """
    
    def __init__(self,
                 topology_model="deepseek-r1:14b",
                 reviewer_model="qwen2.5:7b",
                 optimizer_model="deepseek-r1:14b",
                 max_retries=3,
                 n_candidates=3,
                 ollama_url="http://localhost:11434"):
        
        self.client = OllamaClient(ollama_url)
        self.topology_model = topology_model
        self.reviewer_model = reviewer_model
        self.optimizer_model = optimizer_model
        self.max_retries = max_retries
        self.n_candidates = n_candidates
        self.log = []
    
    # ─────────────────────────────────────────────
    # Stage 1: Node Placement
    # ─────────────────────────────────────────────
    def _place_nodes(self, L, H, b, loads, supports, fck, fy,
                     feedback="", prev_output=None, verbose=True):
        """Ask Node Placer Agent to determine node positions"""
        
        user_prompt = (
            f"Place nodes for this deep beam STM:\n\n"
            f"Beam: {L}mm x {H}mm x {b}mm\n"
            f"Vertical distance between chords: dy = {H} - 275 = {H - 275}mm\n"
            f"Min horizontal for diagonal: {(H-275)*0.466:.0f}mm (65°)\n"
            f"Max horizontal for diagonal: {(H-275)*2.145:.0f}mm (25°)\n"
            f"Optimal horizontal (45°): {H-275}mm\n\n"
            f"Load 1: x={loads[0][0]}mm → top chord node at ({loads[0][0]}, {H-150})\n"
            f"Load 2: x={loads[1][0]}mm → top chord node at ({loads[1][0]}, {H-150})\n\n"
            f"Pin support: x={supports[0][0]}mm → bottom chord node at ({supports[0][0]}, 125)\n"
            f"Roller support: x={supports[1][0]}mm → bottom chord node at ({supports[1][0]}, 125)\n\n"
            f"These 4 nodes are MANDATORY. Add intermediate nodes ONLY if needed "
            f"to keep future diagonal angles in 25-65° range.\n\n"
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
            model=self.topology_model,
            system=NODE_PLACER_PROMPT,
            user=user_prompt,
            temperature=0.3
        )
        
        return parse_json_from_text(response)
    
    # ─────────────────────────────────────────────
    # Stage 2: Connection Planning
    # ─────────────────────────────────────────────
    def _plan_connections(self, nodes_data, L, H,
                          feedback="", prev_output=None, verbose=True):
        """Ask Connection Planner Agent to connect the nodes"""
        
        nodes = nodes_data.get('nodes', {})
        supports = nodes_data.get('supports', {})
        
        # Sort nodes for clarity
        top_nodes = []
        bot_nodes = []
        for nid, (x, y) in nodes.items():
            if y > H / 2:
                top_nodes.append((nid, x, y))
            else:
                bot_nodes.append((nid, x, y))
        
        top_nodes.sort(key=lambda t: t[1])
        bot_nodes.sort(key=lambda t: t[1])
        
        dy = H - 275
        
        user_prompt = (
            f"Connect these nodes into a valid STM truss:\n\n"
            f"Beam: {L}mm x {H}mm\n"
            f"Vertical distance between chords: dy = {dy}mm\n\n"
            f"TOP CHORD NODES (y={H-150}, left to right):\n"
        )
        for nid, x, y in top_nodes:
            user_prompt += f"  {nid}: ({x}, {y})\n"
        
        user_prompt += f"\nBOTTOM CHORD NODES (y=125, left to right):\n"
        for nid, x, y in bot_nodes:
            role = f" [{supports.get(nid, '')}]" if nid in supports else ""
            user_prompt += f"  {nid}: ({x}, {y}){role}\n"
        
        user_prompt += (
            f"\nHorizontal distances between adjacent top-bottom pairs:\n"
        )
        for t_nid, t_x, _ in top_nodes:
            for b_nid, b_x, _ in bot_nodes:
                dx = abs(t_x - b_x)
                angle = math.degrees(math.atan2(dy, dx)) if dx > 0 else 90.0
                if 15 <= angle <= 75:  # show plausible connections
                    user_prompt += f"  {t_nid}↔{b_nid}: dx={dx:.0f}mm → angle={angle:.1f}°"
                    if 25 <= angle <= 65:
                        user_prompt += " ✓ valid"
                    else:
                        user_prompt += " ✗ invalid"
                    user_prompt += "\n"
        
        user_prompt += (
            f"\nRules:\n"
            f"- Connect top chord nodes horizontally (left→right)\n"
            f"- Connect bottom chord nodes horizontally (left→right)\n"
            f"- Add diagonals (25-65° only). Use ONE direction per panel.\n"
            f"- Every node needs >= 2 connections\n"
            f"- Members >= nodes (= {len(nodes)})\n"
            f"- No crossing members\n\n"
            f"Respond with ONLY a JSON object."
        )
        
        if prev_output and feedback:
            user_prompt += (
                f"\n\n## YOUR PREVIOUS ATTEMPT (had errors):\n"
                f"{json.dumps(prev_output, indent=2)}\n\n"
                f"## ERRORS FOUND:\n{feedback}\n\n"
                f"Fix ONLY the errors. Keep correct connections."
            )
        elif feedback:
            user_prompt += f"\n\n## FEEDBACK:\n{feedback}"
        
        response = self.client.chat(
            model=self.topology_model,
            system=CONNECTION_PLANNER_PROMPT,
            user=user_prompt,
            temperature=0.3
        )
        
        return parse_json_from_text(response)
    
    # ─────────────────────────────────────────────
    # Generate a single STM candidate (2-stage)
    # ─────────────────────────────────────────────
    def _generate_one_candidate(self, L, H, b, loads, support_positions,
                                 fck, fy, candidate_id, temperature,
                                 verbose=True):
        """Generate one STM candidate using 2-stage decomposition with retries"""
        
        if verbose:
            print(f"\n  ┌─ Candidate {candidate_id} (temp={temperature:.2f}) ─┐")
        
        # ── Stage 1: Place nodes (with retries) ──
        nodes_data = None
        node_feedback = ""
        prev_node_output = None
        
        for attempt in range(1, self.max_retries + 1):
            if verbose:
                print(f"  │ [Node Placer] Attempt {attempt}/{self.max_retries}...")
            
            t0 = time.time()
            nodes_data = self._place_nodes(
                L, H, b, loads, support_positions, fck, fy,
                feedback=node_feedback,
                prev_output=prev_node_output,
                verbose=verbose
            )
            t1 = time.time()
            
            if verbose:
                print(f"  │   Response: {t1-t0:.1f}s")
            
            if nodes_data is None:
                node_feedback = "Response was not valid JSON. Return ONLY a JSON object."
                if verbose:
                    print(f"  │   [FAIL] JSON parse error")
                continue
            
            # Basic checks on nodes
            nodes = nodes_data.get('nodes', {})
            supports = nodes_data.get('supports', {})
            node_errors = []
            
            if len(nodes) < 4:
                node_errors.append(f"Need at least 4 nodes, got {len(nodes)}")
            
            for nid, (x, y) in nodes.items():
                if x < 0 or x > L:
                    node_errors.append(f"Node {nid}: x={x:.0f} outside beam")
                if y < 0 or y > H:
                    node_errors.append(f"Node {nid}: y={y:.0f} outside beam")
            
            if not any(t == 'pin' for t in supports.values()):
                node_errors.append("Missing pin support")
            if not any(t == 'roller' for t in supports.values()):
                node_errors.append("Missing roller support")
            
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
                print(f"  └────────────────────────┘")
            return None
        
        # ── Stage 2: Plan connections (with retries) ──
        conn_data = None
        conn_feedback = ""
        prev_conn_output = None
        
        for attempt in range(1, self.max_retries + 1):
            if verbose:
                print(f"  │ [Connection Planner] Attempt {attempt}/{self.max_retries}...")
            
            t0 = time.time()
            conn_data = self._plan_connections(
                nodes_data, L, H,
                feedback=conn_feedback,
                prev_output=prev_conn_output,
                verbose=verbose
            )
            t1 = time.time()
            
            if verbose:
                print(f"  │   Response: {t1-t0:.1f}s")
            
            if conn_data is None:
                conn_feedback = "Response was not valid JSON. Return ONLY a JSON object."
                if verbose:
                    print(f"  │   [FAIL] JSON parse error")
                continue
            
            connections = conn_data.get('connections', [])
            if not connections:
                conn_feedback = "No connections provided. You must return a 'connections' list."
                prev_conn_output = conn_data
                if verbose:
                    print(f"  │   [FAIL] No connections")
                continue
            
            # Assemble full STM
            stm_candidate = {
                'nodes': nodes_data['nodes'],
                'connections': connections,
                'supports': nodes_data['supports'],
                'design_notes': conn_data.get('design_notes', '')
            }
            
            # Code validation
            val_result = code_validate(stm_candidate, L, H)
            
            if not val_result['valid']:
                conn_feedback = "\n".join(f"- {e}" for e in val_result['errors'])
                prev_conn_output = conn_data
                if verbose:
                    print(f"  │   [Code Validator] FAIL")
                    for e in val_result['errors']:
                        print(f"  │     {e}")
                continue
            
            if verbose:
                print(f"  │   [Code Validator] PASS ✓")
                for w in val_result['warnings']:
                    print(f"  │     Warning: {w}")
                print(f"  └────────────────────────┘")
            
            return stm_candidate
        
        if verbose:
            print(f"  │ [FAIL] Connection planning failed")
            print(f"  └────────────────────────┘")
        return None
    
    # ─────────────────────────────────────────────
    # Best-of-N selection
    # ─────────────────────────────────────────────
    def _best_of_n(self, L, H, b, loads, support_positions, fck, fy, verbose=True):
        """Generate N candidates and pick the best one"""
        
        if verbose:
            print(f"\n{'─' * 50}")
            print(f"  PHASE 1: Best-of-{self.n_candidates} Candidate Generation")
            print(f"{'─' * 50}")
        
        candidates = []
        temperatures = [0.3 + 0.15 * i for i in range(self.n_candidates)]
        
        for i in range(self.n_candidates):
            temp = temperatures[i]
            candidate = self._generate_one_candidate(
                L, H, b, loads, support_positions, fck, fy,
                candidate_id=i + 1,
                temperature=temp,
                verbose=verbose
            )
            
            if candidate is not None:
                sc = score_stm(candidate, L, H, loads, support_positions)
                candidates.append((candidate, sc))
                
                if verbose:
                    n_nodes = len(candidate['nodes'])
                    n_members = len(candidate['connections'])
                    print(f"  → Candidate {i+1}: {n_nodes} nodes, {n_members} members, score={sc:.1f}")
            else:
                if verbose:
                    print(f"  → Candidate {i+1}: FAILED")
        
        if not candidates:
            return None
        
        # Sort by score (lower is better)
        candidates.sort(key=lambda x: x[1])
        
        best_stm, best_score = candidates[0]
        
        if verbose:
            print(f"\n  ★ Best candidate: score={best_score:.1f} "
                  f"({len(best_stm['nodes'])} nodes, {len(best_stm['connections'])} members)")
            print(f"  Candidates generated: {len(candidates)}/{self.n_candidates}")
        
        self.log.append({
            'phase': 'best_of_n',
            'total_candidates': self.n_candidates,
            'valid_candidates': len(candidates),
            'scores': [s for _, s in candidates],
            'best_score': best_score
        })
        
        return best_stm
    
    # ─────────────────────────────────────────────
    # Engineering Review (LLM — reasonableness only)
    # ─────────────────────────────────────────────
    def _engineering_review(self, stm_data, L, H, loads, support_positions, verbose=True):
        """LLM reviews engineering reasonableness (NOT structural rules)"""
        
        if verbose:
            print(f"\n{'─' * 50}")
            print(f"  PHASE 2: Engineering Review")
            print(f"{'─' * 50}")
            print(f"  [Engineering Reviewer] Assessing reasonableness...")
        
        t0 = time.time()
        response = self.client.chat(
            model=self.reviewer_model,
            system=ENGINEERING_REVIEWER_PROMPT,
            user=(
                f"Beam: {L}x{H}mm, L/H={L/H:.2f}\n"
                f"Loads at x={loads[0][0]}mm ({abs(loads[0][1])}kN) "
                f"and x={loads[1][0]}mm ({abs(loads[1][1])}kN)\n"
                f"Supports at x={support_positions[0][0]}mm (pin) "
                f"and x={support_positions[1][0]}mm (roller)\n\n"
                f"STM (already passed all structural rule checks):\n"
                f"{json.dumps(stm_data, indent=2)}\n\n"
                f"Assess ONLY engineering reasonableness."
            ),
            temperature=0.1
        )
        t1 = time.time()
        
        if verbose:
            print(f"  Response: {t1-t0:.1f}s")
        
        review = parse_json_from_text(response)
        
        if review:
            assessment = review.get('assessment', 'UNKNOWN')
            eng_score = review.get('score', 0)
            comments = review.get('comments', '')
            
            if verbose:
                print(f"  Assessment: {assessment}")
                print(f"  Engineering score: {eng_score}/10")
                if comments:
                    print(f"  Comments: {comments}")
            
            self.log.append({
                'phase': 'engineering_review',
                'assessment': assessment,
                'score': eng_score,
                'comments': comments
            })
            
            return review
        else:
            if verbose:
                print(f"  [WARNING] Could not parse review response")
            return {'assessment': 'ACCEPTABLE', 'score': 5, 'comments': 'Review parse failed'}
    
    # ─────────────────────────────────────────────
    # Optimization
    # ─────────────────────────────────────────────
    def _optimize(self, stm_data, L, H, loads, support_positions, verbose=True):
        """Optimizer Agent suggests improvements"""
        
        if verbose:
            print(f"\n{'─' * 50}")
            print(f"  PHASE 3: Optimization")
            print(f"{'─' * 50}")
            print(f"  [Optimizer Agent] Analyzing for improvements...")
        
        t0 = time.time()
        response = self.client.chat(
            model=self.optimizer_model,
            system=OPTIMIZER_AGENT_PROMPT,
            user=(
                f"Beam: {L}x{H}mm, L/H={L/H:.2f}\n"
                f"Loads at x={loads[0][0]}mm and x={loads[1][0]}mm\n"
                f"Supports at x={support_positions[0][0]}mm (pin) "
                f"and x={support_positions[1][0]}mm (roller)\n\n"
                f"Current valid STM:\n{json.dumps(stm_data, indent=2)}\n\n"
                f"Suggest improvements while keeping all constraints."
            ),
            temperature=0.3
        )
        t1 = time.time()
        
        if verbose:
            print(f"  Response: {t1-t0:.1f}s")
        
        opt_stm = parse_json_from_text(response)
        
        if opt_stm and opt_stm.get('nodes') and opt_stm.get('connections'):
            opt_val = code_validate(opt_stm, L, H)
            
            if opt_val['valid']:
                old_score = score_stm(stm_data, L, H, loads, support_positions)
                new_score = score_stm(opt_stm, L, H, loads, support_positions)
                
                if new_score < old_score:
                    if verbose:
                        print(f"  [Optimizer] Improved! score {old_score:.1f} → {new_score:.1f}")
                        if opt_stm.get('design_notes'):
                            print(f"  Notes: {opt_stm['design_notes']}")
                    
                    self.log.append({
                        'phase': 'optimizer',
                        'status': 'IMPROVED',
                        'old_score': old_score,
                        'new_score': new_score
                    })
                    return opt_stm
                else:
                    if verbose:
                        print(f"  [Optimizer] No improvement (score {old_score:.1f} → {new_score:.1f}), keeping original")
                    self.log.append({
                        'phase': 'optimizer',
                        'status': 'NO_IMPROVEMENT',
                        'old_score': old_score,
                        'new_score': new_score
                    })
                    return stm_data
            else:
                if verbose:
                    print(f"  [Optimizer] Optimized STM has errors, keeping original")
                    for e in opt_val['errors']:
                        print(f"    - {e}")
                self.log.append({'phase': 'optimizer', 'status': 'REJECTED', 'errors': opt_val['errors']})
                return stm_data
        else:
            if verbose:
                print(f"  [Optimizer] No valid output, keeping original")
            self.log.append({'phase': 'optimizer', 'status': 'NO_OUTPUT'})
            return stm_data
    
    # ─────────────────────────────────────────────
    # Main Pipeline
    # ─────────────────────────────────────────────
    def generate(self, beam_length, beam_height, beam_width,
                 loads, support_positions,
                 fck=27.0, fy=400.0, verbose=True):
        """
        Full enhanced MAS pipeline.
        
        Pipeline:
          1. Best-of-N: generate N candidates via 2-stage decomposition
          2. Code validation filters invalid candidates
          3. Score-based selection picks best
          4. Engineering review (LLM) checks reasonableness
          5. Optimizer tries to improve
          6. Final validation
        """
        
        L, H, b = beam_length, beam_height, beam_width
        
        if verbose:
            print("=" * 65)
            print("  Enhanced MAS-based STM Generation")
            print("=" * 65)
            print(f"  Beam: {L} x {H} x {b} mm  (L/H = {L/H:.2f})")
            print(f"  Loads: {loads}")
            print(f"  Supports: {support_positions}")
            print(f"  Topology model: {self.topology_model}")
            print(f"  Reviewer model: {self.reviewer_model}")
            print(f"  Optimizer model: {self.optimizer_model}")
            print(f"  Candidates: {self.n_candidates}, Max retries: {self.max_retries}")
            print("=" * 65)
        
        # Check Ollama
        if not self.client.is_available():
            print("\n[ERROR] Ollama server is not running!")
            print("  Start it with: ollama serve")
            return {'stm_data': None, 'log': self.log}
        
        start_time = time.time()
        
        # ── Phase 1: Best-of-N candidate generation ──
        best_stm = self._best_of_n(L, H, b, loads, support_positions, fck, fy, verbose)
        
        if best_stm is None:
            if verbose:
                print(f"\n[FAIL] No valid candidates generated.")
            return {'stm_data': None, 'log': self.log}
        
        # ── Phase 2: Engineering review ──
        review = self._engineering_review(best_stm, L, H, loads, support_positions, verbose)
        
        # ── Phase 3: Optimization ──
        final_stm = self._optimize(best_stm, L, H, loads, support_positions, verbose)
        
        # ── Final validation ──
        final_val = code_validate(final_stm, L, H)
        final_score = score_stm(final_stm, L, H, loads, support_positions)
        
        elapsed = time.time() - start_time
        
        if verbose:
            print(f"\n{'=' * 65}")
            print("  FINAL RESULT")
            print(f"{'=' * 65}")
            self._print_stm_summary(final_stm)
            print(f"\n  Valid: {final_val['valid']}")
            print(f"  Score: {final_score:.1f}")
            print(f"  Total time: {elapsed:.1f}s")
            print(f"{'=' * 65}")
        
        return {
            'stm_data': final_stm,
            'validation': final_val,
            'score': final_score,
            'engineering_review': review,
            'log': self.log,
            'elapsed_time': elapsed
        }
    
    def _print_stm_summary(self, stm):
        """Print STM details"""
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
                print(f"    {n1}-{n2}: INVALID (undefined node)")
                continue
            x1, y1 = nodes[n1]
            x2, y2 = nodes[n2]
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            length = math.sqrt(dx**2 + dy**2)
            
            if dy < 50:
                mtype = "horizontal"
            elif dx < 50:
                mtype = "vertical"
            else:
                angle = math.degrees(math.atan2(dy, dx))
                mtype = f"diagonal ({angle:.1f}°)"
            
            print(f"    {n1}-{n2}: {length:.0f}mm ({mtype})")
        
        if stm.get('design_notes'):
            print(f"\n  Notes: {stm['design_notes']}")


# ===========================================================
# 6. Visualization
# ===========================================================

def plot_stm(stm_data, beam_length, beam_height,
             loads=None, support_positions=None,
             save_path=None, title=None):
    """Plot STM result"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    
    L, H = beam_length, beam_height
    
    # Beam outline
    beam_rect = patches.Rectangle(
        (0, 0), L, H,
        linewidth=1.5, edgecolor='#94a3b8',
        facecolor='#f1f5f9', alpha=0.3
    )
    ax.add_patch(beam_rect)
    
    nodes = stm_data['nodes']
    connections = stm_data['connections']
    supports = stm_data['supports']
    
    # Members
    for n1_id, n2_id in connections:
        if n1_id not in nodes or n2_id not in nodes:
            continue
        x1, y1 = nodes[n1_id]
        x2, y2 = nodes[n2_id]
        
        # Color by type
        dx, dy = abs(x2-x1), abs(y2-y1)
        if dy < 50:
            color, lw = '#2563eb', 2.0  # horizontal = blue
        elif dx < 50:
            color, lw = '#16a34a', 2.0  # vertical = green
        else:
            color, lw = '#dc2626', 2.5  # diagonal = red
        
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, zorder=2)
        
        # Label
        mx, my = (x1+x2)/2, (y1+y2)/2
        if dy >= 50 and dx >= 50:
            angle = math.degrees(math.atan2(dy, dx))
            label = f'{n1_id}-{n2_id}\n{angle:.1f}°'
        else:
            label = f'{n1_id}-{n2_id}'
        ax.annotate(
            label, (mx, my), fontsize=7,
            ha='center', va='bottom', color='#475569', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor='none', alpha=0.85)
        )
    
    # Nodes
    for nid, (x, y) in nodes.items():
        if nid in supports:
            marker = '^' if supports[nid] == 'pin' else 'o'
            ax.plot(x, y, marker=marker, markersize=14, color='#f59e0b',
                    markeredgecolor='#d97706', markeredgewidth=2, zorder=4)
        else:
            ax.plot(x, y, 'o', markersize=10, color='#1e293b',
                    markeredgecolor='#475569', markeredgewidth=1.5, zorder=4)
        
        offset_y = 80 if y > H/2 else -100
        ax.annotate(
            nid, (x, y + offset_y), fontsize=11, fontweight='bold',
            ha='center', va='center', color='#1e293b',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fef3c7',
                      edgecolor='#f59e0b', alpha=0.9)
        )
    
    # Load arrows
    if loads:
        for lx, lfy in loads:
            ax.annotate('', xy=(lx, H - 50), xytext=(lx, H + 200),
                        arrowprops=dict(arrowstyle='->', color='#dc2626', lw=2.5))
            ax.text(lx, H + 250, f'{abs(lfy):.0f} kN', ha='center',
                    fontsize=10, fontweight='bold', color='#dc2626')
    
    # Support symbols
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
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#2563eb', linewidth=2, label='Horizontal (tie/strut)'),
        Line2D([0], [0], color='#dc2626', linewidth=2.5, label='Diagonal (strut)'),
        Line2D([0], [0], color='#16a34a', linewidth=2, label='Vertical'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    margin = 400
    ax.set_xlim(-margin, L + margin)
    ax.set_ylim(-200, H + 400)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15)
    ax.set_xlabel('X (mm)', fontsize=10)
    ax.set_ylabel('Y (mm)', fontsize=10)
    
    title_text = title or f'Enhanced MAS-Generated STM (L={L:.0f}, H={H:.0f})'
    ax.set_title(title_text, fontsize=13, fontweight='bold', pad=15)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f">> Plot saved: {save_path}")
    
    plt.close()
    return fig


# ===========================================================
# MAIN
# ===========================================================

if __name__ == "__main__":
    
    print("\n" + "=" * 65)
    print("  Enhanced Multi-Agent System STM Generator")
    print("  - Two-stage decomposition (Node Placer + Connection Planner)")
    print("  - Best-of-N sampling with scoring")
    print("  - Feedback with previous output")
    print("  - Engineering review (LLM) + Code validation (deterministic)")
    print("=" * 65)
    
    # KDS Example 10.2 conditions
    beam_length = 6900
    beam_height = 2000
    beam_width = 500
    loads = [(2225, -2000), (4675, -2000)]
    support_positions = [(225, 'pin'), (6675, 'roller')]
    
    # Create enhanced MAS generator
    mas = EnhancedMASSTMGenerator(
        topology_model="deepseek-r1:14b",
        reviewer_model="qwen2.5:7b",
        optimizer_model="deepseek-r1:14b",
        max_retries=3,       # retries per stage per candidate
        n_candidates=3,      # Best-of-N
    )
    
    # Run
    result = mas.generate(
        beam_length=beam_length,
        beam_height=beam_height,
        beam_width=beam_width,
        loads=loads,
        support_positions=support_positions,
        fck=27.0,
        fy=400.0,
        verbose=True
    )
    
    # Plot
    if result['stm_data']:
        plot_stm(
            result['stm_data'],
            beam_length, beam_height,
            loads=loads,
            support_positions=support_positions,
            save_path='mas_stm_enhanced_result.png',
            title='Enhanced MAS-Generated STM - KDS 10.2'
        )
        
        # Print log summary
        print("\n── Pipeline Log ──")
        for entry in result['log']:
            print(f"  {entry.get('phase', entry.get('agent', '?'))}: "
                  f"{entry.get('status', json.dumps({k: v for k, v in entry.items() if k != 'phase'}, default=str)[:100])}")
        
        print(f"\n>> Result saved to mas_stm_enhanced_result.png")
    else:
        print("\n>> Generation failed. Check Ollama server and model availability.")
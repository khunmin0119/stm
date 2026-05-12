"""
PHASE 3: Multi-Agent System (MAS) based STM Auto-Generation
=============================================================
3 Agents using local Ollama LLMs:
  - Topology Agent (deepseek-r1:14b): Generate STM node/member layout
  - Validator Agent (qwen2.5:7b): Verify KDS compliance
  - Optimizer Agent (deepseek-r1:14b): Suggest improvements

Pipeline:
  Input -> Topology Agent -> Validator Agent -> [pass/fail]
  If fail -> feedback -> Topology Agent (retry)
  If pass -> Optimizer Agent -> improved STM -> Validator Agent -> Final STM
"""

import json
import math
import re
import time
import urllib.request

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
# 2. Agent Prompts
# ===========================================================

TOPOLOGY_AGENT_PROMPT = """You are the TOPOLOGY AGENT, a structural engineer specialized in Strut-and-Tie Model (STM) design for reinforced concrete deep beams.

Your job: Given beam geometry and loading, generate node coordinates and member connections.

## RULES (Korean Design Standard KDS 14 20 24)

1. COORDINATE SYSTEM: Origin (0,0) at bottom-left corner of beam
2. SUPPORT NODES: Bottom of beam, y = 125mm
3. LOAD NODES: Top of beam, y = H - 150mm (where H = beam height)  
4. DIAGONAL ANGLES: All diagonal members must be between 25 and 65 degrees from horizontal
5. NO MEMBER CROSSING: No two members may intersect except at shared nodes
6. MINIMUM 4 NODES
7. EXACTLY 2 POINT LOADS

## VALID STM PATTERNS

Pattern STM-1 (8 nodes): Support diagonals + intermediate nodes + vertical ties
  Top:    C---E---------F---G
  Bot:    A---D-----------H---B
  Diags:  A-C, E-D, F-H, B-G  (all going outward from center)
  Verts:  C-D, G-H
  
Pattern STM-2 (4 nodes): Direct diagonal connection
  Top:    B---------C
  Bot:    A---------D
  Diags:  A-B, C-D
  Chords: B-C (top), A-D (bottom)

## OUTPUT FORMAT
Respond with ONLY a JSON object. No explanation, no markdown, no extra text.
All coordinates must be plain numbers (NOT expressions like 6675-75).

## EXAMPLE OUTPUT (for a 6900x2000mm beam with loads at x=2225 and x=4675):

STM-2 (4-node) example:
{"nodes":{"A":[225,125],"B":[2225,1850],"C":[4675,1850],"D":[6675,125]},"connections":[["A","B"],["B","C"],["C","D"],["A","D"]],"supports":{"A":"pin","D":"roller"},"design_notes":"STM-2 4-node direct diagonal"}

STM-1 (8-node) example:
{"nodes":{"A":[225,125],"B":[6675,125],"C":[1225,1850],"D":[1225,125],"E":[2225,1850],"F":[4675,1850],"G":[5675,1850],"H":[5675,125]},"connections":[["C","E"],["E","F"],["F","G"],["A","D"],["D","H"],["H","B"],["A","C"],["C","D"],["B","G"],["G","H"],["E","D"],["F","H"]],"supports":{"A":"pin","B":"roller"},"design_notes":"STM-1 8-node with vertical ties"}

Generate YOUR OWN design based on the beam dimensions given. Use plain numbers only."""

VALIDATOR_AGENT_PROMPT = """You are the VALIDATOR AGENT. You check if a Strut-and-Tie Model (STM) follows KDS design rules.

Given an STM (nodes, connections, supports), check ALL of these rules:

1. MINIMUM 4 NODES
2. ALL NODES within beam boundary (0 <= x <= L, 0 <= y <= H)
3. DIAGONAL ANGLES between 25 and 65 degrees from horizontal
4. NO MEMBER CROSSING (no two members intersect except at shared nodes)
5. LOAD PATH exists from each load point to supports
6. SUPPORTS: must have at least one pin and one roller

For each rule, report PASS or FAIL with specific details.

Respond with ONLY a JSON object:
{
  "overall": "PASS" or "FAIL",
  "checks": [
    {"rule": "rule name", "status": "PASS" or "FAIL", "detail": "explanation"},
    ...
  ],
  "feedback": "If FAIL, specific instructions for what to fix. If PASS, empty string."
}"""

OPTIMIZER_AGENT_PROMPT = """You are the OPTIMIZER AGENT for Strut-and-Tie Model (STM) design.

Given a VALID STM that already passes all KDS rules, suggest improvements to:
1. Make diagonal angles closer to 45 degrees (optimal for force distribution)
2. Improve symmetry if loading is symmetric
3. Reduce total member length (less material)
4. Ensure load paths are as direct as possible

IMPORTANT CONSTRAINTS you must preserve:
- Keep ALL support node positions exactly the same
- Keep ALL load node x-positions the same (y can be at H-150)
- Keep angles between 25-65 degrees
- No member crossing
- Minimum 4 nodes
- Number of members must be >= number of nodes
- Every node must have at least 2 connections (NO dead-end nodes)
- Do NOT remove members that would disconnect any node

Respond with ONLY a JSON object of the improved STM:
{
  "nodes": {"A": [x, y], ...},
  "connections": [["A", "B"], ...],
  "supports": {"A": "pin", ...},
  "design_notes": "what was improved and why"
}

If no improvement is possible, return the original STM unchanged."""


# ===========================================================
# 3. Structural Checks (code-based, not LLM)
# ===========================================================

def code_validate(stm_data, beam_length, beam_height):
    """Deterministic structural validation (supplements LLM validator)"""
    errors = []
    warnings = []
    
    nodes = stm_data.get('nodes', {})
    connections = stm_data.get('connections', [])
    supports = stm_data.get('supports', {})
    
    # 1. Min 4 nodes
    if len(nodes) < 4:
        errors.append(f"Min 4 nodes required (got {len(nodes)})")
    
    # 1b. Min members: at least as many as nodes (closed truss)
    if len(connections) < len(nodes):
        errors.append(f"Too few members: {len(connections)} members for {len(nodes)} nodes (need at least {len(nodes)})")
    
    # 1c. Every node must have at least 2 connections
    if nodes and connections:
        degree = {nid: 0 for nid in nodes}
        for n1, n2 in connections:
            if n1 in degree:
                degree[n1] += 1
            if n2 in degree:
                degree[n2] += 1
        for nid, d in degree.items():
            if d < 2:
                errors.append(f"Node {nid} has only {d} connection(s) (need at least 2)")
    
    # 2. Node positions
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
        if dy < 50 or dx < 50:
            continue
        angle = math.degrees(math.atan2(dy, dx))
        if angle < 25 or angle > 65:
            errors.append(f"Member {n1}-{n2}: angle={angle:.1f} outside 25-65 range")
    
    # 4. Member crossing
    def segments_cross(p1, p2, p3, p4):
        def cp(o, a, b):
            return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
        d1, d2 = cp(p3,p4,p1), cp(p3,p4,p2)
        d3, d4 = cp(p1,p2,p3), cp(p1,p2,p4)
        return ((d1>0 and d2<0) or (d1<0 and d2>0)) and \
               ((d3>0 and d4<0) or (d3<0 and d4>0))
    
    for i in range(len(connections)):
        n1, n2 = connections[i]
        if n1 not in nodes or n2 not in nodes:
            continue
        for j in range(i+1, len(connections)):
            n3, n4 = connections[j]
            if n3 not in nodes or n4 not in nodes:
                continue
            if n1==n3 or n1==n4 or n2==n3 or n2==n4:
                continue
            if segments_cross(nodes[n1], nodes[n2], nodes[n3], nodes[n4]):
                errors.append(f"Members {n1}-{n2} and {n3}-{n4} cross")
    
    # 5. Connectivity
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
    
    # 6. Static determinacy
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


def parse_json_from_text(text):
    """Extract JSON from LLM response (handles markdown, extra text, arithmetic)"""
    if text is None:
        return None
    
    # Remove <think>...</think> blocks (deepseek-r1 reasoning)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # Fix arithmetic expressions in JSON like [6675 - 75, 1850]
    def fix_arithmetic(match):
        expr = match.group(0)
        try:
            # Evaluate simple arithmetic (only +, -, *, /)
            result = eval(expr)
            return str(result)
        except:
            return expr
    
    def preprocess_json(json_str):
        # Find number expressions like "6675 - 75" or "225 + 100"
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
    
    # Try finding JSON object
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
# 4. MAS Controller
# ===========================================================

class MASSTMGenerator:
    """Multi-Agent System for STM Generation
    
    Agent 1: Topology Agent (deepseek-r1:14b) - generates STM
    Agent 2: Validator Agent (qwen2.5:7b) - checks KDS rules
    Agent 3: Optimizer Agent (deepseek-r1:14b) - improves STM
    """
    
    def __init__(self, 
                 topology_model="deepseek-r1:14b",
                 validator_model="qwen2.5:7b",
                 optimizer_model="deepseek-r1:14b",
                 max_retries=3,
                 ollama_url="http://localhost:11434"):
        
        self.client = OllamaClient(ollama_url)
        self.topology_model = topology_model
        self.validator_model = validator_model
        self.optimizer_model = optimizer_model
        self.max_retries = max_retries
        self.log = []
    
    def generate(self, beam_length, beam_height, beam_width,
                 loads, support_positions, 
                 fck=27.0, fy=400.0, verbose=True):
        """
        Full MAS pipeline.
        
        Args:
            beam_length: Beam span (mm)
            beam_height: Beam height (mm)
            beam_width: Beam width (mm)
            loads: [(x1, fy1), (x2, fy2)] - 2 point loads
            support_positions: [(x1, 'pin'), (x2, 'roller')]
            
        Returns:
            dict with 'stm_data', 'validation', 'log', etc.
        """
        
        if verbose:
            print("=" * 65)
            print("  MAS-based STM Generation")
            print("=" * 65)
            print(f"  Beam: {beam_length} x {beam_height} x {beam_width} mm")
            print(f"  L/H: {beam_length/beam_height:.2f}")
            print(f"  Loads: {len(loads)} point loads")
            print(f"  Topology Agent: {self.topology_model}")
            print(f"  Validator Agent: {self.validator_model}")
            print(f"  Optimizer Agent: {self.optimizer_model}")
            print("=" * 65)
        
        # Check Ollama
        if not self.client.is_available():
            print("\n[ERROR] Ollama server is not running!")
            print("  Start it with: ollama serve")
            return {'stm_data': None, 'log': self.log}
        
        # Build user prompt for Topology Agent
        user_prompt = self._build_topology_prompt(
            beam_length, beam_height, beam_width,
            loads, support_positions, fck, fy
        )
        
        # ── PHASE 1: Topology Agent generates STM ──
        stm_data = None
        feedback = ""
        
        for attempt in range(1, self.max_retries + 1):
            if verbose:
                print(f"\n--- Round {attempt}/{self.max_retries} ---")
                print(f"[Topology Agent] Generating STM...")
            
            prompt = user_prompt
            if feedback:
                prompt += f"\n\n## FEEDBACK FROM VALIDATOR (must fix):\n{feedback}"
            
            t0 = time.time()
            response = self.client.chat(
                model=self.topology_model,
                system=TOPOLOGY_AGENT_PROMPT,
                user=prompt,
                temperature=0.3
            )
            t1 = time.time()
            
            if verbose:
                print(f"  Response time: {t1-t0:.1f}s")
            
            if response is None:
                self.log.append({'agent': 'topology', 'attempt': attempt, 'status': 'API_ERROR'})
                continue
            
            # Parse JSON
            stm_data = parse_json_from_text(response)
            
            if stm_data is None:
                if verbose:
                    print(f"  [FAIL] Could not parse JSON")
                    print(f"  Raw (first 300 chars): {response[:300]}")
                self.log.append({'agent': 'topology', 'attempt': attempt, 'status': 'PARSE_ERROR'})
                feedback = "Your previous response was not valid JSON. Respond with ONLY a JSON object, no explanation."
                continue
            
            if verbose:
                n_nodes = len(stm_data.get('nodes', {}))
                n_members = len(stm_data.get('connections', []))
                print(f"  Parsed: {n_nodes} nodes, {n_members} members")
            
            # ── PHASE 2: Code-based validation (deterministic) ──
            code_result = code_validate(stm_data, beam_length, beam_height)
            
            if not code_result['valid']:
                if verbose:
                    print(f"  [Code Validator] FAIL")
                    for e in code_result['errors']:
                        print(f"    - {e}")
                
                feedback = "ERRORS found in your STM:\n"
                for e in code_result['errors']:
                    feedback += f"- {e}\n"
                feedback += "\nFix ALL errors and regenerate. Respond with ONLY JSON."
                
                self.log.append({
                    'agent': 'code_validator', 'attempt': attempt,
                    'status': 'FAIL', 'errors': code_result['errors']
                })
                continue
            
            if verbose:
                print(f"  [Code Validator] PASS")
                for w in code_result['warnings']:
                    print(f"    Warning: {w}")
            
            # ── PHASE 2b: LLM Validator Agent ──
            if verbose:
                print(f"\n[Validator Agent] Checking KDS compliance...")
            
            t0 = time.time()
            val_response = self.client.chat(
                model=self.validator_model,
                system=VALIDATOR_AGENT_PROMPT,
                user=f"Beam: {beam_length}x{beam_height}mm\n\nSTM:\n{json.dumps(stm_data, indent=2)}",
                temperature=0.1
            )
            t1 = time.time()
            
            if verbose:
                print(f"  Response time: {t1-t0:.1f}s")
            
            val_result = parse_json_from_text(val_response)
            
            if val_result and val_result.get('overall') == 'FAIL':
                feedback_text = val_result.get('feedback', '')
                if verbose:
                    print(f"  [Validator Agent] FAIL")
                    if 'checks' in val_result:
                        for c in val_result['checks']:
                            status = c.get('status', '?')
                            print(f"    {c.get('rule','?')}: {status} - {c.get('detail','')}")
                    print(f"  Feedback: {feedback_text}")
                
                feedback = feedback_text if feedback_text else "Validator found issues. Please fix and regenerate."
                self.log.append({
                    'agent': 'validator', 'attempt': attempt,
                    'status': 'FAIL', 'result': val_result
                })
                continue
            
            if verbose:
                print(f"  [Validator Agent] PASS")
            
            self.log.append({
                'agent': 'validator', 'attempt': attempt,
                'status': 'PASS', 'stm': stm_data
            })
            break
        
        if stm_data is None or (code_result and not code_result['valid']):
            if verbose:
                print(f"\n[FAIL] Could not generate valid STM after {self.max_retries} attempts")
            return {'stm_data': None, 'log': self.log}
        
        # ── PHASE 3: Optimizer Agent ──
        if verbose:
            print(f"\n--- Optimization Phase ---")
            print(f"[Optimizer Agent] Analyzing for improvements...")
        
        t0 = time.time()
        opt_response = self.client.chat(
            model=self.optimizer_model,
            system=OPTIMIZER_AGENT_PROMPT,
            user=(
                f"Beam: {beam_length}x{beam_height}mm, L/H={beam_length/beam_height:.2f}\n"
                f"Loads at x={loads[0][0]}mm and x={loads[1][0]}mm\n"
                f"Supports at x={support_positions[0][0]}mm (pin) and x={support_positions[1][0]}mm (roller)\n\n"
                f"Current valid STM:\n{json.dumps(stm_data, indent=2)}\n\n"
                f"Suggest improvements while keeping all constraints."
            ),
            temperature=0.3
        )
        t1 = time.time()
        
        if verbose:
            print(f"  Response time: {t1-t0:.1f}s")
        
        opt_stm = parse_json_from_text(opt_response)
        
        if opt_stm and opt_stm.get('nodes') and opt_stm.get('connections'):
            # Validate optimized version
            opt_code_result = code_validate(opt_stm, beam_length, beam_height)
            
            if opt_code_result['valid']:
                if verbose:
                    print(f"  [Optimizer Agent] Improved STM is valid!")
                    if opt_stm.get('design_notes'):
                        print(f"  Notes: {opt_stm['design_notes']}")
                
                final_stm = opt_stm
                self.log.append({
                    'agent': 'optimizer', 'status': 'IMPROVED',
                    'stm': opt_stm
                })
            else:
                if verbose:
                    print(f"  [Optimizer Agent] Improved STM has errors, keeping original")
                    for e in opt_code_result['errors']:
                        print(f"    - {e}")
                final_stm = stm_data
                self.log.append({
                    'agent': 'optimizer', 'status': 'REJECTED',
                    'errors': opt_code_result['errors']
                })
        else:
            if verbose:
                print(f"  [Optimizer Agent] No valid improvement returned, keeping original")
            final_stm = stm_data
            self.log.append({'agent': 'optimizer', 'status': 'NO_CHANGE'})
        
        # ── Final Summary ──
        final_validation = code_validate(final_stm, beam_length, beam_height)
        
        if verbose:
            print(f"\n{'=' * 65}")
            print("  FINAL RESULT")
            print(f"{'=' * 65}")
            self._print_stm_summary(final_stm)
            print(f"\n  Valid: {final_validation['valid']}")
            print(f"{'=' * 65}")
        
        return {
            'stm_data': final_stm,
            'validation': final_validation,
            'log': self.log
        }
    
    def _build_topology_prompt(self, L, H, b, loads, supports, fck, fy):
        """Build user prompt for Topology Agent"""
        return (
            f"Design an STM for this deep beam:\n\n"
            f"Beam: {L}mm x {H}mm x {b}mm (L/H = {L/H:.2f})\n"
            f"Material: fck={fck}MPa, fy={fy}MPa\n\n"
            f"Load 1: x={loads[0][0]}mm, Fy={loads[0][1]}kN\n"
            f"Load 2: x={loads[1][0]}mm, Fy={loads[1][1]}kN\n\n"
            f"Pin support at x={supports[0][0]}mm\n"
            f"Roller support at x={supports[1][0]}mm\n\n"
            f"Bottom chord y = 125mm\n"
            f"Top chord y = {H - 150}mm\n\n"
            f"CRITICAL REQUIREMENTS:\n"
            f"- All diagonal angles between 25 and 65 degrees\n"
            f"- NO member crossing\n"
            f"- Minimum 4 nodes\n"
            f"- Number of members must be >= number of nodes\n"
            f"- Every node must have at least 2 connections\n"
            f"- Clear load path from loads to supports\n\n"
            f"Respond with ONLY a JSON object."
        )
    
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
            x1, y1 = nodes[n1]
            x2, y2 = nodes[n2]
            dx, dy = abs(x2-x1), abs(y2-y1)
            length = math.sqrt(dx**2 + dy**2)
            
            if dy < 50:
                mtype = "horizontal"
            elif dx < 50:
                mtype = "vertical"
            else:
                angle = math.degrees(math.atan2(dy, dx))
                mtype = f"diagonal ({angle:.1f} deg)"
            
            print(f"    {n1}-{n2}: {length:.0f}mm ({mtype})")
        
        if stm.get('design_notes'):
            print(f"\n  Notes: {stm['design_notes']}")


# ===========================================================
# 5. Visualization
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
        x1, y1 = nodes[n1_id]
        x2, y2 = nodes[n2_id]
        ax.plot([x1, x2], [y1, y2], color='#334155', linewidth=2.0, zorder=2)
        
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.annotate(
            f'{n1_id}-{n2_id}', (mx, my), fontsize=8,
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
    
    margin = 400
    ax.set_xlim(-margin, L + margin)
    ax.set_ylim(-200, H + 400)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15)
    ax.set_xlabel('X (mm)', fontsize=10)
    ax.set_ylabel('Y (mm)', fontsize=10)
    
    title_text = title or f'MAS-Generated STM (L={L:.0f}, H={H:.0f})'
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
    print("  Multi-Agent System STM Generator")
    print("  Using Ollama Local LLMs")
    print("=" * 65)
    
    # KDS Example 10.2 conditions
    beam_length = 6900
    beam_height = 2000
    beam_width = 500
    loads = [(2225, -2000), (4675, -2000)]
    support_positions = [(225, 'pin'), (6675, 'roller')]
    
    # Create MAS generator
    mas = MASSTMGenerator(
        topology_model="deepseek-r1:14b",
        validator_model="qwen2.5:7b",
        optimizer_model="deepseek-r1:14b",
        max_retries=3
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
            save_path='mas_stm_result.png',
            title='MAS-Generated STM - KDS Example 10.2'
        )
        
        print("\n>> Result saved to mas_stm_result.png")
    else:
        print("\n>> Generation failed. Check Ollama server and model availability.")
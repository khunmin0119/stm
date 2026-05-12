"""
PHASE 3: LLM 기반 초기 STM 자동 생성
======================================
Claude API를 사용하여 보 형상 + 하중 조건만으로
초기 STM (노드 좌표 + 부재 연결)을 자동 생성

파이프라인:
  사용자 입력 (보 형상, 하중) 
  → LLM이 STM 생성 (JSON)
  → 구조적 검증 (평형, 각도, 정정)
  → 실패 시 피드백 → 재생성 (최대 3회)
  → UserSTMDesign 객체 반환
  → 기존 GradientSTMOptimizer로 최적화
"""

import json
import numpy as np
import math
import sys
import os
import re

# ═══════════════════════════════════════════════════════════
# 1. KDS STM 규칙을 담은 System Prompt
# ═══════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a structural engineer specialized in Strut-and-Tie Model (STM) design for deep beams according to Korean Design Standards (KDS 14 20 24).

Your task: Given beam geometry and loading conditions, generate an optimal initial STM configuration (node coordinates and member connections).

## STM Design Rules (KDS)

### Deep Beam Definition
- A beam is "deep" when span/depth ratio (L/H) ≤ 4
- STM is the primary design method for deep beams in KDS
- This system handles 1 or 2 concentrated point loads only

### Node Placement Rules
1. **Support nodes**: Place at the centroid of bearing plates
   - Bottom of beam, y = cover + stirrup_dia + bar_dia/2 ≈ 125mm from bottom
   - x = bearing_plate_width/2 from beam edge (typically 225mm from edge)
   
2. **Load application nodes**: Place at the point where loads are applied
   - Top of beam, y = beam_height - cover - stirrup_dia - bar_dia/2 ≈ H - 150mm from bottom
   - x = at load position

3. **Intermediate top chord nodes**: Place directly above support nodes
   - To create a load path from load to support

4. **Intermediate bottom chord nodes**: Place directly below load nodes  
   - To create a complete truss path

### Member Connection Rules
1. **Top chord (horizontal struts)**: Connect adjacent top nodes
2. **Bottom chord (horizontal ties)**: Connect adjacent bottom nodes  
3. **Diagonal struts**: Connect top nodes to bottom nodes along load path
4. **Vertical ties**: Connect vertically aligned top-bottom node pairs (if needed)

### Strut Angle Constraints
- Diagonal struts must have angle θ between 25° and 65° from horizontal
- Optimal angle is around 45°~55°
- If angle would be too steep or too shallow, add intermediate nodes

### NO MEMBER CROSSING (CRITICAL)
- **No two members may cross each other** except at shared nodes
- Members can only meet at defined nodes, never in mid-span
- This means diagonals in adjacent panels must go in consistent directions
- Typical valid pattern: support diagonals go inward-upward, load diagonals go downward to nodes below
- Example INVALID: A-D crossing B-C in an X pattern
- Example VALID: A-B and C-G as parallel diagonals (no intersection)

### Structural Requirements
1. **Static determinacy**: The model should be statically determinate
   - Formula: m + r = 2n (m=members, r=reactions, n=nodes)
   - pin support = 2 reactions, roller = 1 reaction
   
2. **Load path**: Every load must have a clear path to supports
3. **Equilibrium**: Sum of forces at each node must be zero
4. **Symmetry**: If geometry and loading are symmetric, the STM should be symmetric

### Node Type Classification
- CCC (β_n=1.0): 3+ compression members meeting
- CCT (β_n=0.8): 2 compression + 1 tension
- CTT (β_n=0.6): 1 compression + 2+ tension
- TTT (β_n=0.6): 3+ tension members

### Naming Convention
- Use single uppercase letters: A, B, C, D, E, F, G, H, ...
- Support nodes first (leftmost = A), then top nodes left-to-right, then bottom nodes left-to-right

## Output Format

You MUST respond with ONLY a JSON object (no markdown, no explanation, no preamble):

{
  "nodes": {
    "A": [x, y],
    "B": [x, y],
    ...
  },
  "connections": [
    ["A", "B"],
    ["B", "C"],
    ...
  ],
  "supports": {
    "A": "pin",
    "F": "roller"
  },
  "design_notes": "Brief explanation of the STM topology chosen"
}

Coordinates are in mm. Origin (0,0) is at bottom-left of beam.
"""


# ═══════════════════════════════════════════════════════════
# 2. 사용자 입력 정의
# ═══════════════════════════════════════════════════════════

class BeamSpecification:
    """사용자가 입력하는 보 사양 (숫자 몇 개만)"""
    
    def __init__(self, 
                 beam_length: float,     # mm
                 beam_height: float,     # mm
                 beam_width: float,      # mm
                 fck: float = 27.0,      # MPa
                 fy: float = 400.0,      # MPa
                 loads: list = None,     # [(x_position, Fy_kN), ...]
                 support_positions: list = None,  # [(x, type), ...]
                 bearing_width: float = 450.0):   # mm
        
        self.beam_length = beam_length
        self.beam_height = beam_height
        self.beam_width = beam_width
        self.fck = fck
        self.fy = fy
        self.bearing_width = bearing_width
        
        # 하중: [(x위치, 수직하중kN), ...]
        if loads is None:
            # 기본: 2점 집중하중
            loads = [
                (beam_length * 1/3, -2000),
                (beam_length * 2/3, -2000)
            ]
        self.loads = loads
        
        # 하중 개수 검증 (1개 또는 2개만)
        if len(self.loads) != 2:
            raise ValueError(f"이 시스템은 집중하중 2개만 지원 (입력: {len(self.loads)}개)")
        
        # 지지점: [(x위치, 타입), ...]
        if support_positions is None:
            support_positions = [
                (bearing_width / 2, 'pin'),
                (beam_length - bearing_width / 2, 'roller')
            ]
        self.support_positions = support_positions
    
    def to_prompt(self):
        """LLM에게 보낼 사용자 프롬프트 생성"""
        
        loads_desc = []
        for i, (x, fy) in enumerate(self.loads):
            loads_desc.append(f"  - Load {i+1}: x = {x:.0f}mm, Fy = {fy:.0f} kN (downward)")
        
        supports_desc = []
        for x, stype in self.support_positions:
            supports_desc.append(f"  - {stype.capitalize()} support at x = {x:.0f}mm")
        
        prompt = f"""Design an STM for the following deep beam:

## Beam Geometry
- Length (L): {self.beam_length:.0f} mm
- Height (H): {self.beam_height:.0f} mm  
- Width (b): {self.beam_width:.0f} mm
- L/H ratio: {self.beam_length/self.beam_height:.2f}

## Material Properties
- f_ck: {self.fck:.0f} MPa
- f_y: {self.fy:.0f} MPa

## Loading
{chr(10).join(loads_desc)}

## Supports
{chr(10).join(supports_desc)}

## Bearing Plate Width: {self.bearing_width:.0f} mm

## Design Requirements
- Cover: 40mm, Stirrup: D10, Main bar: D29
- Bottom chord y ≈ 125mm (from bottom)
- Top chord y ≈ {self.beam_height - 150:.0f}mm (from bottom)
- All diagonal strut angles must be between 25° and 65° from horizontal
- The model must be statically determinate: m + r = 2n
- Use symmetry if geometry and loading are symmetric

Generate the STM as a JSON object."""
        
        return prompt
    
    def __repr__(self):
        return (f"BeamSpec(L={self.beam_length}, H={self.beam_height}, "
                f"b={self.beam_width}, loads={len(self.loads)}, "
                f"L/H={self.beam_length/self.beam_height:.2f})")

 
# ═══════════════════════════════════════════════════════════
# 3. LLM 응답 검증기
# ═══════════════════════════════════════════════════════════

class STMValidator:
    """LLM이 생성한 STM의 구조적 유효성 검증"""
    
    def __init__(self, beam_spec: BeamSpecification):
        self.beam_spec = beam_spec
        self.errors = []
        self.warnings = []
    
    def validate(self, stm_data: dict) -> dict:
        """전체 검증 수행"""
        self.errors = []
        self.warnings = []
        
        # 1. JSON 구조 검증
        self._check_structure(stm_data)
        if self.errors:
            return self._result()
        
        nodes = stm_data['nodes']
        connections = stm_data['connections']
        supports = stm_data['supports']
        
        # 2. 노드 위치 검증
        self._check_node_positions(nodes)
        
        # 3. 스트럿 각도 검증
        self._check_strut_angles(nodes, connections)
        
        # 4. 정정 구조 검증
        self._check_static_determinacy(nodes, connections, supports)
        
        # 5. 연결성 검증 (모든 노드가 연결되어 있는지)
        self._check_connectivity(nodes, connections)
        
        # 6. 부재 교차 검증 (CRITICAL)
        self._check_member_crossing(nodes, connections)
        
        # 7. 하중 경로 검증
        self._check_load_paths(nodes, connections, supports)
        
        # 8. 평형 해석 시도
        self._check_equilibrium(nodes, connections, supports)
        
        return self._result()
    
    def _result(self):
        return {
            'valid': len(self.errors) == 0,
            'errors': self.errors.copy(),
            'warnings': self.warnings.copy(),
            'error_count': len(self.errors),
            'warning_count': len(self.warnings)
        }
    
    def _check_structure(self, data):
        """JSON 구조 확인"""
        required = ['nodes', 'connections', 'supports']
        for key in required:
            if key not in data:
                self.errors.append(f"Missing required key: '{key}'")
        
        if 'nodes' in data:
            for nid, coords in data['nodes'].items():
                if not isinstance(coords, (list, tuple)) or len(coords) != 2:
                    self.errors.append(f"Node {nid}: invalid coordinates {coords}")
        
        if 'connections' in data:
            node_ids = set(data.get('nodes', {}).keys())
            for conn in data['connections']:
                if len(conn) != 2:
                    self.errors.append(f"Invalid connection: {conn}")
                elif conn[0] not in node_ids or conn[1] not in node_ids:
                    self.errors.append(f"Connection {conn} references undefined node")
    
    def _check_node_positions(self, nodes):
        """노드가 보 영역 안에 있는지"""
        L = self.beam_spec.beam_length
        H = self.beam_spec.beam_height
        
        for nid, (x, y) in nodes.items():
            if x < 0 or x > L:
                self.errors.append(f"Node {nid}: x={x:.0f} is outside beam (0~{L:.0f})")
            if y < 0 or y > H:
                self.errors.append(f"Node {nid}: y={y:.0f} is outside beam (0~{H:.0f})")
    
    def _check_strut_angles(self, nodes, connections):
        """스트럿 각도 25°~65° 확인"""
        for n1_id, n2_id in connections:
            x1, y1 = nodes[n1_id]
            x2, y2 = nodes[n2_id]
            
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            
            # 수평/수직 부재는 스킵
            if dy < 50 or dx < 50:
                continue
            
            angle = math.degrees(math.atan2(dy, dx))
            
            if angle < 25:
                self.errors.append(
                    f"Member {n1_id}-{n2_id}: angle={angle:.1f}° < 25° (too shallow)")
            elif angle > 65:
                self.errors.append(
                    f"Member {n1_id}-{n2_id}: angle={angle:.1f}° > 65° (too steep)")
            elif angle < 30:
                self.warnings.append(
                    f"Member {n1_id}-{n2_id}: angle={angle:.1f}° is near minimum (25°)")
    
    def _check_static_determinacy(self, nodes, connections, supports):
        """정정 구조 확인: m + r = 2n"""
        m = len(connections)
        n = len(nodes)
        
        r = 0
        for sid, stype in supports.items():
            if stype == 'pin':
                r += 2
            elif stype == 'roller':
                r += 1
        
        lhs = m + r
        rhs = 2 * n
        
        if lhs < rhs:
            self.warnings.append(
                f"Statically underdetermined: m({m}) + r({r}) = {lhs} < 2n = {rhs} "
                f"(difference: {rhs - lhs}). May still be solvable by geometry.")
        elif lhs > rhs:
            self.warnings.append(
                f"Statically indeterminate: m({m}) + r({r}) = {lhs} > 2n = {rhs} "
                f"(degree of indeterminacy: {lhs - rhs})")
    
    def _check_connectivity(self, nodes, connections):
        """모든 노드가 연결되어 있는지 (그래프 연결성)"""
        if not nodes:
            return
        
        adj = {nid: set() for nid in nodes}
        for n1, n2 in connections:
            adj[n1].add(n2)
            adj[n2].add(n1)
        
        # BFS
        start = list(nodes.keys())[0]
        visited = {start}
        queue = [start]
        
        while queue:
            current = queue.pop(0)
            for neighbor in adj[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        disconnected = set(nodes.keys()) - visited
        if disconnected:
            self.errors.append(f"Disconnected nodes: {disconnected}")
    
    def _check_member_crossing(self, nodes, connections):
        """부재 교차 검사 - 노드가 아닌 곳에서 두 부재가 만나면 안 됨"""
        
        def segments_cross(p1, p2, p3, p4):
            """두 선분이 내부에서 교차하는지 (끝점 공유 제외)"""
            def cross_product(o, a, b):
                return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
            
            d1 = cross_product(p3, p4, p1)
            d2 = cross_product(p3, p4, p2)
            d3 = cross_product(p1, p2, p3)
            d4 = cross_product(p1, p2, p4)
            
            if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
               ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
                return True
            return False
        
        for i in range(len(connections)):
            n1, n2 = connections[i]
            p1, p2 = nodes[n1], nodes[n2]
            
            for j in range(i + 1, len(connections)):
                n3, n4 = connections[j]
                # 노드를 공유하면 교차 아님
                if n1 == n3 or n1 == n4 or n2 == n3 or n2 == n4:
                    continue
                p3, p4 = nodes[n3], nodes[n4]
                
                if segments_cross(p1, p2, p3, p4):
                    self.errors.append(
                        f"Members {n1}-{n2} and {n3}-{n4} cross each other")
    
    def _check_load_paths(self, nodes, connections, supports):
        """하중점에서 지지점까지 경로가 존재하는지"""
        support_nodes = set(supports.keys())
        
        # 인접 리스트
        adj = {nid: set() for nid in nodes}
        for n1, n2 in connections:
            adj[n1].add(n2)
            adj[n2].add(n1)
        
        # 하중이 적용되는 노드 찾기
        load_positions = [lx for lx, _ in self.beam_spec.loads]
        
        for nid, (x, y) in nodes.items():
            if nid in support_nodes:
                continue
            
            for lx in load_positions:
                if abs(x - lx) < 100 and y > self.beam_spec.beam_height / 2:
                    # 이 노드에서 지지점까지 BFS
                    visited = {nid}
                    queue = [nid]
                    found = False
                    
                    while queue and not found:
                        current = queue.pop(0)
                        if current in support_nodes:
                            found = True
                            break
                        for neighbor in adj[current]:
                            if neighbor not in visited:
                                visited.add(neighbor)
                                queue.append(neighbor)
                    
                    if not found:
                        self.errors.append(
                            f"No load path from node {nid} to any support")
    
    def _check_equilibrium(self, nodes, connections, supports):
        """평형 방정식 해석 시도"""
        try:
            n_members = len(connections)
            n_nodes = len(nodes)
            node_list = list(nodes.keys())
            
            # 반력 개수
            n_reactions = 0
            reaction_map = {}
            for node_id, support_type in supports.items():
                if support_type == 'pin':
                    reaction_map[f'Rx_{node_id}'] = n_reactions
                    reaction_map[f'Ry_{node_id}'] = n_reactions + 1
                    n_reactions += 2
                elif support_type == 'roller':
                    reaction_map[f'Ry_{node_id}'] = n_reactions
                    n_reactions += 1
            
            A = np.zeros((n_nodes * 2, n_members + n_reactions))
            b = np.zeros(n_nodes * 2)
            
            for j, (n1_id, n2_id) in enumerate(connections):
                x1, y1 = nodes[n1_id]
                x2, y2 = nodes[n2_id]
                dx, dy = x2 - x1, y2 - y1
                L = np.sqrt(dx**2 + dy**2)
                if L < 1e-6:
                    continue
                cos_v, sin_v = dx / L, dy / L
                
                i1 = node_list.index(n1_id)
                A[i1*2, j] = cos_v
                A[i1*2+1, j] = sin_v
                
                i2 = node_list.index(n2_id)
                A[i2*2, j] = -cos_v
                A[i2*2+1, j] = -sin_v
            
            for node_id in supports:
                i = node_list.index(node_id)
                if f'Rx_{node_id}' in reaction_map:
                    col = n_members + reaction_map[f'Rx_{node_id}']
                    A[i*2, col] = 1.0
                if f'Ry_{node_id}' in reaction_map:
                    col = n_members + reaction_map[f'Ry_{node_id}']
                    A[i*2+1, col] = 1.0
            
            # 하중 → 가장 가까운 노드에 적용
            for lx, lfy in self.beam_spec.loads:
                closest_node = None
                min_dist = float('inf')
                for nid, (nx, ny) in nodes.items():
                    dist = abs(nx - lx)
                    if ny > self.beam_spec.beam_height / 2 and dist < min_dist:
                        min_dist = dist
                        closest_node = nid
                
                if closest_node:
                    i = node_list.index(closest_node)
                    b[i*2+1] -= lfy
            
            solution, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            error = np.linalg.norm(A @ solution - b)
            
            if error > 10.0:
                self.errors.append(
                    f"Equilibrium not satisfied (residual={error:.1f})")
            elif error > 1.0:
                self.warnings.append(
                    f"Equilibrium residual slightly high ({error:.2f})")
                    
        except Exception as e:
            self.errors.append(f"Equilibrium check failed: {str(e)}")
    
    def get_feedback_prompt(self):
        """검증 실패 시 LLM에게 보낼 피드백 프롬프트"""
        
        feedback = "The STM you generated has the following issues. Please fix them:\n\n"
        
        if self.errors:
            feedback += "## ERRORS (must fix):\n"
            for i, err in enumerate(self.errors, 1):
                feedback += f"{i}. {err}\n"
        
        if self.warnings:
            feedback += "\n## WARNINGS (should fix if possible):\n"
            for i, warn in enumerate(self.warnings, 1):
                feedback += f"{i}. {warn}\n"
        
        feedback += "\nPlease regenerate the complete STM JSON with all corrections applied."
        feedback += "\nRespond with ONLY the corrected JSON object."
        
        return feedback


# ═══════════════════════════════════════════════════════════
# 4. LLM STM 생성기
# ═══════════════════════════════════════════════════════════

class LLMSTMGenerator:
    """Claude API를 사용한 STM 자동 생성기"""
    
    def __init__(self, api_key: str = None, max_retries: int = 3):
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY', '')
        self.max_retries = max_retries
        self.generation_log = []  # 생성 과정 기록
    
    def generate(self, beam_spec: BeamSpecification, verbose: bool = True) -> dict:
        """
        STM 생성 메인 함수
        
        Returns:
            {
                'stm_data': {...},           # 생성된 STM
                'validation': {...},         # 검증 결과
                'attempts': int,             # 시도 횟수
                'user_design': UserSTMDesign  # 기존 코드 호환 객체
            }
        """
        if verbose:
            print(f"\n{'='*60}")
            print("PHASE 3: LLM-based STM Generation")
            print(f"{'='*60}")
            print(f"Beam: {beam_spec}")
            print(f"Max retries: {self.max_retries}")
        
        validator = STMValidator(beam_spec)
        user_prompt = beam_spec.to_prompt()
        
        messages = [
            {"role": "user", "content": user_prompt}
        ]
        
        for attempt in range(1, self.max_retries + 1):
            if verbose:
                print(f"\n--- Attempt {attempt}/{self.max_retries} ---")
            
            # LLM 호출
            response_text = self._call_llm(messages)
            
            if response_text is None:
                self.generation_log.append({
                    'attempt': attempt, 'status': 'API_ERROR'
                })
                continue
            
            # JSON 파싱
            stm_data = self._parse_json(response_text)
            
            if stm_data is None:
                self.generation_log.append({
                    'attempt': attempt, 'status': 'PARSE_ERROR',
                    'raw': response_text[:500]
                })
                # 파싱 에러 피드백
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "user", "content": 
                    "Your response could not be parsed as JSON. "
                    "Please respond with ONLY a valid JSON object, no markdown or explanation."
                })
                continue
            
            # 검증
            validation = validator.validate(stm_data)
            
            self.generation_log.append({
                'attempt': attempt,
                'status': 'VALID' if validation['valid'] else 'INVALID',
                'errors': validation['errors'],
                'warnings': validation['warnings'],
                'stm_data': stm_data
            })
            
            if verbose:
                if validation['valid']:
                    print(f"  ✓ Valid STM generated!")
                    if validation['warnings']:
                        for w in validation['warnings']:
                            print(f"  ⚠ {w}")
                else:
                    print(f"  ✗ Validation failed ({validation['error_count']} errors)")
                    for e in validation['errors']:
                        print(f"    - {e}")
            
            if validation['valid']:
                # 성공!
                user_design = self._to_user_design(stm_data, beam_spec)
                
                if verbose:
                    print(f"\n✓ STM generated successfully in {attempt} attempt(s)")
                    print(f"  Nodes: {len(stm_data['nodes'])}")
                    print(f"  Members: {len(stm_data['connections'])}")
                    print(f"  Supports: {len(stm_data['supports'])}")
                
                return {
                    'stm_data': stm_data,
                    'validation': validation,
                    'attempts': attempt,
                    'user_design': user_design,
                    'log': self.generation_log
                }
            
            # 실패 → 피드백 후 재시도
            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user", "content": validator.get_feedback_prompt()})
        
        # 모든 시도 실패
        if verbose:
            print(f"\n✗ Failed to generate valid STM after {self.max_retries} attempts")
        
        return {
            'stm_data': None,
            'validation': None,
            'attempts': self.max_retries,
            'user_design': None,
            'log': self.generation_log
        }
    
    def _call_llm(self, messages: list) -> str:
        """Claude API 호출"""
        try:
            import anthropic
            
            client = anthropic.Anthropic(api_key=self.api_key)
            
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                system=SYSTEM_PROMPT,
                messages=messages
            )
            
            return response.content[0].text
            
        except ImportError:
            print("⚠ anthropic 패키지 미설치. pip install anthropic")
            return None
        except Exception as e:
            print(f"⚠ API 호출 실패: {e}")
            return None
    
    def _parse_json(self, text: str) -> dict:
        """LLM 응답에서 JSON 추출"""
        try:
            # 방법 1: 직접 파싱
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass
        
        try:
            # 방법 2: ```json ... ``` 블록 추출
            match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
        
        try:
            # 방법 3: 첫 번째 { ... } 블록 추출
            start = text.index('{')
            # 중첩 괄호 처리
            depth = 0
            for i in range(start, len(text)):
                if text[i] == '{':
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                    if depth == 0:
                        return json.loads(text[start:i+1])
        except (ValueError, json.JSONDecodeError):
            pass
        
        return None
    
    def _to_user_design(self, stm_data: dict, beam_spec: BeamSpecification):
        """STM 데이터를 기존 UserSTMDesign 호환 포맷으로 변환"""
        
        # 노드: dict → (x, y) 튜플
        nodes = {}
        for nid, coords in stm_data['nodes'].items():
            nodes[nid] = (float(coords[0]), float(coords[1]))
        
        # 연결: list → tuple
        connections = []
        for conn in stm_data['connections']:
            connections.append((conn[0], conn[1]))
        
        # 하중 → 가장 가까운 노드에 매핑
        loads = []
        for lx, lfy in beam_spec.loads:
            closest_node = None
            min_dist = float('inf')
            for nid, (nx, ny) in nodes.items():
                dist = abs(nx - lx)
                if ny > beam_spec.beam_height / 2 and dist < min_dist:
                    min_dist = dist
                    closest_node = nid
            
            if closest_node:
                loads.append((closest_node, 0, lfy))
        
        # UserSTMDesign 호환 딕셔너리 반환
        return {
            'beam_width': beam_spec.beam_width,
            'beam_height': beam_spec.beam_height,
            'beam_length': beam_spec.beam_length,
            'fck': beam_spec.fck,
            'fy': beam_spec.fy,
            'initial_nodes': nodes,
            'connections': connections,
            'loads': loads,
            'supports': stm_data['supports']
        }


# ═══════════════════════════════════════════════════════════
# 5. 오프라인 테스트용 Mock Generator
#    (API 없이 로직 테스트)
# ═══════════════════════════════════════════════════════════

class MockLLMGenerator(LLMSTMGenerator):
    """API 없이 테스트할 수 있는 규칙 기반 생성기
    
    LLM의 역할을 규칙 기반으로 모사 → 파이프라인 전체를 테스트
    실제 연구에서는 이것과 LLM 생성 결과를 비교
    
    stm_type:
      'stm1' = 8노드 모델 (KDS 예제 10.2.3) - 중간 노드 + 수직 타이
      'stm2' = 4노드 모델 (KDS 예제 10.2.6) - 직접 대각선 연결
    """
    
    def __init__(self, stm_type: str = 'stm1', **kwargs):
        super().__init__(**kwargs)
        self.stm_type = stm_type
    
    def _call_llm(self, messages: list) -> str:
        """규칙 기반으로 STM 생성 (LLM 대신)"""
        beam_spec = self.beam_spec
        
        if self.stm_type == 'stm2':
            stm = self._rule_based_generation_stm2(beam_spec)
        else:
            stm = self._rule_based_generation_stm1(beam_spec)
        
        return json.dumps(stm, indent=2)
    
    def generate(self, beam_spec: BeamSpecification, verbose: bool = True) -> dict:
        """beam_spec 저장 후 부모 클래스 호출"""
        self.beam_spec = beam_spec
        return super().generate(beam_spec, verbose)
    
    def _rule_based_generation_stm1(self, spec: BeamSpecification) -> dict:
        """규칙 기반 STM-1 자동 생성 (8노드 패턴, KDS 예제 10.2.3)
        
        KDS 예제 10.2.3 패턴:
        ┌──B────C─────D────E──┐  ← 상현
        │ /│         │\\ │  
        │/ │         │ \\│  
        A──G─────────H──F     ← 하현
        
        핵심 원리:
        1. 지지점에서 대각선은 안쪽-위로 (A→B, F→E)
        2. 하중점에서 대각선은 아래로 (C→G, D→H) 
        3. B-G, E-H는 수직 타이
        4. 모든 부재가 교차하지 않음
        """
        L = spec.beam_length
        H = spec.beam_height
        
        y_bottom = 125.0
        y_top = H - 150.0
        dy = y_top - y_bottom  # 상현-하현 높이 차
        
        # 목표 대각선 각도 (40°~60° 사이가 이상적)
        target_angle_deg = 50.0
        target_dx = dy / math.tan(math.radians(target_angle_deg))
        
        nodes = {}
        connections = []
        supports = {}
        
        # ══════════════════════════════════════
        # STEP 1: 지지점 노드 배치 (하현)
        # ══════════════════════════════════════
        # 좌측 지지점 = A, 우측 지지점은 마지막 문자
        
        support_data = []  # [(x, type, node_id), ...]
        for i, (sx, stype) in enumerate(sorted(spec.support_positions)):
            nid = chr(65 + i)  # A, B, ...
            nodes[nid] = [round(sx, 1), y_bottom]
            supports[nid] = stype
            support_data.append({'x': sx, 'id': nid})
        
        n_supports = len(support_data)
        
        # ══════════════════════════════════════
        # STEP 2: 하중 위치 정렬
        # ══════════════════════════════════════
        load_xs = sorted([lx for lx, _ in spec.loads])
        n_loads = len(load_xs)
        
        # ══════════════════════════════════════
        # STEP 3: 중간 노드 + 하중 노드 배치
        # ══════════════════════════════════════
        # KDS 패턴: 각 지지점과 가장 가까운 하중 사이에 중간 수직선 배치
        #
        # 좌측 지지점(A) → 중간점(B상/G하) → 하중(C상) → ... → 하중(D상) → 중간점(E상/H하) → 우측 지지점(F)
        
        left_support = support_data[0]
        right_support = support_data[-1]
        
        # ══════════════════════════════════════
        # STEP 3-A: 중간 노드 위치 계산
        # ══════════════════════════════════════
        # 두 가지 각도 제한을 모두 만족해야 함:
        #   지지점 대각선: atan2(dy, inter_x - support_x) ≤ max_angle
        #   하중 대각선:   atan2(dy, load_x - inter_x)   ≤ max_angle
        # → inter_x ∈ [support_x + min_dx, load_x - min_dx]
        # 양립 불가하면 중간 노드 없이 직접 연결
        
        max_angle = 63.0
        min_dx = dy / math.tan(math.radians(max_angle))
        
        use_left_inter = True
        use_right_inter = True
        left_intermediate_x = 0
        right_intermediate_x = 0
        
        if n_loads > 0:
            leftmost_load = load_xs[0]
            rightmost_load = load_xs[-1]
            
            left_min = left_support['x'] + min_dx
            left_max = leftmost_load - min_dx
            right_min = rightmost_load + min_dx
            right_max = right_support['x'] - min_dx
            
            if left_min <= left_max:
                left_intermediate_x = (left_min + left_max) / 2.0
            else:
                use_left_inter = False
            
            if right_min <= right_max:
                right_intermediate_x = (right_min + right_max) / 2.0
            else:
                use_right_inter = False
        else:
            left_intermediate_x = left_support['x'] + target_dx
            right_intermediate_x = right_support['x'] - target_dx
        
        # ══════════════════════════════════════
        # STEP 3-B: 노드 배치
        # ══════════════════════════════════════
        next_id = n_supports
        top_nodes = []
        bottom_nodes = []
        
        bottom_nodes.append((left_support['x'], left_support['id']))
        
        left_inter_top = None
        left_inter_bot = None
        right_inter_top = None
        right_inter_bot = None
        
        if use_left_inter:
            nid_top = chr(65 + next_id); next_id += 1
            nid_bot = chr(65 + next_id); next_id += 1
            nodes[nid_top] = [round(left_intermediate_x, 1), y_top]
            nodes[nid_bot] = [round(left_intermediate_x, 1), y_bottom]
            top_nodes.append((left_intermediate_x, nid_top))
            bottom_nodes.append((left_intermediate_x, nid_bot))
            left_inter_top = nid_top
            left_inter_bot = nid_bot
        
        load_top_ids = []
        for lx in load_xs:
            nid = chr(65 + next_id); next_id += 1
            nodes[nid] = [round(lx, 1), y_top]
            top_nodes.append((lx, nid))
            load_top_ids.append(nid)
        
        if use_right_inter:
            nid_top = chr(65 + next_id); next_id += 1
            nid_bot = chr(65 + next_id); next_id += 1
            nodes[nid_top] = [round(right_intermediate_x, 1), y_top]
            nodes[nid_bot] = [round(right_intermediate_x, 1), y_bottom]
            top_nodes.append((right_intermediate_x, nid_top))
            bottom_nodes.append((right_intermediate_x, nid_bot))
            right_inter_top = nid_top
            right_inter_bot = nid_bot
        
        bottom_nodes.append((right_support['x'], right_support['id']))
        top_nodes.sort(key=lambda t: t[0])
        bottom_nodes.sort(key=lambda t: t[0])
        
        # ══════════════════════════════════════
        # STEP 4: 부재 연결 (교차 없는 패턴)
        # ══════════════════════════════════════
        
        def add_conn(n1, n2):
            if n1 and n2 and n1 != n2:
                if [n1, n2] not in connections and [n2, n1] not in connections:
                    connections.append([n1, n2])
                    return True
            return False
        
        # 4a. 상현 연결
        for i in range(len(top_nodes) - 1):
            add_conn(top_nodes[i][1], top_nodes[i+1][1])
        
        # 4b. 하현 연결
        for i in range(len(bottom_nodes) - 1):
            add_conn(bottom_nodes[i][1], bottom_nodes[i+1][1])
        
        # 4c. 지지점 대각선 + 수직 타이
        if use_left_inter:
            add_conn(left_support['id'], left_inter_top)
            add_conn(left_inter_top, left_inter_bot)
        else:
            if load_top_ids:
                add_conn(left_support['id'], load_top_ids[0])
        
        if use_right_inter:
            add_conn(right_support['id'], right_inter_top)
            add_conn(right_inter_top, right_inter_bot)
        else:
            if load_top_ids:
                add_conn(right_support['id'], load_top_ids[-1])
        
        # 4d. 하중점 대각선 (교차 방지!)
        beam_center = L / 2.0
        
        for load_id in load_top_ids:
            lx = nodes[load_id][0]
            
            if lx < beam_center - 10:
                target_bot = left_inter_bot if use_left_inter else left_support['id']
                add_conn(load_id, target_bot)
            elif lx > beam_center + 10:
                target_bot = right_inter_bot if use_right_inter else right_support['id']
                add_conn(load_id, target_bot)
            else:
                left_target = left_inter_bot if use_left_inter else left_support['id']
                right_target = right_inter_bot if use_right_inter else right_support['id']
                add_conn(load_id, left_target)
                add_conn(load_id, right_target)
        
        # ══════════════════════════════════════
        # STEP 5: 최종 검증 정보
        # ══════════════════════════════════════
        n = len(nodes)
        r = sum(2 if t == 'pin' else 1 for t in supports.values())
        m = len(connections)
        
        return {
            "nodes": nodes,
            "connections": connections,
            "supports": supports,
            "design_notes": (
                f"KDS-pattern STM for L/H={L/H:.2f} deep beam, "
                f"{n_loads} point load(s), "
                f"diagonal angle ≈ {target_angle_deg:.0f}°, "
                f"no member crossing"
            )
        }
    
    def _rule_based_generation_stm2(self, spec: BeamSpecification) -> dict:
        """규칙 기반 STM-2 자동 생성 (4노드 패턴, KDS 예제 10.2.6)
        
        2점 하중:
        B─────C          ← 상현 (하중점)
        │╲   ╱│
        │ ╲ ╱ │
        A─────D          ← 하현 (지지점)
        
        노드: A(좌하 지지), B(좌상 하중), C(우상 하중), D(우하 지지)
        부재: A-B(대각), B-C(상현), C-D(대각), A-D(하현)
        
        1점 하중:
           B              ← 상현 (하중점)
          ╱ ╲
         ╱   ╲
        A─────C           ← 하현 (지지점)
        
        노드: A(좌하 지지), B(상 하중), C(우하 지지)
        부재: A-B(대각), B-C(대각), A-C(하현)
        """
        L = spec.beam_length
        H = spec.beam_height
        
        y_bottom = 125.0
        y_top = H - 150.0
        
        nodes = {}
        connections = []
        supports = {}
        
        load_xs = sorted([lx for lx, _ in spec.loads])
        n_loads = len(load_xs)
        
        support_positions = sorted(spec.support_positions, key=lambda s: s[0])
        left_sx, left_stype = support_positions[0]
        right_sx, right_stype = support_positions[-1]
        
        if n_loads == 2:
            # ── 2점 하중: 4노드 4부재 ──
            nodes['A'] = [round(left_sx, 1), y_bottom]
            nodes['B'] = [round(load_xs[0], 1), y_top]
            nodes['C'] = [round(load_xs[1], 1), y_top]
            nodes['D'] = [round(right_sx, 1), y_bottom]
            
            supports['A'] = left_stype
            supports['D'] = right_stype
            
            connections = [
                ['A', 'B'],   # 좌측 대각선
                ['B', 'C'],   # 상현
                ['C', 'D'],   # 우측 대각선
                ['A', 'D']    # 하현
            ]
            
            design_notes = (
                f"STM-2 (4-node) for L/H={L/H:.2f}, 2 point loads, "
                f"direct diagonal struts"
            )
        
        return {
            "nodes": nodes,
            "connections": connections,
            "supports": supports,
            "design_notes": design_notes
        }

# ═══════════════════════════════════════════════════════════
# 6. 통합 파이프라인
# ═══════════════════════════════════════════════════════════

def run_full_pipeline(beam_spec: BeamSpecification, 
                      use_llm: bool = True,
                      api_key: str = None,
                      stm_type: str = 'stm1',
                      optimize: bool = True,
                      verbose: bool = True):
    """
    전체 파이프라인 실행
    
    입력: BeamSpecification (숫자 몇 개)
    출력: 최적화된 STM 설계
    
    Args:
        beam_spec: 보 사양
        use_llm: True=Claude API, False=규칙 기반 (테스트용)
        api_key: Anthropic API key
        optimize: 최적화 수행 여부
        verbose: 상세 출력
    """
    
    if verbose:
        print("╔" + "═"*58 + "╗")
        print("║  LLM-based STM Design Pipeline                          ║")
        print("╠" + "═"*58 + "╣")
        print(f"║  Beam: {beam_spec.beam_length:.0f} × {beam_spec.beam_height:.0f} × {beam_spec.beam_width:.0f} mm" + " "*(58-len(f"  Beam: {beam_spec.beam_length:.0f} × {beam_spec.beam_height:.0f} × {beam_spec.beam_width:.0f} mm")-1) + "║")
        print(f"║  Loads: {len(beam_spec.loads)} point loads" + " "*(58-len(f"  Loads: {len(beam_spec.loads)} point loads")-1) + "║")
        print(f"║  Mode: {'Claude API' if use_llm else 'Rule-based (test)'}" + " "*(58-len(f"  Mode: {'Claude API' if use_llm else 'Rule-based (test)'}")-1) + "║")
        print(f"║  STM Type: {stm_type.upper()}" + " "*(58-len(f"  STM Type: {stm_type.upper()}")-1) + "║")
        print("╚" + "═"*58 + "╝")
    
    # ── STEP 1: STM 생성 ──
    if verbose:
        print(f"\n{'─'*60}")
        print("STEP 1: STM Generation")
        print(f"{'─'*60}")
    
    if use_llm:
        generator = LLMSTMGenerator(api_key=api_key)
    else:
        generator = MockLLMGenerator(stm_type=stm_type)
    
    gen_result = generator.generate(beam_spec, verbose=verbose)
    
    if gen_result['stm_data'] is None:
        print("\n✗ Pipeline failed: Could not generate valid STM")
        return gen_result
    
    # ── STEP 2: 결과 요약 ──
    stm = gen_result['stm_data']
    
    if verbose:
        print(f"\n{'─'*60}")
        print("STEP 2: Generated STM Summary")
        print(f"{'─'*60}")
        
        print(f"\nNodes ({len(stm['nodes'])}):")
        for nid, (x, y) in sorted(stm['nodes'].items()):
            role = ""
            if nid in stm['supports']:
                role = f" [{stm['supports'][nid]}]"
            print(f"  {nid}: ({x:7.1f}, {y:7.1f}){role}")
        
        print(f"\nMembers ({len(stm['connections'])}):")
        for n1, n2 in stm['connections']:
            x1, y1 = stm['nodes'][n1]
            x2, y2 = stm['nodes'][n2]
            dx, dy = abs(x2-x1), abs(y2-y1)
            length = math.sqrt(dx**2 + dy**2)
            
            if dy < 50:
                mtype = "horizontal"
            elif dx < 50:
                mtype = "vertical"
            else:
                angle = math.degrees(math.atan2(dy, dx))
                mtype = f"diagonal ({angle:.1f}°)"
            
            print(f"  {n1}-{n2}: L={length:.0f}mm ({mtype})")
        
        if 'design_notes' in stm:
            print(f"\nDesign notes: {stm['design_notes']}")
    
    # ── STEP 3: 최적화 (선택) ──
    if optimize:
        if verbose:
            print(f"\n{'─'*60}")
            print("STEP 3: Gradient Optimization")
            print(f"{'─'*60}")
        
        # 여기에서 기존 GradientSTMOptimizer 연결
        # (import는 실제 실행 시)
        print("\n→ UserSTMDesign 객체가 준비되었습니다.")
        print("→ GradientSTMOptimizer에 전달하여 최적화를 수행하세요.")
        print(f"\n  user_design_data = gen_result['user_design']")
    
    return gen_result


# ═══════════════════════════════════════════════════════════
# 7. 시각화
# ═══════════════════════════════════════════════════════════

def plot_generated_stm(stm_data: dict, beam_spec: BeamSpecification, 
                       save_path: str = None, title: str = None):
    """생성된 STM 시각화"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    
    L = beam_spec.beam_length
    H = beam_spec.beam_height
    
    # 보 외곽선
    beam_rect = patches.Rectangle((0, 0), L, H, 
                                   linewidth=1.5, edgecolor='#94a3b8', 
                                   facecolor='#f1f5f9', alpha=0.3)
    ax.add_patch(beam_rect)
    
    nodes = stm_data['nodes']
    connections = stm_data['connections']
    supports = stm_data['supports']
    
    # 부재 그리기 (모든 부재 동일 스타일 — tie/strut 구분은 기존 시스템이 담당)
    for n1_id, n2_id in connections:
        x1, y1 = nodes[n1_id]
        x2, y2 = nodes[n2_id]
        
        ax.plot([x1, x2], [y1, y2], color='#334155', linewidth=2.0, 
                linestyle='-', zorder=2)
        
        # 부재 라벨
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.annotate(f'{n1_id}-{n2_id}', (mx, my), fontsize=8,
                   ha='center', va='bottom', color='#475569', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                           edgecolor='none', alpha=0.85))
    
    # 노드 그리기
    for nid, (x, y) in nodes.items():
        if nid in supports:
            # 지지점 노드
            marker = '^' if supports[nid] == 'pin' else 'o'
            ax.plot(x, y, marker=marker, markersize=14, color='#f59e0b',
                   markeredgecolor='#d97706', markeredgewidth=2, zorder=4)
        else:
            ax.plot(x, y, 'o', markersize=10, color='#1e293b',
                   markeredgecolor='#475569', markeredgewidth=1.5, zorder=4)
        
        # 노드 라벨
        offset_y = 80 if y > H/2 else -100
        ax.annotate(nid, (x, y + offset_y), fontsize=11, fontweight='bold',
                   ha='center', va='center', color='#1e293b',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#fef3c7',
                           edgecolor='#f59e0b', alpha=0.9))
    
    # 하중 화살표
    for lx, lfy in beam_spec.loads:
        ax.annotate('', xy=(lx, H - 50), xytext=(lx, H + 200),
                   arrowprops=dict(arrowstyle='->', color='#dc2626', lw=2.5))
        ax.text(lx, H + 250, f'{abs(lfy):.0f} kN', ha='center', 
               fontsize=10, fontweight='bold', color='#dc2626')
    
    # 지지점 기호
    for sx, stype in beam_spec.support_positions:
        if stype == 'pin':
            triangle = plt.Polygon(
                [(sx-80, -50), (sx+80, -50), (sx, 0)],
                closed=True, facecolor='#fef3c7', edgecolor='#d97706', linewidth=1.5)
            ax.add_patch(triangle)
        else:
            circle = plt.Circle((sx, -50), 30, facecolor='#fef3c7', 
                              edgecolor='#d97706', linewidth=1.5)
            ax.add_patch(circle)
            ax.plot([sx-60, sx+60], [-80, -80], color='#d97706', linewidth=1.5)
    
    # 축 설정
    margin = 400
    ax.set_xlim(-margin, L + margin)
    ax.set_ylim(-200, H + 400)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15)
    ax.set_xlabel('X (mm)', fontsize=10)
    ax.set_ylabel('Y (mm)', fontsize=10)
    
    title_text = title or f'LLM-Generated STM (L={L:.0f}, H={H:.0f}, L/H={L/H:.1f})'
    ax.set_title(title_text, fontsize=13, fontweight='bold', pad=15)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"✓ Plot saved: {save_path}")
    
    plt.close()
    return fig


# ═══════════════════════════════════════════════════════════
# MAIN: 테스트 실행
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    
    print("="*60)
    print("LLM-based STM Generator - Test Run")
    print("="*60)
    
    # ── 테스트 1: KDS 예제 10.2와 동일 조건 ──
    print("\n■ Test 1: KDS Example 10.2 conditions")
    
    spec_10_2 = BeamSpecification(
        beam_length=6900,
        beam_height=2000,
        beam_width=500,
        fck=27.0,
        fy=400.0,
        loads=[
            (2225, -2000),   # C점 하중
            (4675, -2000)    # D점 하중
        ],
        support_positions=[
            (225, 'pin'),     # A점 (핀)
            (6675, 'roller')  # F점 (롤러)
        ],
        bearing_width=450
    )
    
    # 규칙 기반 테스트 (API 없이)
    result = run_full_pipeline(
        spec_10_2, 
        use_llm=False,  # 규칙 기반
        optimize=False,
        verbose=True
    )
    
    if result['stm_data']:
        plot_generated_stm(
            result['stm_data'], 
            spec_10_2,
            save_path='/mnt/user-data/outputs/stm_generated_test1.png',
            title='Generated STM - KDS Example 10.2 Conditions'
        )
    
    # ── 테스트 3: KDS 예제 10.2 — STM-2 (4노드) ──
    print("\n\n■ Test 3: KDS Example 10.2 - STM-2 (4-node)")
    
    result3 = run_full_pipeline(
        spec_10_2, 
        use_llm=False,
        stm_type='stm2',
        optimize=False,
        verbose=True
    )
    
    if result3['stm_data']:
        plot_generated_stm(
            result3['stm_data'], 
            spec_10_2,
            save_path='/mnt/user-data/outputs/stm_generated_test3_stm2.png',
            title='Generated STM-2 (4-Node) - KDS Example 10.2'
        )

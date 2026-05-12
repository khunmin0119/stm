"""
Nodal Zone Design Module (KDS 14 20 24)
========================================
트러스 해석 결과(nodes, connections, forces)를 입력받아
자동으로 절점 종류 판별, 스트럿/타이 폭 계산, KDS 검증, 시각화를 수행.

어떤 보에도 작동하는 범용 함수.
"""

import numpy as np
import math
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon


def load_from_stm_json(json_path):
    """
    mas_stm_final.py가 저장한 stm_result.json을 읽어서
    nodal_zone_design 함수에 맞는 형식으로 변환.

    Returns:
        dict with: nodes, connections, member_forces, reactions, supports,
                   beam_length, beam_height, beam_width, loads
    """
    import json

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1. nodes: {"A": {"x":225, "y":125}} → {"A": [225, 125]}
    nodes = {}
    for nid, coord in data['nodes'].items():
        nodes[nid] = [coord['x'], coord['y']]

    # 2. connections: members 리스트에서 추출
    connections = []
    for m in data['members']:
        connections.append(m['nodes'])

    # 3. member_forces: kN → N, list → dict with tuple key
    member_forces = {}
    for m in data['members']:
        n1, n2 = m['nodes']
        if 'force_kN' in m and m['force_kN'] is not None:
            member_forces[(n1, n2)] = m['force_kN'] * 1000  # kN → N

    # 4. reactions: {"A": {"Rx_kN":0, "Ry_kN":2000}} → {"A": (0, 2000000)}
    reactions = {}
    for nid, r in data.get('reactions', {}).items():
        reactions[nid] = (r['Rx_kN'] * 1000, r['Ry_kN'] * 1000)  # kN → N

    # 5. supports
    supports = data.get('support_nodes', {})

    # 6. beam info
    beam = data.get('beam', {})
    beam_length = beam.get('length_mm', 6900)
    beam_height = beam.get('height_mm', 2000)
    beam_width = beam.get('width_mm', 500)

    # 7. loads
    loads = []
    for ld in data.get('loads', []):
        loads.append((ld['x_mm'], ld['Fy_kN']))

    return {
        'nodes': nodes,
        'connections': connections,
        'member_forces': member_forces,
        'reactions': reactions,
        'supports': supports,
        'beam_length': beam_length,
        'beam_height': beam_height,
        'beam_width': beam_width,
        'loads': loads,
    }


def load_all_candidates_from_json(json_path):
    """
    JSON에서 전체 후보(all_stm_candidates)를 읽어서 각각 변환.
    all_stm_candidates가 없으면 best만 반환.
    """
    import json

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    beam = data.get('beam', {})
    beam_info = {
        'beam_length': beam.get('length_mm', 6900),
        'beam_height': beam.get('height_mm', 2000),
        'beam_width': beam.get('width_mm', 500),
        'loads': [(ld['x_mm'], ld['Fy_kN']) for ld in data.get('loads', [])],
    }

    def convert_candidate(cand):
        nodes = {nid: [c['x'], c['y']] for nid, c in cand['nodes'].items()}
        connections = [m['nodes'] for m in cand['members']]
        member_forces = {}
        for m in cand['members']:
            n1, n2 = m['nodes']
            if 'force_kN' in m and m['force_kN'] is not None:
                member_forces[(n1, n2)] = m['force_kN'] * 1000
        reactions = {}
        for nid, r in cand.get('reactions', {}).items():
            reactions[nid] = (r['Rx_kN'] * 1000, r['Ry_kN'] * 1000)
        supports = cand.get('support_nodes', {})
        return {
            'nodes': nodes, 'connections': connections,
            'member_forces': member_forces, 'reactions': reactions,
            'supports': supports,
            'candidate_id': cand.get('candidate_id', 0),
            'n_pairs': cand.get('n_pairs', 0),
            'score': cand.get('score', 0),
        }

    candidates = []
    if 'all_stm_candidates' in data:
        for cand in data['all_stm_candidates']:
            candidates.append(convert_candidate(cand))
    else:
        # all_stm_candidates가 없으면 best만
        best = load_from_stm_json(json_path)
        best['candidate_id'] = 1
        best['n_pairs'] = 0
        best['score'] = data.get('score', 0)
        candidates.append(best)

    return candidates, beam_info


def design_all_candidates(json_path, fck, fy, bw, bearing_plate, cover,
                           stirrup_dia, main_bar_dia, phi=0.75,
                           save_dir=None, verbose=True):
    """
    전체 후보에 대해 노드 설계 → KDS 통과 여부 판정 → 통과한 것 중 최적 선택.

    Returns:
        dict with: all_results, kds_passed, best_candidate
    """
    candidates, beam_info = load_all_candidates_from_json(json_path)
    H = beam_info['beam_height']
    L = beam_info['beam_length']
    loads = beam_info['loads']

    if verbose:
        print(f"\n{'='*60}")
        print(f"  KDS Nodal Zone Design — {len(candidates)} Candidates")
        print(f"{'='*60}")

    all_results = []
    kds_passed = []

    for cand in candidates:
        cid = cand['candidate_id']
        n_pairs = cand['n_pairs']
        score = cand['score']
        n_nodes = len(cand['nodes'])

        if verbose:
            print(f"\n{'─'*55}")
            print(f"  C{cid}: {n_pairs} pairs ({n_nodes} nodes), Score={score:.4f}")
            print(f"{'─'*55}")

        design = design_nodal_zones(
            cand['nodes'], cand['connections'],
            cand['member_forces'], cand['reactions'], cand['supports'],
            fck=fck, fy=fy, bw=bw, bearing_plate=bearing_plate,
            cover=cover, stirrup_dia=stirrup_dia, main_bar_dia=main_bar_dia,
            phi=phi, beam_height=H, loads=loads, verbose=verbose
        )

        # 전체 노드 PASS 여부
        all_pass = all(nv['all_ok'] for nv in design['node_verification'].values())

        result_entry = {
            'candidate_id': cid, 'n_pairs': n_pairs, 'score': score,
            'n_nodes': n_nodes, 'design': design, 'kds_pass': all_pass,
            'cand_data': cand
        }
        all_results.append(result_entry)

        if all_pass:
            kds_passed.append(result_entry)

        # 시각화
        if save_dir:
            status_tag = "PASS" if all_pass else "FAIL"
            save_path = os.path.join(save_dir,
                f'nodal_C{cid}_{n_pairs}pairs_{status_tag}.png')
            plot_stm_with_nodal_zones(
                cand['nodes'], cand['connections'],
                cand['member_forces'], cand['reactions'], cand['supports'],
                design, beam_length=L, beam_height=H, loads=loads,
                save_path=save_path,
                title=f'C{cid}: {n_pairs} pairs ({n_nodes}n) — KDS [{status_tag}] — Score={score:.4f}'
            )

            # ── 절점 형상 개별 시각화 ──
            shape_path = os.path.join(save_dir,
                f'node_shapes_C{cid}_{n_pairs}pairs_{status_tag}.png')
            plot_node_shapes(
                nodes=cand['nodes'],
                connections=cand['connections'],
                member_forces=cand['member_forces'],
                design_result=design,
                title=f'절점 형상 — C{cid}: {n_pairs}쌍 {n_nodes}노드 [{status_tag}]',
                save_path=shape_path
            )

    # 결과 요약
    if verbose:
        print(f"\n{'='*60}")
        print(f"  KDS Results Summary")
        print(f"{'='*60}")
        for r in all_results:
            s = "PASS" if r['kds_pass'] else "FAIL"
            print(f"  C{r['candidate_id']}: {r['n_pairs']} pairs, "
                  f"{r['n_nodes']}n, Score={r['score']:.4f} → KDS [{s}]")

        if kds_passed:
            best = min(kds_passed, key=lambda r: r['score'])
            print(f"\n  >>> BEST (KDS PASS + lowest Score): "
                  f"C{best['candidate_id']} ({best['n_pairs']} pairs, Score={best['score']:.4f})")
        else:
            print(f"\n  >>> WARNING: No candidate passed KDS!")

    best_candidate = min(kds_passed, key=lambda r: r['score']) if kds_passed else None

    return {
        'all_results': all_results,
        'kds_passed': kds_passed,
        'best_candidate': best_candidate,
    }


def design_nodal_zones(nodes, connections, member_forces, reactions, supports,
                        fck, fy, bw, bearing_plate, cover,
                        stirrup_dia, main_bar_dia, phi=0.75,
                        beam_height=None, loads=None, verbose=True):
    """
    KDS 14 20 24 기반 노드 설계.

    Args:
        nodes: dict {'A': [x, y], ...}
        connections: list [['A','B'], ...]
        member_forces: dict {('A','B'): force_N, ...}  (+tension, -compression)
        reactions: dict {'A': (Rx_N, Ry_N), ...}
        supports: dict {'A': 'pin', ...}
        fck, fy: MPa
        bw: mm (보 폭)
        bearing_plate: mm (지압판 폭)
        cover, stirrup_dia, main_bar_dia: mm
        phi: 강도감소계수 (기본 0.75)
        beam_height: mm (보 전체 높이, None이면 자동 계산)
        loads: list [(x_mm, P_kN), ...] — 외부 하중 점하중 (하중점 노드 판별용).
               None이면 상단 노드 모두 일반 chord로 가정.

    Returns:
        dict with all design results
    """

    # ── 자동 형식 변환: score_stm_detailed 출력 호환 ──
    converted_forces = {}
    for key, val in member_forces.items():
        if isinstance(val, dict):
            converted_forces[key] = val['force_N']
        else:
            converted_forces[key] = val
    member_forces = converted_forces

    # ── 0. 기본값 계산 ──
    all_y = [y for _, (x, y) in nodes.items()]  # fix: handle both list and tuple
    y_top_max = max(all_y)
    y_bot_min = min(all_y)
    if beam_height is None:
        beam_height = y_top_max - y_bot_min + 2 * cover

    # 유효깊이 & 타이/스트럿 폭
    d = beam_height - cover - stirrup_dia - main_bar_dia - main_bar_dia / 2
    # wt: 입력 cover/stirrup/bar 기반 + 노드 좌표(2*y_bot)와 일관성
    wt_input = 2 * (cover + stirrup_dia + main_bar_dia + main_bar_dia / 2)
    wt_input = max(wt_input, 250)
    wt_geom = 2 * y_bot_min  # 노드 좌표에서 역산
    wt = wt_geom if wt_geom > 0 else wt_input
    # ws: 상단 chord 폭 (노드 좌표에서 역산)
    ws = 2 * (beam_height - y_top_max)
    if ws <= 0:
        ws = wt_input  # fallback

    if verbose:
        print(f"\n  [Nodal Zone Design — KDS 14 20 24]")
        print(f"    fck={fck}MPa, fy={fy}MPa, bw={bw}mm, φ={phi}")
        print(f"    Bearing plate={bearing_plate}mm, cover={cover}mm")
        print(f"    d={d:.1f}mm, wt={wt:.1f}mm, ws={ws:.1f}mm")

    # ── Helper: force lookup ──
    def get_force(n1, n2):
        if (n1, n2) in member_forces: return member_forces[(n1, n2)]
        if (n2, n1) in member_forces: return member_forces[(n2, n1)]
        return 0

    def get_coords(nid):
        c = nodes[nid]
        return (c[0], c[1]) if isinstance(c, (list, tuple)) else (c[0], c[1])

    # ── Helper: 수직 타이 폭 — 양옆 노드 x 정사영 중점 합산 ──
    def compute_vertical_tie_width(nid_top, nid_bot):
        """
        수직 타이 폭 = 좌측 인접 거리/2 + 우측 인접 거리/2.

        gen_training_v2.py의 get_w_vert 로직 (양옆 중점 합산).
        예: x=1225, 좌(225)/우(2225) → (1225-225)/2 + (2225-1225)/2 = 1000mm
        대칭 배치에서는 종전 '최소 거리' 방식과 같은 값이지만,
        비대칭일 때 더 정확한 정사영 폭을 산출함.
        """
        # 수직 타이는 양 끝 x 동일 — 하단 노드 기준
        x_n = get_coords(nid_bot)[0]

        # 모든 노드 x좌표 정렬 (부동소수 안정용 1mm 반올림)
        all_xs = sorted(set(round(get_coords(n)[0], 1) for n in nodes))

        # x_n에 가장 가까운 값 매칭 (부동소수 오차 대비)
        x_n_rounded = round(x_n, 1)
        if x_n_rounded not in all_xs:
            x_n_rounded = min(all_xs, key=lambda x: abs(x - x_n_rounded))

        idx = all_xs.index(x_n_rounded)
        left = (all_xs[idx] - all_xs[idx - 1]) / 2 if idx > 0 else 0
        right = (all_xs[idx + 1] - all_xs[idx]) / 2 if idx < len(all_xs) - 1 else 0
        result = left + right

        # fallback: 양옆 노드 모두 없을 경우 wt 반환
        return result if result > 0 else wt

    # ── Helper: 노드별 분류 (지점/하중점/일반) ──
    def is_bottom_node(nid):
        """하단 chord에 속한 노드인지."""
        y_n = get_coords(nid)[1]
        return abs(y_n - y_bot_min) < 50

    def is_top_node(nid):
        """상단 chord에 속한 노드인지."""
        y_n = get_coords(nid)[1]
        return abs(y_n - y_top_max) < 50

    def is_load_node(nid):
        """외부 점하중이 작용하는 노드인지 (loads 인자 기반)."""
        if loads is None:
            return False
        x_n, y_n = get_coords(nid)
        if not is_top_node(nid):
            return False
        for lx, _P in loads:
            if abs(x_n - lx) < 50:
                return True
        return False

    # ── Helper: 노드의 수평 부재 폭 (지점/하중점=bearing, 일반=wt or ws) ──
    def get_horizontal_width(nid):
        """그 노드에 연결된 수평 방향 폭 (정사영의 cos 성분)."""
        if nid in supports:
            return bearing_plate
        if is_load_node(nid):
            return bearing_plate
        if is_bottom_node(nid):
            return wt
        if is_top_node(nid):
            return ws
        # 중간 높이 노드 (드물지만 안전 fallback)
        return wt

    # ── Helper: 노드 시점에서 본 대각선 스트럿 폭 ──
    # gen_training_v2.py의 get_w_diag 로직과 동일.
    # 한 대각선 부재가 양 끝 노드에서 다른 w_act를 가질 수 있음.
    def get_diag_width_at_node(nid, member_angle):
        """
        노드 nid에서 본 대각선 부재의 실제 폭.
            w_act = w_h × |cos(angle)| + w_v × |sin(angle)|
        여기서:
          - w_h = 노드의 수평 부재 폭 (bearing/wt/ws)
          - w_v = 노드의 수직 부재 폭 (compute_vertical_tie_width 로직)
          - angle은 절댓값 처리된 부재 경사각
        """
        sin_a = abs(math.sin(member_angle))
        cos_a = abs(math.cos(member_angle))
        w_h = get_horizontal_width(nid)
        # 수직 폭: 같은 x 정사영 합산 (compute_vertical_tie_width 로직 재사용)
        # nid가 수직 타이 노드가 아니어도 동일 공식 적용
        x_n = get_coords(nid)[0]
        all_xs = sorted(set(round(get_coords(n)[0], 1) for n in nodes))
        x_n_rounded = round(x_n, 1)
        if x_n_rounded not in all_xs:
            x_n_rounded = min(all_xs, key=lambda x: abs(x - x_n_rounded))
        idx = all_xs.index(x_n_rounded)
        left = (all_xs[idx] - all_xs[idx - 1]) / 2 if idx > 0 else 0
        right = (all_xs[idx + 1] - all_xs[idx]) / 2 if idx < len(all_xs) - 1 else 0
        w_v = (left + right) if (left + right) > 0 else wt

        return w_h * cos_a + w_v * sin_a

    # ── 타이 폭 사전 계산 (수직/수평 구분) ──
    tie_widths = {}
    for conn in connections:
        n1, n2 = conn[0], conn[1]
        f = get_force(n1, n2)
        if f <= 0:  # 스트럿이면 스킵
            continue
        x1, y1 = get_coords(n1)
        x2, y2 = get_coords(n2)
        dx, dy = abs(x2 - x1), abs(y2 - y1)

        if dx < 50 and dy > 500:  # 수직 타이
            tw = compute_vertical_tie_width(n1, n2)
            tie_widths[(n1, n2)] = tw
            tie_widths[(n2, n1)] = tw
        else:  # 수평 타이 — 상단/하단 구분
            if abs(y1 - y_bot_min) < 50 and abs(y2 - y_bot_min) < 50:
                tw_h = wt  # 하단 chord 타이
            elif abs(y1 - y_top_max) < 50 and abs(y2 - y_top_max) < 50:
                tw_h = ws  # 상단 chord 타이 (드물지만 가능)
            else:
                tw_h = wt
            tie_widths[(n1, n2)] = tw_h
            tie_widths[(n2, n1)] = tw_h

    if verbose and any(tw != wt for tw in tie_widths.values()):
        print(f"\n    Tie widths (auto-computed):")
        seen = set()
        for (n1, n2), tw in tie_widths.items():
            key = tuple(sorted([n1, n2]))
            if key not in seen:
                seen.add(key)
                label = "vertical" if tw != wt else "horizontal"
                print(f"      {n1}-{n2}: {tw:.0f}mm ({label})")

    # ── 1. 각 노드에 연결된 부재 분석 ──
    node_members = {nid: [] for nid in nodes}
    for conn in connections:
        n1, n2 = conn[0], conn[1]
        f = get_force(n1, n2)
        x1, y1 = get_coords(n1)
        x2, y2 = get_coords(n2)
        dx, dy = x2 - x1, y2 - y1
        angle = math.atan2(dy, dx)
        length = math.sqrt(dx**2 + dy**2)

        info = {'n1': n1, 'n2': n2, 'force': f, 'angle': angle,
                'length': length, 'dx': dx, 'dy': dy}
        node_members[n1].append({**info, 'other': n2, 'direction': 'outgoing'})
        node_members[n2].append({**info, 'other': n1, 'direction': 'incoming',
                                  'angle': math.atan2(-dy, -dx)})

    # ── 2. 절점 종류 자동 판별 ──
    node_types = {}
    for nid in nodes:
        members = node_members[nid]
        n_tension = sum(1 for m in members if m['force'] > 0)
        n_compression = sum(1 for m in members if m['force'] < 0)

        if n_tension == 0:
            ntype, beta_n = 'CCC', 1.0
        elif n_tension == 1:
            ntype, beta_n = 'CCT', 0.8
        elif n_tension >= 2:
            ntype, beta_n = 'CTT', 0.6

        node_types[nid] = {'type': ntype, 'beta_n': beta_n,
                           'n_tension': n_tension, 'n_compression': n_compression}

    if verbose:
        print(f"\n    {'Node':>4s} {'Type':>4s} {'βn':>4s} {'T':>2s} {'C':>2s}")
        print(f"    {'-'*20}")
        for nid in sorted(nodes.keys()):
            info = node_types[nid]
            print(f"    {nid:>4s} {info['type']:>4s} {info['beta_n']:>4.1f} "
                  f"{info['n_tension']:>2d} {info['n_compression']:>2d}")

    # ── 3. 스트럿 종류 & 폭 자동 판별 ──
    # 대각선 스트럿: 양 끝 노드에서 각각 다른 w_act를 가질 수 있음
    # (gen_training_v2.py의 get_w_diag 로직 — 노드별 정사영)
    # 요약값으로는 더 작은 w_act를 사용 (보수적 판정).
    strut_results = {}
    for conn in connections:
        n1, n2 = conn[0], conn[1]
        f = get_force(n1, n2)
        if f >= 0:  # 타이는 스킵
            continue

        x1, y1 = get_coords(n1)
        x2, y2 = get_coords(n2)
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        length = math.sqrt(dx**2 + dy**2)
        angle = math.atan2(dy, dx)
        Fu = abs(f) / 1000  # kN

        # 스트럿 종류: 대각선=병모양(0.6), 수평/수직=프리즘(1.0)
        if dy < 50 or dx < 50:
            beta_s = 1.0
            stype = 'Prismatic'
        else:
            beta_s = 0.6
            stype = 'Bottle-shaped'

        # 실제 스트럿 폭 — 노드별 정사영 (수정 ④)
        if dy > 50 and dx > 50:  # 대각선
            w_act_n1 = get_diag_width_at_node(n1, angle)
            w_act_n2 = get_diag_width_at_node(n2, angle)
            wsb = min(w_act_n1, w_act_n2)  # 요약: 더 작은 쪽이 critical
        elif dx < 50:  # 수직 스트럿 (드물지만)
            tw_v = compute_vertical_tie_width(n1, n2)
            w_act_n1 = w_act_n2 = tw_v
            wsb = tw_v
        else:  # 수평 스트럿 (상단 chord 등)
            if abs(y1 - y_top_max) < 50:
                wsb = ws
            else:
                wsb = wt
            w_act_n1 = w_act_n2 = wsb

        # 요구 스트럿 폭
        w_s_req = (Fu * 1e3) / (phi * 0.85 * beta_s * fck * bw)

        strut_results[(n1, n2)] = {
            'force_kN': Fu, 'beta_s': beta_s, 'type': stype,
            'angle_deg': math.degrees(angle),
            'w_actual': wsb,           # 요약 (보수적 = min)
            'w_act_n1': w_act_n1,      # n1 시점 폭
            'w_act_n2': w_act_n2,      # n2 시점 폭
            'w_required': w_s_req, 'ok': wsb >= w_s_req, 'length': length
        }

    if verbose:
        print(f"\n    {'Strut':>6s} {'βs':>4s} {'F(kN)':>8s} {'w_req':>8s} {'w_act':>8s} {'OK?':>4s}")
        print(f"    {'-'*42}")
        for (n1, n2), sr in sorted(strut_results.items()):
            s = "OK" if sr['ok'] else "NG"
            print(f"    {n1+n2:>6s} {sr['beta_s']:>4.1f} {sr['force_kN']:>8.1f} "
                  f"{sr['w_required']:>8.1f} {sr['w_actual']:>8.1f} {s:>4s}")

    # ── 4. 절점 검증 ──
    node_verification = {}
    for nid in nodes:
        beta_n = node_types[nid]['beta_n']
        ntype = node_types[nid]['type']
        checks = []

        # 반력 체크
        if nid in reactions:
            Rx, Ry = reactions[nid]
            R = max(abs(Rx), abs(Ry)) / 1000
            w_req = (R * 1e3) / (phi * 0.85 * beta_n * fck * bw)
            checks.append({'member': 'R/P', 'force_kN': R,
                           'w_required': w_req, 'w_actual': bearing_plate,
                           'ok': bearing_plate >= w_req})

        # 각 부재 체크
        for m in node_members[nid]:
            n1, n2 = m['n1'], m['n2']
            Fu = abs(m['force']) / 1000
            w_req = (Fu * 1e3) / (phi * 0.85 * beta_n * fck * bw)

            # 실제 폭 결정 — 노드 nid 시점에서 본 폭
            dx, dy = abs(m['dx']), abs(m['dy'])
            if m['force'] > 0:  # 타이
                key_tw = (n1, n2) if (n1, n2) in tie_widths else (n2, n1)
                w_act = tie_widths.get(key_tw, wt)
            else:  # 스트럿
                key = (n1, n2) if (n1, n2) in strut_results else (n2, n1)
                if key in strut_results:
                    sr = strut_results[key]
                    # 대각선이면 nid 쪽 폭 사용 (수정 ④)
                    if 'w_act_n1' in sr and 'w_act_n2' in sr:
                        w_act = sr['w_act_n1'] if nid == key[0] else sr['w_act_n2']
                    else:
                        w_act = sr['w_actual']
                else:
                    w_act = 300

            mname = f"{n1}{n2}" if n1 < n2 else f"{n2}{n1}"
            checks.append({'member': mname, 'force_kN': Fu,
                           'w_required': w_req, 'w_actual': w_act,
                           'ok': w_act >= w_req})

        all_ok = all(c['ok'] for c in checks)
        node_verification[nid] = {'type': ntype, 'beta_n': beta_n,
                                   'checks': checks, 'all_ok': all_ok}

    if verbose:
        print(f"\n    === Node Verification ===")
        for nid in sorted(nodes.keys()):
            nv = node_verification[nid]
            print(f"    [{nid}] {nv['type']} (βn={nv['beta_n']})")
            for c in nv['checks']:
                s = "OK" if c['ok'] else "NG"
                print(f"      {c['member']:>5s}: F={c['force_kN']:>8.1f}kN, "
                      f"w_req={c['w_required']:>7.1f}, w_act={c['w_actual']:>7.1f} [{s}]")
            print(f"      → {'PASS' if nv['all_ok'] else 'FAIL'}")

    return {
        'node_types': node_types,
        'strut_results': strut_results,
        'node_verification': node_verification,
        'tie_widths': tie_widths,
        'd': d, 'wt': wt, 'ws': ws,
        'bearing_plate': bearing_plate,
        'fck': fck, 'fy': fy, 'bw': bw, 'phi': phi,
    }


def plot_stm_with_nodal_zones(nodes, connections, member_forces, reactions, supports,
                                design_result, beam_length, beam_height,
                                loads=None, save_path=None, title=None):
    """
    부재 폭 + 노드 종류가 표시된 STM 시각화 (범용).
    """
    wt = design_result['wt']
    tie_widths = design_result.get('tie_widths', {})
    strut_results = design_result['strut_results']
    node_types = design_result['node_types']

    def get_force(n1, n2):
        if (n1, n2) in member_forces: return member_forces[(n1, n2)]
        if (n2, n1) in member_forces: return member_forces[(n2, n1)]
        return 0

    def get_coords(nid):
        c = nodes[nid]
        return (c[0], c[1])

    def draw_member(ax, x1, y1, x2, y2, w, fc, ec, alpha=0.5):
        dx, dy = x2 - x1, y2 - y1
        l = math.sqrt(dx**2 + dy**2)
        if l < 1: return
        px, py = -dy / l, dx / l
        poly = Polygon([
            (x1 + px*w/2, y1 + py*w/2), (x2 + px*w/2, y2 + py*w/2),
            (x2 - px*w/2, y2 - py*w/2), (x1 - px*w/2, y1 - py*w/2),
        ], closed=True, fc=fc, ec=ec, lw=1, alpha=alpha, zorder=1)
        ax.add_patch(poly)

    def draw_bottle_strut(ax, x1, y1, x2, y2, w_end, fc, ec, alpha=0.5, bulge=1.25):
        """병모양 스트럿: 양 끝은 w_end, 가운데는 w_end * bulge로 불룩."""
        ddx, ddy = x2 - x1, y2 - y1
        l = math.sqrt(ddx**2 + ddy**2)
        if l < 1: return
        px, py = -ddy / l, ddx / l  # 수직 방향
        ax_d, ay_d = ddx / l, ddy / l  # 부재 방향

        w_mid = w_end * bulge
        n_pts = 10
        top_pts = []
        bot_pts = []
        for i in range(n_pts + 1):
            t = i / n_pts
            cx = x1 + ddx * t
            cy = y1 + ddy * t
            w_here = w_end + (w_mid - w_end) * math.sin(math.pi * t)
            top_pts.append((cx + px * w_here / 2, cy + py * w_here / 2))
            bot_pts.append((cx - px * w_here / 2, cy - py * w_here / 2))

        all_pts = top_pts + bot_pts[::-1]
        poly = Polygon(all_pts, closed=True, fc=fc, ec=ec, lw=1, alpha=alpha, zorder=1)
        ax.add_patch(poly)

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    L, H = beam_length, beam_height
    ax.add_patch(mpatches.Rectangle((0, 0), L, H, lw=1, ec='#d1d5db', fc='#fafafa', alpha=0.3))

    vis_scale = 0.5

    # ── 스트럿 (폭 있는 형태) ──
    for conn in connections:
        n1, n2 = conn[0], conn[1]
        f = get_force(n1, n2)
        x1, y1 = get_coords(n1)
        x2, y2 = get_coords(n2)

        if f < 0:  # 압축
            key = (n1, n2) if (n1, n2) in strut_results else (n2, n1)
            w = strut_results[key]['w_actual'] if key in strut_results else 300
            stype = strut_results[key]['type'] if key in strut_results else 'Prismatic'
            if stype == 'Bottle-shaped':
                draw_bottle_strut(ax, x1, y1, x2, y2, w * vis_scale, '#bfdbfe', '#3b82f6', 0.45)
            else:
                draw_member(ax, x1, y1, x2, y2, w * vis_scale, '#bfdbfe', '#3b82f6', 0.45)
        else:  # 인장
            key_tw = (n1, n2) if (n1, n2) in tie_widths else (n2, n1)
            tw = tie_widths.get(key_tw, wt)
            draw_member(ax, x1, y1, x2, y2, tw * vis_scale, '#fecaca', '#dc2626', 0.35)

    # ── 부재 중심선 + 부재력 ──
    for conn in connections:
        n1, n2 = conn[0], conn[1]
        f = get_force(n1, n2)
        x1, y1 = get_coords(n1)
        x2, y2 = get_coords(n2)
        color = '#dc2626' if f > 0 else '#1d4ed8'
        ls = '--' if f > 0 else '-'
        ax.plot([x1, x2], [y1, y2], ls, color=color, lw=1.5, zorder=2, alpha=0.7)

        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        dx, dy = x2 - x1, y2 - y1
        l = math.sqrt(dx**2 + dy**2)
        if l > 1:
            px, py = -dy / l * 80, dx / l * 80
            rot = math.degrees(math.atan2(dy, dx))
            if rot > 90: rot -= 180
            if rot < -90: rot += 180
            f_kN = abs(f) / 1000
            ax.text(mx + px, my + py, f'{f_kN:.0f} kN', ha='center', va='center',
                    fontsize=10, color=color, fontweight='bold', rotation=rot,
                    bbox=dict(fc='white', ec='none', alpha=0.85, pad=2))

    # ── 노드 ──
    zone_c = {'CCC': '#3b82f6', 'CCT': '#f59e0b', 'CTT': '#ef4444'}
    mid_x = beam_length / 2

    for nid in nodes:
        x, y = get_coords(nid)
        ntype = node_types[nid]['type']
        color = zone_c[ntype]
        nv = design_result['node_verification'][nid]
        status = 'OK' if nv['all_ok'] else 'NG'
        status_color = '#16a34a' if nv['all_ok'] else '#dc2626'

        ax.plot(x, y, 'o', ms=14, color=color, mec='black', mew=2, zorder=5)

        if y > H / 2:
            oy = 280
            if x < mid_x * 0.5:       ox = -200
            elif x > mid_x * 1.5:     ox = 200
            else:                      ox = 0
        else:
            oy = -280
            if x < mid_x * 0.5:       ox = -200
            elif x > mid_x * 1.5:     ox = 200
            else:                      ox = 0

        ax.annotate(f'{nid}\n{ntype}\n[{status}]',
                    xy=(x, y), xytext=(x + ox, y + oy),
                    fontsize=11, fontweight='bold', color='black',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=status_color,
                              alpha=0.95, lw=2),
                    arrowprops=dict(arrowstyle='->', color=status_color, lw=1.5),
                    zorder=6)

    # ── 하중/반력 ──
    if loads:
        for lx, lfy in loads:
            ax.annotate('', xy=(lx, H - 20), xytext=(lx, H + 100),
                        arrowprops=dict(arrowstyle='->', color='#991b1b', lw=2.5))
            ax.text(lx, H + 120, f'{abs(lfy):,.0f} kN', ha='center',
                    fontsize=12, fontweight='bold', color='#991b1b')

    for nid, stype in supports.items():
        x, y = get_coords(nid)
        if nid in reactions:
            Rx, Ry = reactions[nid]
            ax.annotate('', xy=(x, 20), xytext=(x, -80),
                        arrowprops=dict(arrowstyle='->', color='#1e40af', lw=2.5))
            ax.text(x, -100, f'{abs(Ry)/1000:,.0f} kN', ha='center',
                    fontsize=9, fontweight='bold', color='#1e40af')
        m = '^' if stype == 'pin' else 'o'
        ax.plot(x, -5, m, ms=14, color='#f59e0b', mec='#92400e', mew=2, zorder=5)

    # ── 범례 ──
    lp = [
        mpatches.Patch(fc='#bfdbfe', ec='#3b82f6', label='Prismatic Strut (βs=1.0)', alpha=0.45),
        mpatches.Patch(fc='#bfdbfe', ec='#3b82f6', label='Bottle-shaped Strut (βs=0.6)', alpha=0.45),
        mpatches.Patch(fc='#fecaca', ec='#dc2626', label='Tie (tension)', alpha=0.35),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#3b82f6',
                   ms=10, mec='black', label='CCC (βn=1.0)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#f59e0b',
                   ms=10, mec='black', label='CCT (βn=0.8)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ef4444',
                   ms=10, mec='black', label='CTT (βn=0.6)'),
    ]
    ax.legend(handles=lp, loc='center', fontsize=8, framealpha=0.95)

    ax.set_xlim(-400, L + 400)
    ax.set_ylim(-400, H + 400)
    ax.set_aspect('equal')
    ax.set_xlabel('X (mm)', fontsize=11)
    ax.set_ylabel('Y (mm)', fontsize=11)
    ax.set_title(title or 'STM with Nodal Zones (KDS 14 20 24)',
                 fontsize=13, fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  >> Saved: {save_path}")
    plt.close()
    return fig


def plot_node_shapes(nodes, connections, member_forces, design_result,
                     title=None, save_path=None):
    """
    각 노드의 절점 영역 기하학적 형상 개별 시각화 (KDS 14 20 24).

    CCC → 육각형 절점 영역
    CCT → 삼각형 절점 영역 (lb 하단, wt 측면, wsb 대각)
    CTT → 평행사변형 절점 영역

    Args:
        nodes: dict {'A': [x, y], ...}
        connections: list [['A','B'], ...]
        member_forces: dict {(n1, n2): force_N, ...}
        design_result: design_nodal_zones() 반환값
        title: 그림 제목
        save_path: 저장 경로 (None이면 저장 안 함)
    """
    node_types    = design_result['node_types']
    node_verif    = design_result['node_verification']
    bearing_plate = design_result['bearing_plate']
    wt            = design_result['wt']

    def get_force(n1, n2):
        if (n1, n2) in member_forces: return member_forces[(n1, n2)]
        if (n2, n1) in member_forces: return member_forces[(n2, n1)]
        return 0

    def get_coords(nid):
        c = nodes[nid]
        return (c[0], c[1])

    # ── 각 노드별 연결 부재 & 방향 각도 수집 ──
    node_members = {nid: [] for nid in nodes}
    for conn in connections:
        n1, n2 = conn[0], conn[1]
        f = get_force(n1, n2)
        x1, y1 = get_coords(n1)
        x2, y2 = get_coords(n2)
        node_members[n1].append({
            'other': n2, 'force': f,
            'angle': math.atan2(y2 - y1, x2 - x1)
        })
        node_members[n2].append({
            'other': n1, 'force': f,
            'angle': math.atan2(y1 - y2, x1 - x2)
        })

    # ── 레이아웃 ──
    n     = len(nodes)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4.5 * ncols, 4.5 * nrows),
                              squeeze=False)

    for idx, nid in enumerate(sorted(nodes.keys())):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        ntype  = node_types[nid]['type']
        bn     = node_types[nid]['beta_n']
        nv     = node_verif[nid]
        all_ok = nv['all_ok']

        L  = 130   # 화살표 길이 (도면 단위)
        hw = 20    # 부재 밴드 반폭

        # ── 부재 밴드 & 화살표 ──
        for m in node_members[nid]:
            a  = m['angle']
            dx = math.cos(a) * L
            dy = math.sin(a) * L
            px = -math.sin(a) * hw
            py =  math.cos(a) * hw

            is_tie  = m['force'] > 0
            fc_band = '#FCEBEB' if is_tie else '#E6F1FB'
            ec_band = '#F09595' if is_tie else '#85B7EB'
            arr_col = '#A32D2D' if is_tie else '#185FA5'

            # 밴드 (사다리꼴)
            band = Polygon([
                (-px, -py), (px, py),
                (dx + px, dy + py), (dx - px, dy - py)
            ], closed=True, facecolor=fc_band, edgecolor=ec_band,
               linewidth=0.5, alpha=0.7, zorder=1)
            ax.add_patch(band)

            # 화살표 (절점 방향으로 압축/인장 표시)
            ax.annotate('',
                        xy=(dx * 0.85, dy * 0.85),
                        xytext=(dx * 0.45, dy * 0.45),
                        arrowprops=dict(arrowstyle='->', color=arr_col, lw=2.0),
                        zorder=3)

            # 부재 이름 라벨
            mname = (f"{nid}{m['other']}" if nid < m['other']
                     else f"{m['other']}{nid}")
            ax.text(dx * 1.18, dy * 1.18, mname,
                    ha='center', va='center', fontsize=7.5,
                    color=arr_col, fontweight='bold')

        # ── 절점 영역 다각형 ──
        zone_fc = '#B5D4F4' if ntype != 'CTT' else '#F7C1C1'
        zone_ec = '#185FA5' if ntype != 'CTT' else '#A32D2D'

        if ntype == 'CCC':
            # 정육각형
            r   = 32
            pts = [
                (r * math.cos(math.pi / 6 + i * math.pi / 3),
                 r * math.sin(math.pi / 6 + i * math.pi / 3))
                for i in range(6)
            ]

        elif ntype == 'CCT':
            # 직각삼각형: 하단=lb(지압판), 측면=wt(타이), 빗변=wsb(스트럿 면)
            lb_px = min(bearing_plate / 10, 55)
            wt_px = min(wt / 10, 40)
            pts = [
                (-lb_px / 2, -wt_px / 2),
                ( lb_px / 2, -wt_px / 2),
                (-lb_px / 2,  wt_px / 2),
            ]

        else:  # CTT
            # 평행사변형: 양쪽 타이 + 상단 스트럿
            lb_px = min(bearing_plate / 12, 45)
            wt_px = min(wt / 10, 38)
            shift = 12
            pts = [
                (-lb_px / 2,         -wt_px / 2),
                ( lb_px / 2,         -wt_px / 2),
                ( lb_px / 2 + shift,  wt_px / 2),
                (-lb_px / 2 + shift,  wt_px / 2),
            ]

        ax.add_patch(Polygon(pts, closed=True,
                             facecolor=zone_fc, edgecolor=zone_ec,
                             linewidth=1.5, zorder=4))

        # ── 절점 이름 & 타입 라벨 ──
        lbl_col = '#0C447C' if ntype != 'CTT' else '#791F1F'
        ax.text(0, 0, f"{nid}  {ntype}\nβn = {bn:.1f}",
                ha='center', va='center', fontsize=9,
                fontweight='bold', color=lbl_col, zorder=5)

        # ── PASS / FAIL ──
        s_str = 'PASS' if all_ok else 'FAIL'
        s_col = '#16a34a' if all_ok else '#dc2626'
        ax.text(0, -162, s_str,
                ha='center', va='bottom', fontsize=10,
                fontweight='bold', color=s_col)

        # ── 가장 critical한 부재의 w_req / w_act 표시 ──
        if nv['checks']:
            worst = max(nv['checks'],
                        key=lambda c: c['w_required'] / max(c['w_actual'], 1))
            w_color = '#dc2626' if not worst['ok'] else '#5F5E5A'
            ax.text(0, 158,
                    f"w_req={worst['w_required']:.0f}\nw_act={worst['w_actual']:.0f}",
                    ha='center', va='top', fontsize=7.5, color=w_color)

        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.set_aspect('equal')
        ax.axis('off')

    # ── 빈 subplot 숨기기 ──
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    # ── 전체 범례 ──
    legend_handles = [
        mpatches.Patch(facecolor='#E6F1FB', edgecolor='#85B7EB', label='스트럿 (압축)'),
        mpatches.Patch(facecolor='#FCEBEB', edgecolor='#F09595', label='타이 (인장)'),
        mpatches.Patch(facecolor='#B5D4F4', edgecolor='#185FA5', label='CCC / CCT 절점'),
        mpatches.Patch(facecolor='#F7C1C1', edgecolor='#A32D2D', label='CTT 절점'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=4,
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(title or '절점 형상 (KDS 14 20 24)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  >> Saved: {save_path}")
    plt.close()
    return fig


# ═══════════════════════════════════════════════════
# TEST: 교과서 예제 10.2 (STM-1)
# ═══════════════════════════════════════════════════
if __name__ == "__main__":
    y_bot, y_top = 125, 1850
    nodes = {
        'A': [225, y_bot], 'B': [1225, y_top], 'C': [2225, y_top],
        'D': [4675, y_top], 'E': [5675, y_top], 'F': [6675, y_bot],
        'G': [1225, y_bot], 'H': [5675, y_bot],
    }
    connections = [
        ['A','B'], ['E','F'], ['B','C'], ['D','E'], ['C','D'],
        ['A','G'], ['H','F'], ['G','H'],
        ['B','G'], ['E','H'], ['C','G'], ['D','H'],
    ]
    member_forces = {
        ('A','B'): -2311.7e3, ('E','F'): -2311.7e3,
        ('B','C'): -1159.3e3, ('D','E'): -1159.3e3, ('C','D'): -2318.6e3,
        ('A','G'): 1159.3e3, ('H','F'): 1159.3e3, ('G','H'): 2318.6e3,
        ('B','G'): 2000.0e3, ('E','H'): 2000.0e3,
        ('C','G'): -2311.7e3, ('D','H'): -2311.7e3,
    }
    reactions = {'A': (0, 2000.0e3), 'F': (0, 2000.0e3)}
    supports = {'A': 'pin', 'F': 'roller'}
    loads = [(2225, -2000), (4675, -2000)]

    print("="*60)
    print("  KDS Example 10.2 — Automated Nodal Zone Design")
    print("="*60)

    result = design_nodal_zones(
        nodes, connections, member_forces, reactions, supports,
        fck=27, fy=400, bw=500, bearing_plate=450,
        cover=40, stirrup_dia=16, main_bar_dia=32,
        beam_height=2000
    )

    plot_stm_with_nodal_zones(
        nodes, connections, member_forces, reactions, supports,
        result, beam_length=6900, beam_height=2000,
        loads=loads,
        save_path='/home/claude/stm_nodal_zones_auto.png',
        title='STM-1 with Nodal Zones — Automated (KDS 14 20 24 예제 10.2)'
    )

    plot_node_shapes(
        nodes, connections, member_forces, result,
        title='절점 형상 — KDS 예제 10.2 (STM-1)',
        save_path='/home/claude/node_shapes_example.png'
    )
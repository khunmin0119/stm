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
            phi=phi, beam_height=H, verbose=verbose
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
                        beam_height=None, verbose=True):
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
    if beam_height is None:
        beam_height = max(all_y) - min(all_y) + 2 * cover

    # 유효깊이 & 타이 폭
    d = beam_height - cover - stirrup_dia - main_bar_dia - main_bar_dia / 2
    wt = 2 * (cover + stirrup_dia + main_bar_dia + main_bar_dia / 2)
    wt = max(wt, 250)

    if verbose:
        print(f"\n  [Nodal Zone Design — KDS 14 20 24]")
        print(f"    fck={fck}MPa, fy={fy}MPa, bw={bw}mm, φ={phi}")
        print(f"    Bearing plate={bearing_plate}mm, cover={cover}mm")
        print(f"    d={d:.1f}mm, wt={wt:.1f}mm")

    # ── Helper: force lookup ──
    def get_force(n1, n2):
        if (n1, n2) in member_forces: return member_forces[(n1, n2)]
        if (n2, n1) in member_forces: return member_forces[(n2, n1)]
        return 0

    def get_coords(nid):
        c = nodes[nid]
        return (c[0], c[1]) if isinstance(c, (list, tuple)) else (c[0], c[1])

    # ── Helper: 수직 타이 폭 자동 계산 ──
    def compute_vertical_tie_width(nid_top, nid_bot):
        """수직 타이의 폭 = 같은 높이의 인접 노드까지 최소 거리."""
        tx, ty = get_coords(nid_top)
        bx, by = get_coords(nid_bot)
        
        # 두 노드 각각에서 같은 높이의 인접 노드 거리 계산
        min_dist = float('inf')
        for nid in nodes:
            if nid == nid_top and nid == nid_bot:
                continue
            nx, ny = get_coords(nid)
            # 상단 노드와 같은 높이
            if abs(ny - ty) < 50 and nid != nid_top:
                dist = abs(nx - tx)
                if 50 < dist < min_dist:
                    min_dist = dist
            # 하단 노드와 같은 높이
            if abs(ny - by) < 50 and nid != nid_bot:
                dist = abs(nx - bx)
                if 50 < dist < min_dist:
                    min_dist = dist
        
        return min_dist if min_dist < float('inf') else wt

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
        else:  # 수평/기타 타이
            tie_widths[(n1, n2)] = wt
            tie_widths[(n2, n1)] = wt

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

        # 실제 스트럿 폭
        if dy > 50 and dx > 50:  # 대각선
            # 지지점/하중점 근처: lb 사용
            # 판단: 연결된 노드 중 지지점 또는 하중점(반력 있는 곳)에 가까우면 lb
            lb_eff = bearing_plate
            # 하중점 간 대각선이면 하중 간격 사용
            n1_has_reaction = n1 in reactions or n1 in supports
            n2_has_reaction = n2 in reactions or n2 in supports
            if not n1_has_reaction and not n2_has_reaction:
                # 두 노드 모두 반력 없음 → 하중 간격 추정
                lb_eff = dx  # 수평 거리를 지압 길이로 사용
            wsb = lb_eff * math.sin(angle) + wt * math.cos(angle)
        else:
            # 수평 스트럿: 등가응력블록에서 결정
            # ws ≈ a (등가응력블록 깊이) 또는 최소 ws
            wsb = max(d * 0.15, 300)  # 기본값, 이후 정밀 계산 가능

        # 요구 스트럿 폭
        w_s_req = (Fu * 1e3) / (phi * 0.85 * beta_s * fck * bw)

        strut_results[(n1, n2)] = {
            'force_kN': Fu, 'beta_s': beta_s, 'type': stype,
            'angle_deg': math.degrees(angle), 'w_actual': wsb,
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

            # 실제 폭 결정
            dx, dy = abs(m['dx']), abs(m['dy'])
            if m['force'] > 0:  # 타이
                key_tw = (n1, n2) if (n1, n2) in tie_widths else (n2, n1)
                w_act = tie_widths.get(key_tw, wt)
            else:  # 스트럿
                key = (n1, n2) if (n1, n2) in strut_results else (n2, n1)
                if key in strut_results:
                    w_act = strut_results[key]['w_actual']
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
        'd': d, 'wt': wt,
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
        # 10개 점으로 부드러운 곡선
        n_pts = 10
        top_pts = []
        bot_pts = []
        for i in range(n_pts + 1):
            t = i / n_pts  # 0 ~ 1
            # 현재 위치
            cx = x1 + ddx * t
            cy = y1 + ddy * t
            # 폭: 양 끝은 w_end, 가운데는 w_mid (sin 곡선)
            w_here = w_end + (w_mid - w_end) * math.sin(math.pi * t)
            top_pts.append((cx + px * w_here / 2, cy + py * w_here / 2))
            bot_pts.append((cx - px * w_here / 2, cy - py * w_here / 2))

        all_pts = top_pts + bot_pts[::-1]
        poly = Polygon(all_pts, closed=True, fc=fc, ec=ec, lw=1, alpha=alpha, zorder=1)
        ax.add_patch(poly)

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    L, H = beam_length, beam_height
    ax.add_patch(mpatches.Rectangle((0, 0), L, H, lw=1, ec='#d1d5db', fc='#fafafa', alpha=0.3))

    # 시각화용 폭 스케일 (실제 값이 너무 크면 보기 어려우므로 축소)
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

        # 부재력 라벨
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
        
        # 라벨 위치: 보 바깥쪽으로 치우기
        if y > H / 2:  # 상단 노드 → 위로
            oy = 280
            if x < mid_x * 0.5:       ox = -200   # 왼쪽 끝 → 왼쪽으로
            elif x > mid_x * 1.5:     ox = 200    # 오른쪽 끝 → 오른쪽으로
            else:                      ox = 0      # 중간 → 가운데
        else:  # 하단 노드 → 아래로
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
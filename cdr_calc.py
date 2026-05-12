"""
CDR 계산 코드
수정 필요한 부분에 ### TODO 표시
"""
import math
from scipy.optimize import minimize

R = 2000
fck = 27
phi = 0.75
bw = 500
bearing = 450
fcu = 0.85 * fck  # 22.95
H = 2000

def calc_CDR_all(x_mid, y_bot, y_top):
    dy = y_top - y_bot
    wt = 2 * y_bot
    ws = 2 * (H - y_top)
    dx_BA = x_mid - 225
    dx_CG = 2225 - x_mid

    angle = math.atan2(dy, dx_BA)
    sin_a, cos_a = math.sin(angle), math.cos(angle)

    # 부재력 (절점법)
    F_BA = R / sin_a
    F_AG = R * cos_a / sin_a
    F_BG = 2000.0
    F_CG = F_BA
    F_GH = 2 * F_AG
    F_BC = F_AG
    F_CD = F_GH

    # ⑤ 수직 타이 폭 — 양옆 노드 x 정사영 중점 합산
    # 교과서 8노드 (A=225, x_mid, 하중점=2225, 4675, 5450 대칭, D=6675)
    # x_mid가 좌측 절반(<3450)이면 양옆 = pin(225)과 하중점(2225)
    # x=1225일 때: (1225-225)/2 + (2225-1225)/2 = 500 + 500 = 1000mm
    if x_mid < 3450:
        w_vert = (x_mid - 225) / 2 + (2225 - x_mid) / 2
    else:
        w_vert = (x_mid - 4675) / 2 + (6675 - x_mid) / 2

    # ④ 대각선 실제 폭 — 노드별 정사영 (w_h × cos + w_v × sin)
    # A노드(지점): nodal zone = bearing(수평) × wt(수직)
    #   → wsb_A = bearing × sin(θ) + wt × cos(θ)
    # G노드(중간): nodal zone = w_vert(양옆 합산) × wt(수직)
    #   → wsb_G = w_vert × sin(θ) + wt × cos(θ)
    w_diag_A = bearing * sin_a + wt * cos_a   # A노드 시점 (지점)
    w_diag_G = w_vert * sin_a + wt * cos_a    # G노드 시점 (중간)

    cdrs = []
    labels = []

    # Node A (CCT, βn=0.8)
    fc = phi * 0.8 * fcu
    cdrs.append(bearing / (R*1000/(fc*bw)))
    labels.append(('A', 'R (반력)', R, R*1000/(fc*bw), bearing))

    cdrs.append(w_diag_A / (F_BA*1000/(fc*bw)))
    labels.append(('A', 'AB (대각)', F_BA, F_BA*1000/(fc*bw), w_diag_A))

    cdrs.append(wt / (F_AG*1000/(fc*bw)))
    labels.append(('A', 'AG (하단)', F_AG, F_AG*1000/(fc*bw), wt))

    # Node B (CCT, βn=0.8)
    cdrs.append(w_diag_A / (F_BA*1000/(fc*bw)))
    labels.append(('B', 'AB (대각)', F_BA, F_BA*1000/(fc*bw), w_diag_A))

    cdrs.append(ws / (F_BC*1000/(fc*bw)))
    labels.append(('B', 'BC (상단)', F_BC, F_BC*1000/(fc*bw), ws))

    cdrs.append(w_vert / (F_BG*1000/(fc*bw)))
    labels.append(('B', 'BG (수직)', F_BG, F_BG*1000/(fc*bw), w_vert))

    # Node C (CCC, βn=1.0)
    fc = phi * 1.0 * fcu
    cdrs.append(bearing / (R*1000/(fc*bw)))
    labels.append(('C', 'P (하중)', R, R*1000/(fc*bw), bearing))

    cdrs.append(ws / (F_BC*1000/(fc*bw)))
    labels.append(('C', 'BC (상단)', F_BC, F_BC*1000/(fc*bw), ws))

    cdrs.append(ws / (F_CD*1000/(fc*bw)))
    labels.append(('C', 'CD (상단)', F_CD, F_CD*1000/(fc*bw), ws))

    cdrs.append(w_diag_G / (F_CG*1000/(fc*bw)))
    labels.append(('C', 'CG (대각)', F_CG, F_CG*1000/(fc*bw), w_diag_G))

    # Node G (CTT, βn=0.6)
    fc = phi * 0.6 * fcu
    cdrs.append(wt / (F_AG*1000/(fc*bw)))
    labels.append(('G', 'AG (하단)', F_AG, F_AG*1000/(fc*bw), wt))

    cdrs.append(wt / (F_GH*1000/(fc*bw)))
    labels.append(('G', 'GH (하단)', F_GH, F_GH*1000/(fc*bw), wt))

    cdrs.append(w_vert / (F_BG*1000/(fc*bw)))
    labels.append(('G', 'BG (수직)', F_BG, F_BG*1000/(fc*bw), w_vert))

    cdrs.append(w_diag_G / (F_CG*1000/(fc*bw)))
    labels.append(('G', 'CG (대각)', F_CG, F_CG*1000/(fc*bw), w_diag_G))

    return cdrs, labels


# NM 목적함수: CDR>=1.0 유지하면서 1.0에 가깝게
def objective(v):
    x, yb, yt = v
    if yb < 72 or yb > 500 or yt < 1500 or yt > 1928:
        return 1e6
    if x < 300 or x > 2100 or yt - yb < 800:
        return 1e6
    result = calc_CDR_all(x, yb, yt)
    if result is None:
        return 1e6
    cdrs, _ = result
    penalty = sum(100 * (1.0 - c)**2 for c in cdrs if c < 1.0)
    deviation = sum((c - 1.0)**2 for c in cdrs if c >= 1.0)
    return penalty + deviation


if __name__ == '__main__':
    # 교과서 결과
    cdrs, labels = calc_CDR_all(1225, 125, 1850)
    print("=== 교과서 ===")
    print(f"{'노드':<4} {'부재명':<12} {'부재력':>10} {'요구폭':>10} {'실제폭':>10} {'CDR':>6}")
    for i, (node, name, F, w_req, w_act) in enumerate(labels):
        mark = ' FAIL' if cdrs[i] < 1.0 else ''
        print(f"{node:<4} {name:<12} {F:>10.1f} {w_req:>10.1f} {w_act:>10.1f} {cdrs[i]:>6.3f}{mark}")

    # NM 최적화
    res = minimize(objective, [1225, 200, 1850], method='Nelder-Mead',
                   options={'maxiter': 10000, 'xatol': 0.5, 'fatol': 1e-9})
    x, yb, yt = int(round(res.x[0])), round(res.x[1],1), round(res.x[2],1)
    cdrs, labels = calc_CDR_all(x, yb, yt)
    print(f"\n=== NM 최적: x={x}, y_bot={yb}, y_top={yt} ===")
    print(f"{'노드':<4} {'부재명':<12} {'부재력':>10} {'요구폭':>10} {'실제폭':>10} {'CDR':>6}")
    for i, (node, name, F, w_req, w_act) in enumerate(labels):
        mark = ' FAIL' if cdrs[i] < 1.0 else ''
        print(f"{node:<4} {name:<12} {F:>10.1f} {w_req:>10.1f} {w_act:>10.1f} {cdrs[i]:>6.3f}{mark}")

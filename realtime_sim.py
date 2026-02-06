"""
Real_test_data 실시간 진단 시뮬레이터

CompetitionSystem을 재사용하여 훈련과 동일한
피처 삭제/재정렬/전처리 파이프라인을 보장한다.

실행하면 9개 시나리오 목록이 뜨고, 번호를 선택하면
해당 시나리오를 1초씩 읽으며 매 초 진단 로그를 출력합니다.

사용법:
  python realtime_sim.py                  # 대화형 (시나리오 선택)
  python realtime_sim.py --speed 0        # 딜레이 없이 즉시
  python realtime_sim.py --speed 1        # 실시간 1초 간격
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from practice.dataloader import LABELS, ID2LABEL, LABEL2ID
from competition_interface import CompetitionSystem, find_valid_models


# ─────────────────────────────────────────────
# ANSI 색상
# ─────────────────────────────────────────────
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    GREEN   = "\033[92m"
    RED     = "\033[91m"
    YELLOW  = "\033[93m"
    CYAN    = "\033[96m"
    MAGENTA = "\033[95m"
    WHITE   = "\033[97m"
    DIM     = "\033[2m"
    BG_RED  = "\033[41m"
    BG_GREEN = "\033[42m"


def color_by_class(name):
    if name == "NORMAL":
        return C.GREEN
    elif "LOCA" in name:
        return C.RED
    elif "SGTR" in name:
        return C.MAGENTA
    elif "ESDE" in name:
        return C.YELLOW
    return C.WHITE


def extract_ground_truth(filename):
    """파일명에서 정답 레이블 및 dt(사고시점) 추출"""
    basename = os.path.splitext(os.path.basename(filename))[0]

    if "NORMAL" in basename:
        return "NORMAL", None

    dt = None
    if "_dt=" in basename:
        dt = int(basename.split("_dt=")[-1])

    for label in LABELS:
        if label == "NORMAL":
            continue
        if basename.startswith(label):
            return label, dt

    return "UNKNOWN", dt


def select_model():
    """models/ 에서 유효한 모델만 선택 (._* 제외, 아티팩트 완전성 검증)"""
    candidates = find_valid_models("models")
    if not candidates:
        print("  models/ 디렉토리에 유효한 모델이 없습니다.")
        print("  (아티팩트 3종 __model.keras + __feature_transformer.pkl + __preprocessing_metadata.json 필요)")
        sys.exit(1)

    print(f"\n  {C.BOLD}사용할 모델을 선택하세요:{C.RESET}\n")
    for i, c in enumerate(candidates):
        name = os.path.basename(c)
        model_type = name.split("__")[0].upper()
        print(f"    {C.CYAN}[{i+1}]{C.RESET} {model_type:15s}  {C.DIM}{name}{C.RESET}")

    print()
    while True:
        try:
            choice = input(f"  번호 입력 (기본=1): ").strip()
            if choice == "":
                idx = 0
            else:
                idx = int(choice) - 1
            if 0 <= idx < len(candidates):
                return candidates[idx]
        except (ValueError, KeyboardInterrupt):
            pass
        print(f"  1~{len(candidates)} 사이 번호를 입력하세요.")


def select_scenario(data_dir):
    """시나리오(CSV) 선택 메뉴"""
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files:
        print(f"  {data_dir}에 CSV 파일이 없습니다.")
        sys.exit(1)

    print(f"\n  {C.BOLD}진단할 시나리오를 선택하세요:{C.RESET}\n")

    for i, f in enumerate(csv_files):
        basename = os.path.basename(f)
        ground_truth, dt = extract_ground_truth(f)
        gt_color = color_by_class(ground_truth)

        if dt is not None:
            info = f"사고시점={dt}s"
        else:
            info = "전구간 정상"

        print(f"    {C.CYAN}[{i+1}]{C.RESET}  {gt_color}{C.BOLD}{ground_truth:12s}{C.RESET}  "
              f"{C.DIM}({info}){C.RESET}")
        print(f"         {C.DIM}{basename}{C.RESET}")

    print()
    while True:
        try:
            choice = input(f"  번호 입력: ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(csv_files):
                return csv_files[idx]
        except (ValueError, KeyboardInterrupt):
            print()
            sys.exit(0)
        print(f"  1~{len(csv_files)} 사이 번호를 입력하세요.")


def run_simulation(system, csv_path, window_size=3, speed=0.3):
    """
    단일 CSV 실시간 시뮬레이션.
    CompetitionSystem.receive_and_predict()를 매 행마다 호출하여
    훈련과 동일한 파이프라인으로 추론한다.
    """
    ground_truth, dt = extract_ground_truth(csv_path)
    basename = os.path.basename(csv_path)

    # CSV 로드
    df = pd.read_csv(csv_path)
    if "KCNTOMS" in df.columns:
        df = df.drop(columns=["KCNTOMS"])

    # CompetitionSystem에 피처 매핑 설정 (삭제 + 재정렬 보장)
    system.set_feature_names(list(df.columns))

    total_rows = len(df)
    n_input = len(system.preprocessing.feature_names_in)
    n_output = len(system.preprocessing.feature_names_out)

    # 헤더
    print(f"\n{'='*80}")
    print(f"  {C.BOLD}  NAS 실시간 진단 시뮬레이션{C.RESET}")
    print(f"{'='*80}")
    print(f"  시나리오  : {C.BOLD}{basename}{C.RESET}")
    if dt is not None:
        print(f"  정답      : {C.BOLD}{color_by_class(ground_truth)}{ground_truth}{C.RESET}  (사고시점 dt={dt}초)")
    else:
        print(f"  정답      : {C.BOLD}{C.GREEN}NORMAL (전구간){C.RESET}")
    print(f"  데이터    : {total_rows}초 분량  |  CSV={len(df.columns)}컬럼 → 매핑={n_input} → physics → {n_output}피처 → {window_size}초 윈도우")
    print(f"  출력 속도 : {speed}초 간격")
    print(f"{'='*80}")

    print()
    print(f"  {C.DIM}{'시간':>6s}  {'진단결과':<15s}  {'확신도':>8s}  {'1-2위 마진':>10s}  {'상태':<20s}{C.RESET}")
    print(f"  {'─'*70}")

    results_log = []

    # 새 시나리오 시작 시 CompetitionSystem 상태 리셋
    system.reset()

    for i in range(total_rows):
        step = i + 1

        # 매 행을 receive_and_predict에 전달 (피처 삭제/재정렬은 내부에서 처리)
        row = df.iloc[i].values
        result = system.receive_and_predict(row)

        probs = result['Class probabilities']
        results_log.append(result)

        # ─── 로그 한 줄 출력 ───
        res_name = result['results']
        res_color = color_by_class(res_name)
        max_prob = max(probs) if any(p > 0 for p in probs) else 0

        if any(p > 0 for p in probs):
            sp = sorted(probs, reverse=True)
            margin_val = sp[0] - sp[1] if len(sp) > 1 else sp[0]
        else:
            margin_val = 0

        # 상태 메시지
        if result['Diagnostic_time'] is not None:
            if result['Diagnostic_time'] == step:
                status_msg = f"{C.BG_GREEN}{C.WHITE} ✅ 확정! {C.RESET}"
            else:
                status_msg = f"{C.BG_GREEN}{C.WHITE} 확정됨 ({result['Diagnostic_time']:.0f}s) {C.RESET}"
        elif step < window_size:
            status_msg = f"{C.DIM}윈도우 축적 중 ({step}/{window_size}){C.RESET}"
        else:
            status_msg = ""

        extra = ""
        if dt is not None and step == dt:
            extra = f"  {C.BG_RED}{C.WHITE} ⚡ 사고발생! {C.RESET}"

        print(f"  {C.BOLD}[{step:3d}s]{C.RESET}  "
              f"{res_color}{C.BOLD}{res_name:15s}{C.RESET}  "
              f"{max_prob*100:6.2f}%    "
              f"{margin_val*100:6.2f}%      "
              f"{status_msg}{extra}")

        # 확률 상세 출력 (확정 시점 + 사고 시점)
        if (result['Diagnostic_time'] is not None and result['Diagnostic_time'] == step) or \
           (dt is not None and step == dt):
            print()
            for j, (label, prob) in enumerate(zip(LABELS, probs)):
                bar_len = int(prob * 30)
                bar = "█" * bar_len + "░" * (30 - bar_len)
                lbl_color = color_by_class(label)
                if j == int(np.argmax(probs)):
                    print(f"         {C.BOLD}{lbl_color}{label:12s}{C.RESET} {bar} {prob*100:6.2f}%")
                else:
                    print(f"         {C.DIM}{label:12s} {bar} {prob*100:6.2f}%{C.RESET}")
            print()

        if speed > 0:
            time.sleep(speed)

    # ─── 최종 결과 ───
    final = results_log[-1]
    print(f"\n{'='*80}")
    print(f"  {C.BOLD}최종 반환 결과 (팀 → 멘토){C.RESET}")
    print(f"{'='*80}")
    print(f"  {{")
    print(f"      'results': {C.BOLD}{color_by_class(final['results'])}{final['results']!r}{C.RESET},")
    print(f"      'Diagnostic_time': {C.CYAN}{final['Diagnostic_time']}{C.RESET},")
    probs_rounded = [round(p, 6) for p in final['Class probabilities']]
    print(f"      'Class probabilities': {probs_rounded}")
    print(f"  }}")

    # 정오답 판정
    correct = (final['results'] == ground_truth) if ground_truth != "NORMAL" else (final['results'] == "NORMAL")
    print()
    if ground_truth != "NORMAL":
        if final['Diagnostic_time'] is not None:
            reaction = final['Diagnostic_time'] - dt if dt else None
            mark = "✅ 정답!" if correct else "❌ 오답"
            print(f"  정답: {ground_truth}  →  예측: {final['results']}  {mark}")
            if reaction is not None:
                print(f"  사고 후 {C.CYAN}{C.BOLD}{reaction:.0f}초{C.RESET} 만에 진단 완료")
        else:
            print(f"  정답: {ground_truth}  →  예측: {final['results']}  ⚠️ 미확정 (시간 내 확정 실패)")
    else:
        stayed = all(r['results'] == 'NORMAL' for r in results_log[window_size:])
        print(f"  정답: NORMAL  →  {'✅ NORMAL 유지' if stayed else '❌ 오탐 발생'}")

    print(f"{'='*80}\n")

    return final


def main():
    parser = argparse.ArgumentParser(description="NAS 실시간 진단 시뮬레이터")
    parser.add_argument("--data_dir", type=str, default="data/Real_test_data")
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--speed", type=float, default=0.3,
                        help="출력 간격(초). 0=즉시, 1=실시간 (default: 0.3)")
    args = parser.parse_args()

    print(f"\n{C.BOLD}{'='*80}{C.RESET}")
    print(f"  {C.BOLD}🔬 NAS 원전 사고 실시간 진단 시뮬레이터{C.RESET}")
    print(f"{C.BOLD}{'='*80}{C.RESET}")

    # Step 1: 모델 선택 (._* 제외, 아티팩트 완전성 검증)
    model_path = select_model()
    model_name = os.path.basename(model_path).split("__")[0].upper()
    print(f"\n  → 선택: {C.BOLD}{model_name}{C.RESET}  ({os.path.basename(model_path)})")

    # Step 2: CompetitionSystem으로 모델 + 파이프라인 로드
    print(f"\n  모델 로딩 중...")
    system = CompetitionSystem(model_path=model_path, window_size=args.window)
    n_in = len(system.preprocessing.feature_names_in)
    n_out = len(system.preprocessing.feature_names_out)
    print(f"  ✅ 로드 완료 (입력 피처: {n_in}개 → 전처리 후: {n_out}개)")

    # Step 3: 시나리오 선택
    csv_path = select_scenario(args.data_dir)
    ground_truth, dt = extract_ground_truth(csv_path)
    print(f"\n  → 선택: {C.BOLD}{color_by_class(ground_truth)}{ground_truth}{C.RESET}  "
          f"({os.path.basename(csv_path)})")

    # Step 4: 시뮬레이션 시작
    input(f"\n  {C.BOLD}Enter 키를 누르면 시뮬레이션을 시작합니다...{C.RESET}")

    run_simulation(system, csv_path,
                   window_size=args.window, speed=args.speed)

    # 다시 할지 물어보기
    while True:
        again = input(f"  다른 시나리오도 테스트하시겠습니까? (y/n): ").strip().lower()
        if again in ('y', 'yes', 'ㅛ'):
            csv_path = select_scenario(args.data_dir)
            input(f"\n  {C.BOLD}Enter 키를 누르면 시작...{C.RESET}")
            run_simulation(system, csv_path,
                           window_size=args.window, speed=args.speed)
        else:
            print(f"\n  시뮬레이터를 종료합니다.\n")
            break


if __name__ == "__main__":
    main()

"""
수집된 선호 데이터에서 평가 지표를 뽑는다.

README의 "어떻게 평가하는가"에 적은 항목을 그대로 계산한다.
파싱 성공률, 경고 유형별 빈도, 행동 분포, 거리 구간별 선택.

주의: 이 파일이 읽는 jsonl에는 폐기 규칙을 통과한 쌍만 들어 있다.
따라서 파싱 성공률은 전체 생성분이 아니라 "적립된 쌍 안의 후보" 기준이다.
전체 생성분 대비 채택률을 보려면 루프가 돈 횟수를 따로 세야 한다.

사용법:
    python eval_logs.py [경로 ...]
    python eval_logs.py old.jsonl new.jsonl   # 어댑터 교체 전후 비교
"""
import json
import sys
from collections import Counter
from typing import Any, Dict, Iterator, List, Tuple

DEFAULT_PATH = "dpo_preference_data_v2.jsonl"

# README 채점 기준과 같은 경계를 쓴다.
DIST_BINS: List[Tuple[str, float, float]] = [
    ("< 300m", float("-inf"), 300.0),
    ("300~2000m", 300.0, 2000.0),
    ("> 2000m", 2000.0, float("inf")),
]


def load_entries(path: str) -> Iterator[Dict[str, Any]]:
    """깨진 줄은 건너뛰고 개수만 세서 돌려준다."""
    bad = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                bad += 1
    if bad:
        print(f"  [!] 읽지 못한 줄 {bad}개는 건너뜀")


def dist_bin(dist: Any) -> str:
    try:
        d = float(dist)
    except (TypeError, ValueError):
        return "미상"
    for label, lo, hi in DIST_BINS:
        if lo <= d < hi:
            return label
    return "미상"


def pct(part: int, whole: int) -> str:
    if whole == 0:
        return "-"
    return f"{part / whole * 100:.1f}%"


def print_counter(title: str, counter: Counter, total: int, indent: str = "    ") -> None:
    print(f"\n  {title}")
    if not counter:
        print(f"{indent}(없음)")
        return
    width = max(len(str(k)) for k in counter)
    for key, n in counter.most_common():
        print(f"{indent}{str(key):<{width}}  {n:>5}  {pct(n, total)}")


def analyze(path: str) -> None:
    print(f"\n{'=' * 60}\n{path}\n{'=' * 60}")

    pairs = 0
    candidates = 0
    parse_ok = 0
    warn_kinds: Counter = Counter()
    chosen_actions: Counter = Counter()
    rejected_actions: Counter = Counter()
    all_actions: Counter = Counter()
    by_dist: Counter = Counter()
    chosen_by_dist: Counter = Counter()
    margins: List[float] = []
    no_meta = 0

    for entry in load_entries(path):
        pairs += 1
        meta = entry.get("meta") or {}
        if not meta:
            no_meta += 1
            continue

        chosen_actions[meta.get("chosen_action") or "미상"] += 1
        rejected_actions[meta.get("rejected_action") or "미상"] += 1

        bin_label = dist_bin(meta.get("dist_m"))
        by_dist[bin_label] += 1
        chosen_by_dist[(bin_label, meta.get("chosen_action") or "미상")] += 1

        sc, sr = meta.get("score_chosen"), meta.get("score_rejected")
        if isinstance(sc, (int, float)) and isinstance(sr, (int, float)):
            margins.append(float(sc) - float(sr))

        for cand in meta.get("candidates_scored") or []:
            candidates += 1
            if cand.get("parse_ok"):
                parse_ok += 1
            all_actions[cand.get("action") or "미상"] += 1
            for w in cand.get("warnings") or []:
                # "missing:a,b" 는 파라미터별로 쪼개서 센다.
                w = str(w)
                if w.startswith("missing:"):
                    for name in w[len("missing:"):].split(","):
                        if name:
                            warn_kinds[f"missing:{name}"] += 1
                else:
                    warn_kinds[w] += 1

    if pairs == 0:
        print("  적립된 쌍이 없다. sima_app.py를 AUTO로 돌려 데이터를 먼저 모을 것.")
        return

    print(f"\n  선호 쌍          {pairs}")
    print(f"  후보 총계        {candidates}")
    print(f"  파싱 성공률      {pct(parse_ok, candidates)}  ({parse_ok}/{candidates})")
    if margins:
        avg = sum(margins) / len(margins)
        print(f"  점수 차 평균     {avg:.2f}  (최소 {min(margins):.1f} / 최대 {max(margins):.1f})")
    if no_meta:
        print(f"  meta 없는 쌍     {no_meta}")

    print_counter("경고 유형", warn_kinds, candidates)
    print_counter("후보 전체의 행동 분포", all_actions, candidates)
    print_counter("chosen 행동", chosen_actions, pairs)
    print_counter("rejected 행동", rejected_actions, pairs)

    print("\n  거리 구간별 chosen")
    for label, _, _ in DIST_BINS + [("미상", 0.0, 0.0)]:
        total = by_dist.get(label, 0)
        if not total:
            continue
        picks = {a: n for (b, a), n in chosen_by_dist.items() if b == label}
        detail = ", ".join(
            f"{a} {n}" for a, n in sorted(picks.items(), key=lambda kv: -kv[1])
        )
        print(f"    {label:<10}  {total:>5}  |  {detail}")


def main() -> int:
    paths = sys.argv[1:] or [DEFAULT_PATH]
    missing = 0
    for path in paths:
        try:
            analyze(path)
        except FileNotFoundError:
            print(f"\n[Error] 파일이 없다: {path}")
            missing += 1
    print()
    return 1 if missing == len(paths) else 0


if __name__ == "__main__":
    raise SystemExit(main())

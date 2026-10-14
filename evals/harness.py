import json
from collections import defaultdict


def load_goldens(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_by_metric(result):
    scores, passes = defaultdict(list), defaultdict(list)
    for tr in result.test_results:
        for m in tr.metrics_data:
            scores[m.name].append(m.score)
            passes[m.name].append(bool(m.success))
    return {
        name: {
            "avg_score": sum(s) / len(s),
            "pass_rate": sum(passes[name]) / len(passes[name]),
            "n": len(s),
        }
        for name, s in scores.items()
    }


def print_summary(label: str, summary: dict):
    print(f"\n=== {label} ===")
    for name, s in summary.items():
        print(f"{name}: avg={s['avg_score']:.2f}  pass_rate={s['pass_rate']:.0%}  (n={s['n']})")
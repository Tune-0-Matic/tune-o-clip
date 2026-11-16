"""benchmark_winners.py pure logic: pacing from word timings, aggregates, the A-vs-B table and the 14B prompt.
Run: python backend/test_benchmark.py   (no video, no Whisper, no network)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_winners as bw  # noqa: E402

w = lambda word, s, e: {"word": word, "start": s, "end": e}
words = [w("No", 0.3, 0.5), w("way", 0.55, 0.8), w("bro", 1.6, 1.9), w("what", 2.0, 2.2), w("happened", 4.0, 4.6)]
sp = bw.speech(words, 6.0)
assert sp["first_word_ms"] == 300 and sp["words"] == 5 and sp["wpm"] == 50.0, sp
assert sp["gaps"] == 2 and sp["max_gap_s"] == 1.8 and sp["dead_air_s"] == round(0.8 + 1.8 + 0.3, 2), sp  # >0.4 s only
assert sp["tail_after_last_word_s"] == 1.4 and sp["hook_text"] == "No way bro what" and "happened" in sp["last_text"], sp
assert bw.speech([], 10.0)["first_word_ms"] is None and bw.speech([], 10.0)["wpm"] == 0.0

rows = [{"aspect": "720x960", "duration_s": 40, "wpm": 150, "first_word_ms": None},
        {"aspect": "720x1280", "duration_s": 20, "wpm": 170, "first_word_ms": 200},
        {"aspect": "720x960", "duration_s": 60, "wpm": 160, "first_word_ms": 400}]
ag = bw.aggregate(rows)
assert ag["duration_s"]["median"] == 40 and ag["wpm"]["mean"] == 160 and ag["first_word_ms"]["n"] == 2, ag
assert ag["aspects"] == {"720x1280": 1, "720x960": 2} and ag["lufs"] is None

result = {"groups": {"A_winners": {"summary": ag, "clips": [dict(rows[0], hook_text="yo", last_text="bye")]},
                     "B_ours": {"summary": bw.aggregate([dict(rows[1], duration_s=10, wpm=100)]),
                                "clips": [dict(rows[1], hook_text="hi", last_text="ok")]}}}
tb = bw.table(result)
assert "| duration_s | 40 | 10 | -30.00 |" in tb and "| wpm | 160 | 100 | -60.00 |" in tb, tb
pr = bw.prompt(result)
assert "Group A openings" in pr and "'yo'" in pr and "'hi'" in pr and "Rules to adopt" in pr and tb in pr
print("benchmark ok")

import csv
import tempfile
import unittest
from pathlib import Path

from DLC.vllm_throughput import (
    append_throughput_history,
    parse_throughput,
    select_best_total,
)
from DLC.vllm_profile_runner import build_command, load_model_config


class VllmThroughputTests(unittest.TestCase):
    def test_qwen3_path_can_bypass_external_model_registry(self):
        model = load_model_config(
            "/tmp/missing-benchmark-models.json",
            "Qwen3-8B",
            model_path="/mnt/jfs/ci_models/Qwen3-8B",
        )
        command = build_command(model)
        self.assertEqual(command[command.index("--model") + 1],
                         "/mnt/jfs/ci_models/Qwen3-8B")
        self.assertIn("Qwen3-8B", " ".join(command))

    def test_parses_total_and_output_tokens_per_second(self):
        result = parse_throughput(
            "Throughput: 0.25 requests/s, 1160.50 total tokens/s, "
            "580.25 output tokens/s"
        )
        self.assertEqual(result["total_tokens_per_s"], 1160.50)
        self.assertEqual(result["output_tokens_per_s"], 580.25)
        self.assertEqual(result["score"], -1160.50)

    def test_rejects_missing_or_nonfinite_metrics(self):
        for text in (
            "benchmark finished without throughput",
            "Throughput: 1 requests/s, nan total tokens/s, 2 output tokens/s",
            "Throughput: 1 requests/s, 2 total tokens/s, inf output tokens/s",
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_throughput(text)

    def test_selects_highest_total_throughput_and_preserves_output_metric(self):
        records = [
            {"candidate": "baseline", "total_tokens_per_s": 100.0,
             "output_tokens_per_s": 50.0},
            {"candidate": "candidate-a", "total_tokens_per_s": 103.0,
             "output_tokens_per_s": 51.0},
            {"candidate": "candidate-b", "total_tokens_per_s": 101.0,
             "output_tokens_per_s": 52.0},
        ]
        best = select_best_total(records)
        self.assertEqual(best["candidate"], "candidate-a")
        self.assertEqual(best["output_tokens_per_s"], 51.0)

    def test_appends_shareable_throughput_history_with_header(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "throughput_data_qwen3_8b.csv"
            append_throughput_history(path, "2026-09-19", 962.77)
            append_throughput_history(path, "2026-09-20", 970.25)

            with path.open(newline="") as csvfile:
                rows = list(csv.DictReader(csvfile))

        self.assertEqual(rows, [
            {"date": "2026-09-19", "total_tokens_per_s": "962.77"},
            {"date": "2026-09-20", "total_tokens_per_s": "970.25"},
        ])


if __name__ == "__main__":
    unittest.main()

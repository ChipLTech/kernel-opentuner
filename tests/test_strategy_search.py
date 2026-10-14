import ast
import hashlib
import importlib.util
import json
import mmap
import os
import subprocess
import sys
import time
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

spec = importlib.util.find_spec("DLC.strategy_search")
if spec:
    from DLC import strategy_search as search
else:
    search = None


class ModuleTests(unittest.TestCase):
    def test_r4_is_an_importable_algorithm_not_an_arsenal_copy(self):
        self.assertIsNotNone(spec, "R4 search_policy has not been integrated")


@unittest.skipIf(search is None, "R4 module pending")
class SearchTests(unittest.TestCase):
    def test_whole_ab_group_restarts_when_epoch_changes_between_pairs(self):
        calls = []
        def run(data, label):
            calls.append(label)
            return {"total": 101 if data == b"candidate" else 100,
                    "recovery_epoch": 0 if len(calls) < 6 else 1}
        pairs = search.confirm_pairs(run, b"control", b"candidate")
        self.assertEqual(len(calls), 16)
        self.assertEqual({p["recovery_epoch"] for p in pairs}, {1})

    def test_runtime_evidence_checks_exact_r4_tokens_and_batching(self):
        text = ("Command: vllm bench throughput --model /model --dataset /dataset "
                "--num-prompts 32 --output-len 512 --seed 1024 --dtype bfloat16 "
                "--block-size 256 --max-num-batched-tokens 4096 --gpu-memory-utilization 0.95\n"
                "non-default args: {'seed': 1024, 'dtype': 'bfloat16', 'block_size': 256, "
                "'max_num_batched_tokens': 4096, 'gpu_memory_utilization': 0.95, 'enforce_eager': True}\n"
                "Total num prompt tokens: 6854\nTotal num output tokens: 16384\n")
        workload = search.observed_workload(text)
        self.assertEqual(workload["input_tokens"], 6854)
        self.assertEqual(workload["output_tokens"], 16384)
        with self.assertRaises(search.IntegrityFailure):
            search.observed_workload(text.replace("4096", "1024"))
        with self.assertRaises(RuntimeError):
            search.observed_workload(text.replace("16384", "16383"))

    def test_cli_rejects_multiple_cards_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "output"
            result = subprocess.run([sys.executable, search.__file__, "search",
                                     "--original-policy", "/fixture.csv", "--output", str(path),
                                     "--device", "0,1"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("one assigned device", result.stderr)
            self.assertFalse(path.exists())

    def test_search_core_matches_the_hardware_accepted_r4(self):
        # R4 archive SHA256 8735890e6b5fbd635add527d85d6348322b4da03c7e9f1ca8cba685c54aeab52.
        names = {"policy_rows", "policy_header", "policy_identity", "replace_rows",
                 "kernel_family", "summarize_profiles", "cycle_guidance",
                 "beam_search", "search_policy"}
        tree = ast.parse(Path(search.__file__).read_text())
        parts = [ast.dump(node, include_attributes=False) for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
        self.assertEqual(len(parts), 9)
        self.assertEqual(hashlib.sha256("\n".join(parts).encode()).hexdigest(),
                         "861c70d807445a2f099f5a9a92f394cac25f65b724fa13340333b8e74e2d7177")

    def test_beam_crosses_regressions_without_duplicate_measurements(self):
        calls = []
        scores = {(0,): 10, (1,): 9, (2,): 6, (3,): 14}
        beam, _ = search.beam_search(
            (0,), [], lambda s: [(s[0] + 1,)] if s[0] < 3 else [],
            lambda s: calls.append(s) or scores[s], lambda _: 0,
            width=2, rounds=9, budget=4)
        self.assertEqual(beam[0], (3,))
        self.assertEqual(len(calls), len(set(calls)))

    def test_whole_row_search_combines_donors_automatically(self):
        base = b"Test,X\nmm_32_bf16,0\nmm_256_bf16,0\n"
        donors = [base.replace(b",0", b",1"), base.replace(b",0", b",2")]
        winner = b"Test,X\nmm_32_bf16,1\nmm_256_bf16,2\n"
        result = search.search_policy(base, donors, {}, [{}, {}],
                                      lambda data: 14 if data == winner else 10,
                                      budget=50, rounds=9,
                                      confirm=lambda data: 14 if data == winner else 10)
        self.assertEqual(result["strategy"], winner)
        self.assertTrue(result["auto_selected"])
        self.assertTrue(all(len(f["totals"]) == 3 for f in result["finalists"]))

    def test_single_lucky_screen_peak_does_not_win_repeats(self):
        base = b"Test,X\na,0\n"
        donor = b"Test,X\na,1\n"
        result = search.search_policy(base, donor, {}, {},
                                      lambda data: 1000 if data == donor else 10,
                                      budget=50, rounds=9,
                                      confirm=lambda data: 9 if data == donor else 10)
        self.assertEqual(result["strategy"], base)

    def test_duplicate_rows_keep_cmake_last_occurrence(self):
        data = b"Test,X\na,shadow\na,active\n"
        self.assertEqual(search.policy_rows(data)["a"], ("a", "active"))
        self.assertEqual(search.replace_rows(data, {"a": ("a", "new")}),
                         b"Test,X\na,shadow\na,new\n")

    def test_five_independent_pairs_and_positive_interval_required(self):
        passed = search.confirmation_report(
            [{"reference": 100, "candidate": 101 + i * .05, "recovery_epoch": 0}
             for i in range(5)])
        self.assertTrue(passed["passed"])
        failed = search.confirmation_report(
            [{"reference": 100, "candidate": c, "recovery_epoch": 0}
             for c in [103, 99, 100.1, 99, 100.1]])
        self.assertFalse(failed["passed"])
        for candidates in ([101] * 4, [101, 101, float("nan"), 101, 101], [0] * 5):
            with self.subTest(candidates=candidates), self.assertRaises(ValueError):
                search.confirmation_report(
                    [{"reference": 100, "candidate": c} for c in candidates])

    def test_all_five_pairs_must_share_one_epoch(self):
        with self.assertRaises(ValueError):
            search.confirmation_report(
                [{"reference": 100, "candidate": 101, "recovery_epoch": i // 3}
                 for i in range(5)])

    def test_ab_ba_order_and_fixed_candidate(self):
        calls = []
        def run(data, label):
            calls.append(data)
            return {"total": 101 if data == b"candidate" else 100, "recovery_epoch": 0}
        pairs = search.confirm_pairs(run, b"control", b"candidate")
        self.assertEqual(calls, [b"control", b"candidate", b"candidate", b"control",
                                b"control", b"candidate", b"candidate", b"control",
                                b"control", b"candidate"])
        self.assertEqual(len(pairs), 5)

    def test_atomic_library_install_preserves_existing_mmap(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "library.so"
            path.write_bytes(b"old-library")
            with path.open("rb") as source, mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
                search.atomic_bytes(path, b"new-library")
                self.assertEqual(mapped[:], b"old-library")
                self.assertEqual(path.read_bytes(), b"new-library")


@unittest.skipIf(search is None, "R4 module pending")
class SeedTests(unittest.TestCase):
    def test_base_and_donors_are_measured_inputs_not_postprocess_snapshots(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = b"Test,X\na,0\n"
            for i, total in enumerate([100, 105, 104]):
                directory = root / "123" / ("profile_iter" + str(i))
                directory.mkdir(parents=True)
                (directory / "executed_strategy.csv").write_bytes(reference.replace(b",0", ("," + str(i)).encode()))
                (directory / "throughput_metrics.json").write_text(json.dumps({"total_tokens_per_s": total}))
            (root / "123/strategy_iter2.csv").write_bytes(reference.replace(b",0", b",999"))
            control, base, donors, manifest = search.select_seed_policies(root, reference)
            self.assertEqual(control, reference)
            self.assertEqual(base, b"Test,X\na,1\n")
            self.assertEqual(donors, [b"Test,X\na,2\n"])
            self.assertEqual(len(manifest), 2)
            self.assertNotIn(b"999", base + b"".join(donors))

    def test_no_distinct_donor_is_a_clear_failure_not_fake_success(self):
        with tempfile.TemporaryDirectory() as temp, self.assertRaises(ValueError):
            search.select_seed_policies(Path(temp), b"Test,X\na,0\n")

    def test_new_kernel_rows_align_using_existing_default_semantics(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            header = b"Test,A,B,C,D,E,F,G,H,I,J,K,L,M\n"
            defaults = b",,,,,,1.0,all,-1.0,-1.0,15,0,,0\n"
            reference = header + b"a" + defaults
            for i in (1, 2):
                directory = root / ("profile_iter" + str(i))
                directory.mkdir()
                (directory / "executed_strategy.csv").write_bytes(
                    reference + b"new," + str(i).encode() + defaults[1:])
                (directory / "throughput_metrics.json").write_text(json.dumps({"total_tokens_per_s": 100 + i}))
            control, base, donors, _ = search.select_seed_policies(root, reference)
            self.assertEqual(search.policy_rows(control).keys(), search.policy_rows(base).keys())
            self.assertEqual(search.policy_rows(control)["new"], ("new", *search.DEFAULT_POLICY))
            self.assertEqual(len(donors), 1)


@unittest.skipIf(search is None, "R4 module pending")
class IntegrityTests(unittest.TestCase):
    def test_timed_out_owned_process_is_reaped_before_next_candidate(self):
        with tempfile.TemporaryDirectory() as temp:
            evaluator = search.HardwareEvaluator.__new__(search.HardwareEvaluator)
            evaluator.environment = dict(os.environ)
            started = time.monotonic()
            code = evaluator.command([sys.executable, "-c", "import time; time.sleep(10)"],
                                     Path(temp) / "timeout.log", .1)
            self.assertEqual(code, 124)
            self.assertLess(time.monotonic() - started, 3)
            self.assertFalse(getattr(evaluator, "cleanup_failed", False))

    def test_initial_profiling_environment_does_not_depend_on_inherited_flags(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            evaluator = self.evaluator(root)
            args = SimpleNamespace(output=root / "output", kernel_root=root / "kernel",
                                   library=evaluator.library, device="5", profiling="off",
                                   original_policy=evaluator.policy, original_library=evaluator.library)
            path = args.kernel_root / "dlc_src/opt_flag_data/autotune_strategies.csv"
            path.parent.mkdir(parents=True)
            path.write_bytes(evaluator.original)
            with patch.object(search.HardwareEvaluator, "source_identity", return_value={}), patch.dict(
                "os.environ", {}, clear=True
            ):
                actual = search.HardwareEvaluator(args)
            for key in ("DLC_SYN_DEBUG", "DLC_SYN_VERBOSE", "DLC_SYN_PROF_CYCLE"):
                self.assertEqual(actual.environment.get(key), "0")

    def evaluator(self, root):
        evaluator = search.HardwareEvaluator.__new__(search.HardwareEvaluator)
        evaluator.out = root
        evaluator.policy = root / "policy.csv"
        evaluator.library = root / "library.so"
        evaluator.original = b"Test,X\na,original\n"
        evaluator.original_library = b"original-library"
        evaluator.policy.write_bytes(b"Test,X\na,candidate\n")
        evaluator.library.write_bytes(b"candidate-library")
        evaluator.expected = search.digest(evaluator.policy.read_bytes())
        evaluator.current_library = search.digest(evaluator.library.read_bytes())
        evaluator.sources = {}
        evaluator.source_identity = lambda: {}
        evaluator.status = lambda **fields: None
        return evaluator

    def test_integrity_error_is_fatal_and_never_retried(self):
        evaluator = search.HardwareEvaluator.__new__(search.HardwareEvaluator)
        calls = []
        def run(*args):
            calls.append(args)
            raise search.IntegrityFailure("external modification")
        evaluator.run = run
        evaluator.status = lambda **fields: None
        with self.assertRaises(search.IntegrityFailure):
            evaluator.retry_run(b"policy", "label")
        self.assertEqual(len(calls), 1)

    def test_failed_qualification_restores_control_not_primary_winner(self):
        with tempfile.TemporaryDirectory() as temp:
            evaluator = self.evaluator(Path(temp))
            evaluator.restore()
            self.assertEqual(evaluator.policy.read_bytes(), evaluator.original)
            self.assertEqual(evaluator.library.read_bytes(), evaluator.original_library)

    def test_external_policy_or_library_is_preserved_during_cleanup(self):
        for target in ("policy", "library"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temp:
                evaluator = self.evaluator(Path(temp))
                getattr(evaluator, target).write_bytes(b"external-content")
                with self.assertRaises(search.IntegrityFailure):
                    evaluator.restore()
                self.assertEqual(getattr(evaluator, target).read_bytes(), b"external-content")


@unittest.skipIf(search is None, "R4 module pending")
class StageTests(unittest.TestCase):
    def test_qualification_rejects_changed_rollback_snapshots_or_assigned_card(self):
        for target in ("original_policy", "original_library", "device"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                self.execute_plan(root, True)
                policy = root / "kernel/dlc_src/opt_flag_data/autotune_strategies.csv"
                previous = policy.read_bytes()
                args = SimpleNamespace(phase="qualify", profiling="on", output=root / "r4/on-new",
                                       source=root / "r4/off", device="5", kernel_root=root / "kernel",
                                       library=root / "library.so", original_policy=root / "before.csv",
                                       original_library=root / "before.so", budget=50, rounds=9)
                if target == "device":
                    args.device = "6"
                else:
                    getattr(args, target).write_bytes(b"changed-rollback")
                with self.assertRaises(search.IntegrityFailure):
                    search.run_stage(args)
                self.assertEqual(policy.read_bytes(), previous)
                self.assertFalse(args.output.exists())

    def test_source_identity_includes_untracked_build_and_python_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.test",
                            "commit", "--allow-empty", "-qm", "fixture"], check=True)
            (root / "DLC").mkdir()
            (root / "DLC/vllm_profile_runner.py").write_text("# fixture\n")
            config = root / "model-config.json"
            config.write_text("{}")
            evaluator = search.HardwareEvaluator.__new__(search.HardwareEvaluator)
            evaluator.args = SimpleNamespace(kernel_root=root, tuner_root=root, model_config=config)
            extra = root / "kernel.cpp"
            extra.write_text("int policy = 1;\n")
            first = evaluator.source_identity()
            extra.write_text("int policy = 2;\n")
            self.assertNotEqual(first, evaluator.source_identity())

    def execute_plan(self, root, on_pass):
        before = b"Test,X\na,0\n"
        base, donor = b"Test,X\na,1\n", b"Test,X\na,2\n"
        policy = root / "kernel/dlc_src/opt_flag_data/autotune_strategies.csv"
        policy.parent.mkdir(parents=True)
        policy.write_bytes(donor)
        library = root / "library.so"
        library.write_bytes(b"seed-library")
        original, original_library = root / "before.csv", root / "before.so"
        original.write_bytes(before)
        original_library.write_bytes(b"control-library")
        inputs = root / "inputs"
        inputs.mkdir()
        for name, data in (("reference.csv", before), ("base.csv", base), ("donor-1.csv", donor)):
            (inputs / name).write_bytes(data)
        search.atomic_json(inputs / "manifest.json", {
            "reference_sha": sha(before), "base_sha": sha(base), "donor_shas": [sha(donor)],
            "original_strategy_sha": sha(before), "original_library_sha": sha(b"control-library")})
        args = SimpleNamespace(phase="search", profiling="off", output=root / "r4/off", source=None,
                               device="5", kernel_root=root / "kernel", library=library,
                               original_policy=original, original_library=original_library,
                               inputs=inputs, budget=50, rounds=9)
        calls = []

        class FixtureEvaluator(search.HardwareEvaluator):
            def __init__(self, args):
                self.args, self.out = args, args.output
                self.out.mkdir(parents=True)
                (self.out / "original.csv").write_bytes(before)
                (self.out / "original.so").write_bytes(b"control-library")
                self.policy, self.library = policy, library
                self.original, self.original_library = before, b"control-library"
                self.expected, self.current_library = sha(policy.read_bytes()), sha(library.read_bytes())
                self.sources, self.counter, self.workload = {"fixture": 1}, 0, None
                self.environment = {key: "0" for key in ("DLC_SYN_DEBUG", "DLC_SYN_VERBOSE", "DLC_SYN_PROF_CYCLE")}
            def source_identity(self):
                return {"fixture": 1}
            def status(self, **kwargs):
                pass
            def install(self, data, directory):
                self.assert_integrity()
                policy.write_bytes(data)
                library.write_bytes(b"compiled-" + data)
                self.expected, self.current_library = sha(data), sha(library.read_bytes())
            def run(self, data, label):
                self.counter += 1
                calls.append((self.args.phase, data, label))
                self.install(data, self.out)
                directory = self.out / str(self.counter)
                (directory / "profile").mkdir(parents=True)
                (directory / "profile/syn_fixture.ansi").write_text("kernel_name: custom_a cycles: 100\n" * 32)
                total = 100 if data == before else (101 if data == base else 100.5)
                if self.args.phase == "qualify" and not on_pass and data == base:
                    total = 99.5
                self.workload = {"fixture_workload": 1}
                return {"total": total, "path": str(directory), "workload": self.workload, "recovery_epoch": 0}

        with patch.object(search, "HardwareEvaluator", FixtureEvaluator), patch.object(
            search, "search_policy", wraps=search.search_policy
        ) as selection:
            search.run_stage(args)
            args.phase, args.profiling, args.source, args.output = "qualify", "on", args.output, root / "r4/on"
            search.run_stage(args)
            self.assertEqual(selection.call_count, 1, "on confirmation must not choose another winner")
        return policy, library, calls

    def test_actual_stage_orchestration_uses_one_search_and_two_independent_groups(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            policy, _, calls = self.execute_plan(root, True)
            self.assertEqual(policy.read_bytes(), b"Test,X\na,1\n")
            self.assertTrue(json.loads((root / "r4/acceptance.json").read_text())["passed"])
            for phase in ("search", "qualify"):
                self.assertEqual(sum(p == phase and label.startswith("pair-") for p, _, label in calls), 10)

    def test_secondary_failure_restores_the_pre_seed_control_and_library(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            policy, library, _ = self.execute_plan(root, False)
            self.assertEqual(policy.read_bytes(), b"Test,X\na,0\n")
            self.assertEqual(library.read_bytes(), b"control-library")
            self.assertFalse(json.loads((root / "r4/acceptance.json").read_text())["passed"])


def sha(data):
    return search.digest(data)


if __name__ == "__main__":
    unittest.main()

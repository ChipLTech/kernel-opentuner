import os
import sys
import tempfile
import unittest
from pathlib import Path


DLC_ROOT = Path(__file__).resolve().parents[1] / "DLC"
sys.path.insert(0, str(DLC_ROOT))

from llamatool import diagnose_llama_result, get_kernel_launches  # noqa: E402
from save_kernels import get_latest_ansi, parse_profile, update_dlcutils  # noqa: E402


SEPARATOR = "─" * 83


def launch(name, body="tensor body\n"):
    return f"{SEPARATOR} launch  {name}  on dlc:0\n{body}"


def named_cycles(name, cycles, batch_id=2, kernel_index=0):
    return (
        f"{SEPARATOR} batch_id: {batch_id}  kernel_index: {kernel_index}  "
        f"kernel_name: {name}  cycles: {cycles}  ops: 4096  bytes: 32768  "
        "crt: 0  gflops: 3.5453  gb/s: 28.3624\n"
    )


class SaveKernelsTests(unittest.TestCase):
    def test_legacy_cycles_keep_launch_body_and_aggregate(self):
        profile = (
            launch("custom_matmul", "matmul body\n")
            + f"{SEPARATOR} cycles: 1000  ops: 8  bytes: 16  crt: 1\n"
            + launch("custom_matmul")
            + f"{SEPARATOR} cycles: 4000  ops: 8  bytes: 16  crt: 1\n"
            + launch("custom_sum")
            + f"{SEPARATOR} cycles: 1500  ops: 8  bytes: 16  crt: 1\n"
        )

        self.assertEqual(
            diagnose_llama_result(profile),
            ({"matmul": 5000, "sum": 1500}, 6500),
        )
        metrics = get_kernel_launches(profile)
        self.assertEqual(len(metrics), 3)
        self.assertEqual(metrics[0].body, "matmul body\n")

    def test_named_cycles_ignore_xys_diagnostics(self):
        profile = (
            launch("custom_full")
            + named_cycles("custom_full", 1733)
            + "\nbatch_id: 2  kernel_index: 0  kernel_name: custom_full  "
            "xys0_cycle: 1733  xys1_cycle: 1837\n"
        )

        self.assertEqual(diagnose_llama_result(profile), ({"full": 1733}, 1733))
        metrics = get_kernel_launches(profile)
        self.assertEqual(len(metrics), 1)
        self.assertEqual(metrics[0].body, "tensor body\n")

    def test_named_cycles_use_footer_name_for_interleaved_launches(self):
        profile = (
            launch("custom_full", "full body\n")
            + launch("custom_sum", "sum body\n")
            + named_cycles("custom_full", 1733)
            + named_cycles("custom_sum", 300)
        )

        self.assertEqual(
            diagnose_llama_result(profile), ({"full": 1733, "sum": 300}, 2033)
        )
        metrics = get_kernel_launches(profile)
        self.assertEqual(metrics[0].body, "")
        self.assertEqual(metrics[1].body, "sum body\n")

    def test_named_cycles_accept_out_of_order_batches_and_multiple_kernels(self):
        profile = (
            named_cycles("custom_full", 1733, batch_id=2)
            + named_cycles("custom_full", 2764, batch_id=1)
            + named_cycles("custom_sum", 300, batch_id=1, kernel_index=1)
        )

        self.assertEqual(
            diagnose_llama_result(profile), ({"full": 4497, "sum": 300}, 4797)
        )
        self.assertEqual(len(get_kernel_launches(profile)), 3)

    def test_mixed_legacy_and_named_cycles_are_both_counted(self):
        profile = (
            launch("custom_matmul")
            + f"{SEPARATOR} cycles: 1000  ops: 8  bytes: 16  crt: 1\n"
            + launch("custom_full")
            + named_cycles("custom_full", 1733)
        )

        self.assertEqual(
            diagnose_llama_result(profile), ({"full": 1733, "matmul": 1000}, 2733)
        )

    def test_ansi_control_lines_accept_a_footer_without_final_newline(self):
        for footer in (
            f"{SEPARATOR} cycles: 1733  ops: 4096  bytes: 32768  crt: 0",
            named_cycles("custom_full", 1733).rstrip("\n"),
        ):
            with self.subTest(footer=footer):
                profile = (
                    f"\x1b[32m{SEPARATOR} launch  custom_full  on dlc:0\x1b[0m\n"
                    "\x1b[31mtensor body\x1b[0m\n"
                    f"\x1b[32m{footer}\x1b[0m"
                )
                self.assertEqual(
                    diagnose_llama_result(profile), ({"full": 1733}, 1733)
                )
                self.assertEqual(get_kernel_launches(profile)[0].body, "tensor body\n")

    def test_a_new_launch_discards_an_incomplete_legacy_launch(self):
        profile = (
            launch("custom_incomplete", "incomplete body\n")
            + launch("custom_sum", "sum body\n")
            + f"{SEPARATOR} cycles: 1500  ops: 8  bytes: 16  crt: 1\n"
        )

        self.assertEqual(diagnose_llama_result(profile), ({"sum": 1500}, 1500))
        self.assertEqual(get_kernel_launches(profile)[0].body, "sum body\n")

    def test_empty_incomplete_and_malformed_profiles_fail_closed(self):
        invalid_profiles = (
            "",
            launch("custom_full"),
            launch("custom_full")
            + "batch_id: 2  kernel_index: 0  kernel_name: custom_full  "
            "xys0_cycle: 1733  xys1_cycle: 1837\n",
            launch("custom_full") + f"{SEPARATOR} cycles: 1733\n",
            launch("custom_full")
            + named_cycles("custom_full", 1733).replace("ops: 4096", "ops: invalid"),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "profile.ansi"
            for profile in invalid_profiles:
                with self.subTest(profile=profile):
                    path.write_text(profile)
                    self.assertEqual(get_kernel_launches(profile), [])
                    with self.assertRaisesRegex(RuntimeError, "No kernel cycles parsed"):
                        parse_profile(path)

    def test_parse_profile_consumes_named_cycles(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "profile.ansi"
            path.write_text(launch("custom_full") + named_cycles("custom_full", 1733))

            self.assertEqual(parse_profile(path), (["full"], 1733))

    def test_concatenated_summaries_do_not_mix_names_and_counters(self):
        first = named_cycles("custom_full", 1733).rstrip("\n")
        invalid = first.replace("cycles: 1733", "cycles: invalid")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "profile.ansi"
            for summary in (
                first, invalid,
                first.replace("  cycles:", "\tcycles:"),
                invalid.replace("  cycles:", "\tcycles:"),
            ):
                with self.subTest(summary=summary):
                    profile = (
                        launch("custom_full") + summary + named_cycles("custom_sum", 300)
                    )
                    path.write_text(profile)

                    self.assertEqual(get_kernel_launches(profile), [])
                    with self.assertRaisesRegex(RuntimeError, "No kernel cycles parsed"):
                        parse_profile(path)

    def test_get_latest_ansi_selects_the_newest_parseable_named_profile(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            old = root / "old.ansi"
            old.write_text(
                launch("custom_sum")
                + f"{SEPARATOR} cycles: 1500  ops: 8  bytes: 16  crt: 1\n"
            )
            current = root / "current.ansi"
            current.write_text(launch("custom_full") + named_cycles("custom_full", 1733))
            incomplete = root / "incomplete.ansi"
            incomplete.write_text(launch("custom_full"))
            for timestamp, path in enumerate((old, current, incomplete), start=1):
                os.utime(path, (timestamp, timestamp))

            self.assertEqual(get_latest_ansi(root), current)

    def test_updates_delegating_qwen_function_from_profile_kernels(self):
        source = """\
def get_qwen3_8b_kernels():
  # Seed before the first profile is available.
  return get_deepseek_qwen_7b_kernels()


def get_deepseek_qwen_7b_kernels():
    return [\"seed\"]
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "dlcutils.py"
            path.write_text(source)

            update_dlcutils(path, "get_qwen3_8b_kernels", [
                "matmul_t_bf16_pingpong",
                "fused_qkv_and_rotary_embedding_bf16",
            ])

            updated = path.read_text()
            self.assertIn(
                'return [\n'
                '        "matmul_t_bf16_pingpong",\n'
                '        "fused_qkv_and_rotary_embedding_bf16",\n'
                '    ]',
                updated,
            )
            self.assertIn('return ["seed"]', updated)


if __name__ == "__main__":
    unittest.main()

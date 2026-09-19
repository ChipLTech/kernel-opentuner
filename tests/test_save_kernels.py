import sys
import tempfile
import unittest
from pathlib import Path


DLC_ROOT = Path(__file__).resolve().parents[1] / "DLC"
sys.path.insert(0, str(DLC_ROOT))

from save_kernels import update_dlcutils  # noqa: E402


class SaveKernelsTests(unittest.TestCase):
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

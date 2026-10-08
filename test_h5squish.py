# AI-generated h5squish tests begin.
"""Small, readable checks for the public commands and decoding."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

import h5squish


SCRIPT = Path(__file__).with_name("h5squish.py")


class H5SquishTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.src = self.folder / "source.h5"
        self.dst = self.folder / "copy.h5"
        self.cfg = self.folder / "input.cfg"

        with h5py.File(self.src, "w") as file:
            file.attrs["survey"] = "test"
            group = file.create_group("TRUE")
            group.attrs["frame"] = "ICRS"
            flux = group.create_dataset(
                "flux", data=[0.0, 1.0, 2.0, 3.0,
                              -np.inf, np.inf, np.nan])
            flux.attrs["units"] = "Jy"
            group.create_dataset("special", data=[-np.inf, np.inf, np.nan])
            group.create_dataset("constant", data=[7.0, 7.0])
            group.create_dataset("scalar", data=np.float64(2.5))
            group.create_dataset("id", data=np.array([0, 100, 200],
                                                      dtype="int16"))
            group.create_dataset("type", data=np.array(
                [b"dwarf", b"dwarf", b"giant", b"dwarf"], dtype="S8"))

    def config(self, *rows):
        lines = ["# name type offset span fraction new_type accuracy"]
        for path, old_type, new_type, accuracy in rows:
            lines.append(f"{path} {old_type} ------ ------ 0.0 "
                         f"{new_type} {accuracy}")
        self.cfg.write_text("\n".join(lines) + "\n", encoding="ascii")

    def copy(self):
        subprocess.run([sys.executable, str(SCRIPT), "copy",
                        f"--src={self.src}", f"--dst={self.dst}",
                        f"--cfg={self.cfg}"], check=True)

    def test_generated_config_has_absolute_paths_and_fractions(self):
        output = subprocess.check_output(
            [sys.executable, str(SCRIPT), "config", f"--src={self.src}"],
            text=True)
        rows = {line.split()[0]: line.split() for line in output.splitlines()
                if line and not line.startswith("#")}
        self.assertIn("/TRUE/flux", rows)
        self.assertEqual(rows["/TRUE/type"][1:4],
                         ["string", "------", "2"])
        with h5py.File(self.src) as file:
            expected = file["/TRUE/flux"].id.get_storage_size()
        fraction = expected / self.src.stat().st_size
        self.assertAlmostEqual(float(rows["/TRUE/flux"][4]), fraction,
                               delta=5e-7)

    def test_unconfigured_copy_and_replace_preserve_data(self):
        self.config()
        self.dst.write_bytes(b"old contents")
        self.copy()
        subprocess.run([sys.executable, str(SCRIPT), "replace",
                        f"--src={self.dst}", f"--cfg={self.cfg}"],
                       check=True)
        with h5py.File(self.src) as original, h5py.File(self.dst) as copied:
            self.assertEqual(copied.attrs["survey"], "test")
            self.assertEqual(copied["TRUE"].attrs["frame"], "ICRS")
            self.assertEqual(copied["TRUE/flux"].attrs["units"], "Jy")
            np.testing.assert_array_equal(original["TRUE/type"][:],
                                          copied["TRUE/type"][:])
            np.testing.assert_array_equal(original["TRUE/id"][:],
                                          copied["TRUE/id"][:])
            self.assertEqual(copied["TRUE/flux"].dtype, np.dtype("float64"))

    def test_quantization_reserves_special_codes_and_decodes(self):
        self.config(("/TRUE/flux", "float64", "uint8", "1"))
        self.copy()
        with h5py.File(self.dst) as file:
            data = file["TRUE/flux"]
            self.assertEqual(data.dtype, np.dtype("uint8"))
            np.testing.assert_array_equal(data[:],
                                          [0, 1, 2, 2, 253, 254, 255])
            self.assertEqual(data.attrs["offset"], 0)
            self.assertEqual(data.attrs["dx"], 1)
            self.assertEqual(data.attrs["units"], "Jy")

            first = h5squish.decode(file, "/TRUE/flux")
            second = h5squish.decode(file, "/TRUE/flux")
            np.testing.assert_array_equal(first, second)
            self.assertTrue(np.all(np.abs(first[:4] - [0, 1, 2, 3]) < 1))
            self.assertTrue(np.isneginf(first[4]))
            self.assertTrue(np.isposinf(first[5]))
            self.assertTrue(np.isnan(first[6]))

    def test_inferred_type_and_no_finite_values(self):
        self.config(("/TRUE/flux", "float64", "------", "0.1"),
                    ("/TRUE/special", "float64", "uint8", "------"),
                    ("/TRUE/constant", "float64", "uint8", "------"))
        self.copy()
        with h5py.File(self.dst) as file:
            self.assertEqual(file["TRUE/flux"].dtype, np.dtype("uint8"))
            special = file["TRUE/special"]
            np.testing.assert_array_equal(special[:], [253, 254, 255])
            self.assertEqual(special.attrs["offset"], 0)
            self.assertEqual(special.attrs["dx"], 0)
            self.assertEqual(file["TRUE/constant"].attrs["dx"], 0)
            np.testing.assert_array_equal(
                h5squish.decode(file, "/TRUE/constant"), [7, 7])

    def test_signed_codes_and_scalar(self):
        self.config(("/TRUE/flux", "float64", "int8", "------"),
                    ("/TRUE/scalar", "float64", "uint8", "1"))
        self.copy()
        with h5py.File(self.dst) as file:
            codes = file["TRUE/flux"][:]
            self.assertEqual(codes[0], -128)
            self.assertEqual(codes[3], 124)
            np.testing.assert_array_equal(codes[-3:], [125, 126, 127])
            self.assertEqual(h5squish.decode(file, "/TRUE/scalar"), 2.5)

    def test_replace_uses_the_conversion_path(self):
        self.config(("/TRUE/id", "int16", "uint8", "------"))
        subprocess.run([sys.executable, str(SCRIPT), "replace",
                        f"--src={self.src}", f"--cfg={self.cfg}"],
                       check=True)
        with h5py.File(self.src) as file:
            self.assertEqual(file["TRUE/id"].dtype, np.dtype("uint8"))
            np.testing.assert_array_equal(file["TRUE/id"][:], [0, 100, 200])
            self.assertEqual(file.attrs["survey"], "test")

    def test_string_lookup_and_checked_integer_cast(self):
        self.config(("/TRUE/type", "string", "uint8", "------"),
                    ("/TRUE/id", "int16", "uint8", "------"))
        self.copy()
        with h5py.File(self.dst) as file:
            types = file["TRUE/type"]
            np.testing.assert_array_equal(types[:], [0, 0, 1, 0])
            np.testing.assert_array_equal(types.attrs["lookup"],
                                          [b"dwarf", b"giant"])
            np.testing.assert_array_equal(
                h5squish.decode(file, "/TRUE/type"),
                [b"dwarf", b"dwarf", b"giant", b"dwarf"])
            np.testing.assert_array_equal(file["TRUE/id"][:], [0, 100, 200])

        self.config(("/TRUE/id", "int16", "int8", "------"))
        rules = h5squish.read_config(self.cfg, self.src)
        with self.assertRaisesRegex(ValueError, "do not fit"):
            h5squish.copy_file(self.src, self.dst, rules)
        with h5py.File(self.dst) as file:
            np.testing.assert_array_equal(file["TRUE/id"][:], [0, 100, 200])

    def test_float_cast_and_bad_config(self):
        self.config(("/TRUE/flux", "float64", "float32", "------"))
        self.copy()
        with h5py.File(self.dst) as file:
            self.assertEqual(file["TRUE/flux"].dtype, np.dtype("float32"))
            self.assertTrue(np.isnan(file["TRUE/flux"][-1]))

        self.config(("/TRUE/flux", "float32", "uint8", "1"))
        with self.assertRaisesRegex(ValueError, "has type float64"):
            h5squish.read_config(self.cfg, self.src)

        self.config(("/TRUE/id", "int16", "uint8", "------"),
                    ("/TRUE/id", "int16", "uint8", "------"))
        with self.assertRaisesRegex(ValueError, "duplicate path"):
            h5squish.read_config(self.cfg, self.src)


if __name__ == "__main__":
    unittest.main()
# AI-generated h5squish tests end.

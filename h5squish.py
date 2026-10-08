# AI-generated h5squish implementation begins.
"""Convert HDF5 datasets using a text config and decode converted data."""

import argparse
import math
import os
import secrets
import stat
import tempfile
from collections import Counter
from fractions import Fraction
from pathlib import Path

import h5py
import numpy as np


NUMERIC_TYPES = (
    "float16", "float32", "float64",
    "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
)


def blocks(dataset):
    if dataset.ndim == 0:
        yield ()
        return

    row_bytes = dataset.dtype.itemsize * math.prod(dataset.shape[1:])
    rows = max(1, 8 * 1024 * 1024 // max(1, row_bytes))
    for start in range(0, dataset.shape[0], rows):
        yield slice(start, min(start + rows, dataset.shape[0]))


def finite_bounds(dataset):
    low = high = None
    for block in blocks(dataset):
        data = np.asarray(dataset[block])
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            continue
        block_low = finite.min().item()
        block_high = finite.max().item()
        low = block_low if low is None else min(low, block_low)
        high = block_high if high is None else max(high, block_high)
    return low, high


def limits(dataset):
    if h5py.check_string_dtype(dataset.dtype) is not None:
        unique = set()
        for block in blocks(dataset):
            unique.update(np.asarray(dataset[block]).flat)
        return "------", str(len(unique))

    if dataset.dtype.kind not in "biuf":
        return "------", "------"

    low, high = finite_bounds(dataset)
    if low is None:
        return "------", "------"
    if dataset.dtype.kind in "biu":
        return str(int(low)), str(int(high) - int(low))

    digits = 5 if dataset.dtype.itemsize <= 4 else 12
    return format(low, f".{digits}g"), format(high - low, f".{digits}g")


def write_config(src):
    rows = []
    file_bytes = src.stat().st_size
    with h5py.File(src, "r") as source:
        def add_dataset(name, obj):
            if not isinstance(obj, h5py.Dataset):
                return
            is_string = h5py.check_string_dtype(obj.dtype) is not None
            dtype = "string" if is_string else obj.dtype.name
            offset, span = limits(obj)
            fraction = obj.id.get_storage_size() / file_bytes
            rows.append(("/" + name, dtype, offset, span,
                         f"{fraction:.3e}", "------", "------"))

        source.visititems(add_dataset)

    print("# 0 - name")
    print("# 1 - type (NumPy dtype or string)")
    print("# 2 - offset (minimum finite value)")
    print("# 3 - span (maximum minus minimum; unique count for strings)")
    print("# 4 - fraction of file size (dataset storage / file size)")
    print("# 5 - new type")
    print("# 6 - target accuracy")
    if rows:
        widths = [max(len(row[i]) for row in rows) for i in range(6)]
        for row in rows:
            print("  ".join(value.ljust(widths[i])
                            for i, value in enumerate(row[:6])), row[6])


def read_config(cfg, src):
    rules = {}
    with h5py.File(src, "r") as source, cfg.open(encoding="ascii") as file:
        for number, line in enumerate(file, 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split()
            if len(fields) != 7:
                raise ValueError(f"config line {number}: expected 7 columns")

            path, old_type, _, _, _, new_type, accuracy = fields
            if not path.startswith("/") or path == "/":
                raise ValueError(f"config line {number}: invalid path {path}")
            if path in rules:
                raise ValueError(f"config line {number}: duplicate path {path}")
            if path not in source or not isinstance(source[path], h5py.Dataset):
                raise ValueError(f"config line {number}: "
                                 f"unknown dataset {path}")

            dataset = source[path]
            is_string = h5py.check_string_dtype(dataset.dtype) is not None
            actual_type = "string" if is_string else dataset.dtype.name
            if old_type != actual_type:
                raise ValueError(f"config line {number}: {path} has type "
                                 f"{actual_type}, not {old_type}")

            if accuracy == "------":
                accuracy = None
            else:
                try:
                    accuracy = float(accuracy)
                except ValueError as error:
                    raise ValueError(
                        f"config line {number}: bad accuracy"
                    ) from error
                if not math.isfinite(accuracy) or accuracy <= 0:
                    raise ValueError(f"config line {number}: accuracy must "
                                     "be positive and finite")

            if new_type == "------":
                target = None
            elif new_type == "string":
                target = "string"
            else:
                try:
                    target = np.dtype(new_type)
                except (TypeError, ValueError) as error:
                    raise ValueError(f"config line {number}: bad type "
                                     f"{new_type}") from error
                if new_type not in NUMERIC_TYPES:
                    raise ValueError(f"config line {number}: unsupported "
                                     f"type {new_type}")

            kind = dataset.dtype.kind
            int_target = (isinstance(target, np.dtype) and
                          target.kind in "iu")
            float_target = (isinstance(target, np.dtype) and
                            target.kind == "f")
            if target is None and accuracy is None:
                mode = "copy"
            elif is_string and target == "string" and accuracy is None:
                mode = "copy"
            elif is_string and int_target and accuracy is None:
                mode = "lookup"
            elif kind == "f" and float_target and accuracy is None:
                mode = "float"
            elif kind == "f" and (target is None or int_target):
                mode = "quantize"
            elif kind in "iu" and int_target and accuracy is None:
                mode = "integer"
            else:
                raise ValueError(f"config line {number}: unsupported "
                                 f"conversion for {path}")

            rules[path] = (mode, target, accuracy)
    return rules


def copy_attrs(source, target):
    for name, value in source.attrs.items():
        dtype = source.attrs.get_id(name).dtype
        target.attrs.create(name, value, dtype=dtype)


def quantization(dataset, target, accuracy):
    for name in ("offset", "dx", "seed"):
        if name in dataset.attrs:
            raise ValueError(f"{dataset.name}: attribute {name} already exists")

    low, high = finite_bounds(dataset)
    if low is None:
        low = high = 0.0
    span = high - low
    if not math.isfinite(span):
        raise ValueError(f"{dataset.name}: finite range is too large")

    if span == 0:
        bins = 1
    elif accuracy is None:
        bins = None
    else:
        needed = span / accuracy
        if not math.isfinite(needed):
            raise ValueError(f"{dataset.name}: accuracy needs too many bins")
        bins = max(1, math.ceil(needed))

    if target is None:
        for name in ("uint8", "uint16", "uint32", "uint64"):
            candidate = np.dtype(name)
            info = np.iinfo(candidate)
            if bins <= int(info.max) - int(info.min) - 2:
                target = candidate
                break
        if target is None:
            raise ValueError(f"{dataset.name}: accuracy needs more than "
                             "64 integer bits")

    info = np.iinfo(target)
    capacity = int(info.max) - int(info.min) - 2
    if bins is None:
        bins = capacity
    if bins > capacity:
        raise ValueError(f"{dataset.name}: {target.name} cannot meet "
                         "the requested accuracy")

    dx = span / bins
    if span > 0 and dx == 0:
        raise ValueError(f"{dataset.name}: bin width underflows float64")
    return target, low, high, dx, bins


def quantized_values(data, target, low, high, dx, bins):
    info = np.iinfo(target)
    imin, imax = int(info.min), int(info.max)
    codes = np.empty(data.shape, dtype=target)
    codes[np.isneginf(data)] = imax - 2
    codes[np.isposinf(data)] = imax - 1
    codes[np.isnan(data)] = imax

    finite = np.isfinite(data)
    values = data[finite]
    if values.size == 0:
        return codes
    if dx == 0:
        codes[finite] = imin
    elif bins <= 2**53:
        indices = np.floor((values.astype("float64") - low) / dx)
        indices = np.clip(indices, 0, bins - 1).astype("int64")
        codes[finite] = (indices + imin).astype(target)
    else:
        lo_frac = Fraction.from_float(float(low))
        range_frac = Fraction.from_float(float(high)) - lo_frac
        indices = []
        for value in values:
            position = (Fraction.from_float(float(value)) - lo_frac)
            index = position * bins // range_frac
            indices.append(imin + min(bins - 1, max(0, index)))
        codes[finite] = np.asarray(indices, dtype=target)
    return codes


def string_lookup(dataset, target):
    if "lookup" in dataset.attrs:
        raise ValueError(f"{dataset.name}: attribute lookup already exists")
    counts = Counter()
    for block in blocks(dataset):
        counts.update(np.asarray(dataset[block]).flat)
    values = sorted(counts, key=lambda value: (-counts[value], value))
    if len(values) - 1 > int(np.iinfo(target).max):
        raise ValueError(f"{dataset.name}: {target.name} cannot hold "
                         "all string values")
    return np.asarray(values, dtype=dataset.dtype)


def copy_dataset(name, source, target_group, rule):
    mode, target, accuracy = rule
    if mode == "quantize":
        target, low, high, dx, bins = quantization(
            source, target, accuracy)
    elif mode == "lookup":
        lookup = string_lookup(source, target)
        codes = {value: i for i, value in enumerate(lookup)}

    dtype = source.dtype if mode == "copy" else target
    target_dataset = target_group.create_dataset(
        name, shape=source.shape, dtype=dtype)
    copy_attrs(source, target_dataset)

    for block in blocks(source):
        data = np.asarray(source[block])
        shape = data.shape
        if mode == "integer":
            info = np.iinfo(target)
            if data.size and (int(data.min()) < int(info.min) or
                              int(data.max()) > int(info.max)):
                raise ValueError(f"{source.name}: values do not fit "
                                 f"{target.name}")
            data = data.astype(target)
        elif mode == "float":
            with np.errstate(over="ignore", invalid="ignore"):
                converted = data.astype(target)
            if np.any(np.isfinite(data) & ~np.isfinite(converted)):
                raise ValueError(f"{source.name}: values overflow "
                                 f"{target.name}")
            data = converted
        elif mode == "quantize":
            data = quantized_values(data, target, low, high, dx, bins)
        elif mode == "lookup":
            data = np.fromiter((codes[value] for value in data.flat),
                               dtype=target, count=data.size)
            data = data.reshape(shape)
        target_dataset[block] = data

    if mode == "quantize":
        target_dataset.attrs["offset"] = np.float64(low)
        target_dataset.attrs["dx"] = np.float64(dx)
        target_dataset.attrs["seed"] = np.uint64(secrets.randbits(64))
    elif mode == "lookup":
        target_dataset.attrs["lookup"] = lookup


def copy_group(source, target, rules):
    copy_attrs(source, target)
    for name, obj in source.items():
        if isinstance(obj, h5py.Group):
            copy_group(obj, target.create_group(name), rules)
            continue

        copy_dataset(name, obj, target,
                     rules.get(obj.name, ("copy", None, None)))


def copy_file(src, dst, rules):
    fd, name = tempfile.mkstemp(prefix=f".{dst.name}.",
                                suffix=".h5", dir=dst.parent)
    os.close(fd)
    temp = Path(name)
    try:
        with h5py.File(src, "r") as source, h5py.File(temp, "w") as target:
            copy_group(source, target, rules)
        previous = dst if dst.exists() else src
        temp.chmod(stat.S_IMODE(previous.stat().st_mode))
        os.replace(temp, dst)
    finally:
        temp.unlink(missing_ok=True)


def replace_file(src, rules):
    copy_file(src, src, rules)


def decode(h5_file, field_name):
    """Return the field's values, decoding h5squish attributes if present."""
    dataset = h5_file[field_name]
    if not isinstance(dataset, h5py.Dataset):
        raise ValueError(f"{field_name} is not a dataset")

    attrs = set(dataset.attrs)
    has_lookup = "lookup" in attrs
    quant_attrs = attrs & {"offset", "dx", "seed"}
    if has_lookup and quant_attrs:
        raise ValueError(f"{field_name}: conflicting encoding attributes")
    if quant_attrs and quant_attrs != {"offset", "dx", "seed"}:
        raise ValueError(f"{field_name}: incomplete encoding attributes")
    if (has_lookup or quant_attrs) and dataset.dtype.kind not in "iu":
        raise ValueError(f"{field_name}: encoding needs an integer dataset")

    values = np.asarray(dataset[()])
    if has_lookup:
        lookup = np.asarray(dataset.attrs["lookup"])
        if np.any(values < 0) or np.any(values >= len(lookup)):
            raise ValueError(f"{field_name}: lookup code out of range")
        result = lookup[values.astype(np.intp)]
    elif quant_attrs:
        info = np.iinfo(dataset.dtype)
        imin, imax = int(info.min), int(info.max)
        result = np.empty(values.shape, dtype="float64")
        result[values == imax - 2] = -np.inf
        result[values == imax - 1] = np.inf
        result[values == imax] = np.nan

        finite = values <= imax - 3
        codes = values[finite]
        if dataset.dtype.itemsize <= 4:
            indices = codes.astype("int64") - imin
        else:
            indices = np.fromiter((int(code) - imin for code in codes),
                                  dtype="float64", count=codes.size)
        rng = np.random.default_rng(int(dataset.attrs["seed"]))
        offset = float(dataset.attrs["offset"])
        dx = float(dataset.attrs["dx"])
        if dx == 0:
            result[finite] = offset
        else:
            result[finite] = offset + (indices + rng.random(codes.size)) * dx
    else:
        result = values

    return result[()] if dataset.ndim == 0 else result


def main():
    parser = argparse.ArgumentParser(prog="h5squish")
    commands = parser.add_subparsers(dest="command", required=True)

    for name in ("config", "copy", "replace"):
        command = commands.add_parser(name)
        command.add_argument("--src", type=Path, required=True)
        if name == "copy":
            command.add_argument("--dst", type=Path, required=True)
        if name in ("copy", "replace"):
            command.add_argument("--cfg", type=Path, required=True)

    args = parser.parse_args()
    if not args.src.is_file():
        parser.error(f"source is not a file: {args.src}")
    if args.command == "config":
        write_config(args.src)
        return
    if not args.cfg.is_file():
        parser.error(f"config is not a file: {args.cfg}")
    try:
        rules = read_config(args.cfg, args.src)
    except (OSError, UnicodeError, ValueError) as error:
        parser.error(str(error))

    if args.command == "copy":
        if args.src.resolve() == args.dst.resolve():
            parser.error("source and destination must differ")
        if args.cfg.resolve() == args.dst.resolve():
            parser.error("config and destination must differ")
        if not args.dst.parent.is_dir():
            parser.error(f"destination directory does not exist: "
                         f"{args.dst.parent}")
        try:
            copy_file(args.src, args.dst, rules)
        except (OSError, ValueError) as error:
            parser.error(str(error))
    else:
        if args.src.is_symlink():
            parser.error("replace does not support a source symlink")
        try:
            replace_file(args.src, rules)
        except (OSError, ValueError) as error:
            parser.error(str(error))


if __name__ == "__main__":
    main()
# AI-generated h5squish implementation ends.

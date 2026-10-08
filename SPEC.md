# h5squish conversion contract

This document formalizes `rules.txt` and the decisions in the conversation.

## Scope and commands

`config --src=FILE` writes an ASCII table to standard output.
`copy --src=FILE --dst=FILE --cfg=CFG` writes a new HDF5 file, replacing an
existing destination.
`replace --src=FILE --cfg=CFG` uses the same copy path and then replaces the
source with the completed temporary file. Neither command promises identical
HDF5 bytes: values, types, names, groups, and existing attributes are the
relevant contents.

The supported source objects are nested groups, their attributes, and datasets
of scalar numeric or string values. Unsupported dataset types will be copied
unchanged, but must not be converted silently.

## Config file

Each noncomment row has exactly seven whitespace-separated fields:

| Column | Meaning |
| --- | --- |
| 0 | Absolute HDF5 dataset path, beginning with `/` |
| 1 | Source type: NumPy name for numeric data, `string` for strings |
| 2 | Display offset: minimum finite numeric value, or `------` |
| 3 | Display span: finite maximum minus minimum; unique count for strings |
| 4 | Stored dataset bytes divided by whole-file bytes |
| 5 | Requested new NumPy type, `string`, or `------` |
| 6 | Requested positive accuracy, or `------` |

Blank lines and lines beginning with `#` are ignored. Columns 0–4 describe the
source. Their rounded numbers are informational: encoding recomputes exact
finite bounds from the source and stores full-precision parameters. Only
columns 5 and 6 request a conversion. If both are `------`, the dataset is
copied unchanged, including its existing attributes.

A missing dataset row also means copy unchanged. Duplicate paths,
unknown paths, malformed rows, and a source type that disagrees with column 1
are errors. Dataset paths cannot contain whitespace in this format.

## Conversion dispatch

| Source | New type | Accuracy | Action |
| --- | --- | --- | --- |
| Any | `------` | `------` | Copy unchanged |
| Float | Float | `------` | Cast to requested float type |
| Float | Integer | `------` or value | Quantize as below |
| Float | `------` | Value | Quantize to smallest fitting unsigned type |
| Integer | Integer | `------` | Cast after range validation |
| String | `string` | `------` | Copy unchanged |
| String | Integer | `------` | Encode as lookup indices |
| Any other combination | | | Error |

For integer casts, every source value must fit the requested target type.
Finite float values must not overflow during float-to-float casts. An accuracy
setting on a float-to-float cast is an error; the program does not claim that
casting meets it. A supplied accuracy must be finite and greater than zero.

## Float-to-integer quantization

Let `lo` and `hi` be the exact minimum and maximum of the **finite** source
values, and `span = hi - lo`. A target integer type has inclusive limits
`imin` and `imax`. Its three greatest codes are always reserved:

| Code | Meaning |
| --- | --- |
| `imax - 2` | `-inf` |
| `imax - 1` | `+inf` |
| `imax` | `NaN` |

Finite codes run from `imin` through `imax - 3`; their capacity is
`C = imax - imin - 2` (the number of all integer codes minus three). This
applies even if the source has no non-finite values. For example, `uint8`
has 253 finite codes (`0`–`252`) and the special codes `253`–`255`.

If no accuracy is given, use `N = C` equal-width finite bins. If accuracy `A`
is given, use `N = max(1, ceil(span / A))` bins. The requested integer type
must have `C >= N`; otherwise conversion fails. If no type is requested,
choose the smallest supported unsigned integer type with `C >= N`.

For `span > 0`, set `dx = span / N`. A finite value `x` gets bin index
`j = min(N - 1, floor((x - lo) / dx))`, and stored code `imin + j`. Thus the
bins cover `[lo, hi]` without a gap, and `hi` belongs to the final bin. The
accuracy guarantee is `dx <= A`; a decoded random point can differ from its
input by at most one bin width, apart from floating arithmetic roundoff.
For `span = 0`, use one finite code, `dx = 0`, and decode all finite values
exactly as `lo`. Non-finite values map to their reserved codes, and NaN payloads
are not preserved. If there are no finite values, use
`offset = 0`, `dx = 0`, and encode only the special values (if any).

The output dataset retains its path, shape, and original attributes and adds
full-precision numeric attributes `offset = lo`, `dx`, and `seed`.
Conversion fails if any of those names already exists, rather
than overwriting source metadata.

## String-to-integer lookup

Count each distinct source string, then sort strings by descending frequency.
Break frequency ties by the string value, for deterministic
output. The first string gets code `0`, the next `1`, and so on. The requested
integer type must represent `0` through `unique_count - 1`. Its highest three
codes are **not** reserved; those reservations apply only to quantized floats.

Store the ordered strings in an attribute named `lookup` on the new dataset.
The output dataset retains its path, shape, and other original attributes.
Conversion fails if an original `lookup` attribute exists. The
lookup preserves the original string values and encoding.

## Decoding

Export `decode(h5_file, field_name)`, where `h5_file` is an open `h5py.File`
and `field_name` is an absolute dataset path.

1. An integer dataset with `lookup` decodes each code by indexing that table.
2. An integer dataset with `offset`, `dx`, and `seed` decodes special codes to
   `-inf`, `+inf`, and `NaN`. For finite code `imin + j`, draw `u` uniformly
   from `[0, 1)` and return `offset + (j + u) * dx`. If `dx = 0`, return
   `offset` without a random draw.
3. A dataset with neither encoding is returned unchanged. Conflicting or
   incomplete encoding attributes are errors.

Only quantized float datasets receive `seed`. Generate a fresh
unsigned 64-bit seed when encoding each such dataset and store it as an
attribute. Each call to `decode` initializes its generator from that seed, so
repeated full-dataset decodes return the same random reconstruction.

`decode` should always return a NumPy array (or scalar for scalar datasets).

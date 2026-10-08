# h5squish

`h5squish` rewrites an HDF5 file using dataset types and accuracy limits from
a text config. It currently supports the simple group, attribute, numeric
dataset, and string dataset layouts described in [SPEC.md](SPEC.md).

## Setup

From this directory, create a Python environment and install the two runtime
dependencies:

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install h5py numpy
```

Run the tool as `python3 h5squish.py`. There is no separate package install
step. [Python's `venv` documentation](https://docs.python.org/3/library/venv.html)
and [h5py's installation guide](https://docs.h5py.org/en/stable/build.html)
cover other environments.

## Make a config

```sh
python3 h5squish.py config --src=example.h5 > my.cfg
```

The file has one row per dataset and seven whitespace-separated columns:

```text
# path                 type     offset  span   file fraction  new type  accuracy
/TRUE/dec              float64  -30     1.75   4.851e-03      float32   ------
/TRUE/star_type        string   ------  2      4.851e-03      uint8     ------
/OBS/photometry/g_err  float32  0.01    0.035  2.426e-03      ------    0.01
```

Edit the last two columns. `------` means no request. A row with both fields
unset, or a dataset omitted from the config, is copied unchanged. The first
five columns describe the source; their rounded numbers are informational.
`accuracy` is the maximum width of a float quantization bin. A float dataset
with accuracy set and no new type gets the smallest fitting unsigned integer
type. A string dataset changed to an integer type uses a lookup table.

See [SPEC.md](SPEC.md) for the full conversion rules. `example.cfg` is a
sample config to inspect and edit.

## Copy or replace

```sh
python3 h5squish.py copy --src=example.h5 --dst=copy.h5 --cfg=my.cfg

# Alternatively, change the source file itself:
python3 h5squish.py replace --src=example.h5 --cfg=my.cfg
```

`copy` overwrites its destination. `replace` rewrites its source path. Both
write a temporary file beside the destination and replace it only after the
conversion succeeds. They need temporary disk space for the new file.

## Read converted datasets

```python
import h5py
import h5squish

with h5py.File("copy.h5", "r") as file:
    values = h5squish.decode(file, "/OBS/photometry/g_err")
```

`decode` returns a NumPy array, or a scalar for a scalar dataset. It restores
encoded strings through their lookup table and quantized floats to
reproducible random positions within their bins. Open existing files with
`"r"`; h5py's `"w"` mode truncates them.

## Check the implementation

```sh
python3 -m unittest -v test_h5squish
```

The tests create their own small HDF5 files in temporary directories.

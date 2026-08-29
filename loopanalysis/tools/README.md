# `qoffload-clang++`

A one-command wrapper around the clang → opt → clang → link pipeline that
`test/integration/run_*_integration.sh` each carry out by hand. Opt-in
only: with no `--qpu-pass`, it's a plain passthrough to `clang++`.

## Setup

Every *absolute, machine-specific* path this script needs is a required
environment variable with **no default** — see [Environment](#environment)
below. Assuming a path like `/home/<user>/llvm-project` exists on whatever
machine this runs on is exactly the kind of hidden assumption that breaks
silently elsewhere; the script refuses to guess and errors out immediately
if something it needs isn't set. `$CLANG` is the one exception: it
defaults to `clang++` resolved via `PATH`, since that's portable by
construction, not a machine-specific guess.

`$OPT`/`$LLVM_LINK`/`$CUDAQ_DIR`/`$C2CUDAQ_ROOT` are only checked once at
least one `--qpu-pass` is requested, so a plain passthrough compile
doesn't need the cudaq toolchain configured at all.

## Usage

```bash
export OPT=/path/to/llvm-project/install/bin/opt
export LLVM_LINK=/path/to/llvm-project/install/bin/llvm-link
export CUDAQ_DIR=/path/to/.cudaq
export C2CUDAQ_ROOT=/path/to/c2cudaq
# CLANG defaults to "clang++"; export it only to override.

tools/qoffload-clang++ --qpu-pass=<name>[,<name>...] [--qpu-verbose] <normal clang++ args...>
```

`<name>` is one of `maxcut`, `tsp`, `kcolor`, `clique`, `factor`.

```bash
# Single pass
tools/qoffload-clang++ --qpu-pass=clique test/integration/clique_e2e.cpp -o clique_bin

# Multiple passes -- each is tried independently; only ones that actually
# match anything in the source do anything
tools/qoffload-clang++ --qpu-pass=clique,factor --qpu-verbose test/integration/clique_e2e.cpp -o out

# -c (compile-only) is supported too
tools/qoffload-clang++ --qpu-pass=factor -c test/factor.cpp
```

`--qpu-verbose` prints the pass-manager's own match/replace reporting plus
a summary line per requested pass (`[fired] clique -> @clique_impl` or
`[no match] kcolor`), and reports the final linked binary's path either
way.

## Why this exists

Each of the five `*-pass` plugins needs its own specific `opt` pipeline —
running the wrong one (or the standard `-O1`/`-O2` pipeline) silently
breaks the match. Two examples already hit and documented:
`analysis/clique/clique.md` §1c and `analysis/factor/factor.md` §1c both
found that `indvars` buries the loop-bound argument inside an
`llvm.smax` call, breaking the `NArg` check both passes rely on. This
wrapper exists so a caller doesn't need to know or reconstruct those
pipeline requirements by hand every time — it already knows them.

**Why not just `clang -fpass-plugin=MinPass.so -O1`?** `-fpass-plugin`
only auto-inserts a pass into the pipeline if it's registered at a
pipeline extension point. These five are registered via
`registerPipelineParsingCallback`, so they only run when explicitly named
in a `-passes=` string. More fundamentally, hooking into a standard
`-O1`/`-O2` pipeline wouldn't be safe here regardless of registration
mechanism: `kcolor-pass`/`clique-pass`/`factor-pass` are actively broken
by `indvars`, which standard `-O1`+ runs, while `maxcut-pass`/`tsp-pass`
expect `loop-simplify`/`lcssa`/`indvars` to have already run. The two
pipelines are incompatible, which is why the script keeps two pipeline
"families" and never merges them into one pipeline string.

## Design

| Pass | Pipeline family | Pass-manager kind |
|---|---|---|
| `maxcut` | loop (`loop-simplify,lcssa,indvars,...`) | Function (`FPM.addPass`) |
| `tsp` | loop | Function |
| `kcolor` | reduced (no loop-simplify/lcssa/indvars) | Module (`MPM.addPass`) |
| `clique` | reduced | Module |
| `factor` | reduced | Function |

Two things fall out of this table directly:

1. **Two pipeline families, run as separate `opt` invocations** (reduced
   family first, then loop family, if both are requested) — not one
   merged pipeline string. A source file containing both a
   clique-shaped and a tsp-shaped function (unusual, but not forbidden)
   gets each family's canonicalization applied independently.
2. **Within a family, canonicalization runs once, then each requested
   custom pass runs as its own separate `opt` invocation, chained.** Not
   because it's slow to combine them, but because it's *unsafe* to:
   `kcolor-pass`/`clique-pass` are Module passes, and appending a Module
   pass name to a pipeline string whose context was already established
   as function-only (by the canonicalization passes preceding it) fails
   with `unknown function pass ...` rather than auto-promoting — this bit
   the first version of this script, caught by testing `--qpu-pass=clique`
   directly rather than just factor/tsp (which happen to both be Function
   passes, so the bug didn't show up there). Every `update.py` /
   `run_*_integration.sh` in this project already avoids this by running
   canonicalization and the custom pass as separate `opt` calls; this
   generalizes that to a chain of N custom passes instead of always
   exactly one.

## Known scope limits (v1)

- Only `.cpp`/`.cc`/`.cxx` source inputs; multiple sources are
  `llvm-link`'d together before the QPU passes run (untested beyond the
  single-source case every `test/integration/` script already exercises).
- Extra clang flags (`-std=`, `-I`, `-D`, ...) are forwarded to the
  frontend (source → IR) step only, not to the backend/link steps —
  matches what every `test/integration/run_*_integration.sh` already
  does.
- `-c` is supported and stops after producing the object file; anything
  else (`-S`, `-E`, syntax-only) isn't specially handled and falls
  through to the full compile+link path.
- Linking always pulls in `libc2cudaq.a` and the full cudaq runtime once
  any `--qpu-pass` is requested, whether or not it actually fired —
  `bridge.cpp` is one translation unit, so even a single referenced
  bridge symbol drags in the whole object's cudaq dependencies.

## Environment

No variable below has a default except `CLANG` — each of the others is
checked with `require_env` and the script exits with an explicit error
naming the missing one, rather than silently falling back to an absolute
path that happens to exist on the machine this was written on.

| Variable | Default | Required for | Must point to |
|---|---|---|---|
| `CLANG` | `clang++` (via `PATH`) | every invocation | a clang++ binary (frontend, backend, and link steps) |
| `OPT` | *(none)* | any `--qpu-pass` | the `opt` binary from the LLVM build `MinPass.so` was built against |
| `LLVM_LINK` | *(none)* | any `--qpu-pass` | the matching `llvm-link` binary (used when more than one source file is given) |
| `CUDAQ_DIR` | *(none)* | any `--qpu-pass` | a cudaq install (`$CUDAQ_DIR/lib` is added as an `-L`/rpath at link time) |
| `C2CUDAQ_ROOT` | *(none)* | any `--qpu-pass` | the `c2cudaq` repo root (`$C2CUDAQ_ROOT/build/libc2cudaq.a` must exist, built first) |

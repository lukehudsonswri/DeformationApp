# febio_bin/

A minimal, standalone copy of the `febio4` console solver + its exact
runtime DLL dependencies, so `ik_check/` can run its FEBio solve step
without depending on FEBioStudio being installed system-wide (or on knowing
its install path).

**Not committed to git** (~146 MB of binaries -- see `.gitignore`). If this
folder is empty/missing, `build_hip_flexion_sim.py`'s `run_febio()` falls
back to `febio4` on PATH, then a few common FEBioStudio install locations
(see `_FEBIO_EXECUTABLE_CANDIDATES` in that file) -- so IK_CHECK still works
without this folder, just not "standalone".

## Regenerating this folder

This is deliberately NOT the whole FEBioStudio `bin/` folder (~460 MB,
mostly its GUI/Qt/FFmpeg/CAD libraries the console solver never touches) --
only `febio4.exe`'s actual recursive import-table dependencies, resolved
with the `pefile` package:

```python
import pefile, os

bin_dir = r"C:\Program Files\FEBioStudio\bin"   # your own FEBioStudio install
visited, to_visit, needed = set(), ["febio4.exe"], set()
while to_visit:
    name = to_visit.pop()
    if name.lower() in visited:
        continue
    visited.add(name.lower())
    path = os.path.join(bin_dir, name)
    if not os.path.isfile(path):
        continue
    needed.add(name)
    pe = pefile.PE(path, fast_load=True)
    pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]])
    for entry in getattr(pe, "DIRECTORY_ENTRY_IMPORT", []):
        to_visit.append(entry.dll.decode())

# `needed` is now the exact file list to copy from bin_dir into febio_bin/.
```

As of FEBio 4.12.0, that list is:

```
feamr.dll, febio4.exe, febiofluid.dll, febiolib.dll, febiomech.dll,
febiomix.dll, febioopt.dll, febioplot.dll, febiorve.dll, febioxml.dll,
fecore.dll, feimglib.dll, fftw3.dll, libiomp5md.dll, numcore.dll, zlib1.dll
```

(`numcore.dll` alone is ~123 MB.) `febio4.exe` also links a handful of
Windows system DLLs (`KERNEL32.dll`, `ole32.dll`) and the MSVC runtime
(`MSVCP140.dll`, `VCRUNTIME140*.dll`, `api-ms-win-crt-*.dll`) -- these are
NOT bundled here; they're expected to already be present via the standard
Microsoft Visual C++ Redistributable (already required to run FEBioStudio
itself).

# Security Policy

## Supported Versions

STL Meddler is distributed from the `main` branch; only the latest commit is
supported with security fixes.

## Threat model

STL Meddler is a **local desktop application** with no network surface: it
listens on no ports, makes no outbound connections, and runs no subprocesses.
The realistic attack vector is a **malicious file opened by the user**.

Untrusted inputs (mesh files `*.stl`/`*.obj`, project bundles `*.stlproj`,
snap-list JSON) are treated as hostile:

- decompression budgets bound `.stlproj` zip entries *before* reading
  (zip-bomb defence), with a shared budget across all parts;
- file size and vertex/triangle caps guard the native mesh parsers;
- pickle is disabled for all array loading (`allow_pickle=False`);
- archives are read entry-by-entry and never extracted to disk, so path
  traversal cannot escape them.

These behaviours have regression tests in `tests/test_scene.py`,
`tests/test_load_save.py` and `tests/test_reconstruct.py`.

## Known limits

- Mesh parsing happens inside native C++ libraries (Open3D, trimesh, OCCT).
  A memory-safety bug there is possible despite the budgets above; such
  crashes are usually worth reporting upstream as well.
- Error dialogs may quote raw exception text (file paths, library internals).

## Reporting a vulnerability

Please use [GitHub private vulnerability
reporting](https://github.com/shaneodd/stlmeddler/security/advisories/new)
rather than a public issue. You should receive a response within a few
business days; fixes ship in the next commit on `main`.

"""
DeformationApp -- GUI entry point (Stage 3 of the pipeline, see README.md).

Run this after main_1.py (seat + write .feb) and main_2.py (build the GUI
case) from the repo root:

    .venv\\Scripts\\python.exe viewer_app.py

This is a thin launcher so the GUI can be started the same way as
main_1.py/main_2.py, without needing to remember the `-m app.viewer_app`
module-invocation form. The real implementation lives in
`app/viewer_app.py`.
"""
from app.viewer_app import main

if __name__ == "__main__":
    main()

"""
pipeline/__main__.py
CLI entrypoint for running the pipeline as a module:
    python -m pipeline --help
"""

from .orchestrator import main

if __name__ == "__main__":
    main()


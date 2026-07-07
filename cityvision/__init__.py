"""CityVision shared package: single source of truth for the class taxonomy,
colour palette, and model architectures.

Kept intentionally minimal — importing this package must NOT pull in torch, so
that torch-free consumers (e.g. the Streamlit frontend) can do
``from cityvision.constants import PALETTE`` cheaply. Model classes (which need
torch) live in ``cityvision.models`` and are imported explicitly.
"""

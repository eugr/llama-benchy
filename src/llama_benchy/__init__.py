"""
llama-benchy - llama-bench style benchmarking tool for all backends

This package provides a benchmarking tool for OpenAI-compatible LLM endpoints,
generating statistics similar to `llama-bench`.
"""

from ._version import __version__

# Extract build number from the version string
# Version format is like: '0.1.dev34+g33f03d886.d20260105'
# We want to extract the git hash part: '33f03d886'
__build__ = "unknown"
if "+" in __version__:
    try:
        # Extract the part after the '+' and before the '.'
        build_part = __version__.split("+")[1].split(".")[0]
        # Remove the 'g' prefix if it exists
        if build_part.startswith("g"):
            __build__ = build_part[1:]
        else:
            __build__ = build_part
    except (IndexError, AttributeError):
        pass
else:
    # For release versions (no '+'), set build to empty string
    __build__ = ""
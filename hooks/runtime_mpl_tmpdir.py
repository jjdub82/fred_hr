# hooks/runtime_mpl_tmpdir.py
import os
import tempfile

# Force matplotlib to write config files to a temporary folder instead of the user profile.
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mplconfig"))

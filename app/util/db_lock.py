from __future__ import annotations

import threading


# Process-wide lock for serialized SQLite write sections that are vulnerable
# to contention when invoked from threaded workloads.
DB_WRITE_LOCK = threading.Lock()


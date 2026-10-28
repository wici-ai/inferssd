# 0003: Admit real allocations before first touch

Status: accepted.

Linux contexts share one process arbiter and register separate budget clients.
Tickets cover full new allocations, including first touch, and are committed
only after materialization. Known frees/replacements invalidate observation
intervals. CPU expert cache pages, graph buffers, staging and server state use
explicit ownership. CUDA allocations count as host demand on unified memory.
Linux hard limits and swap restrictions remain real OS constraints; a settings
receipt does not establish resource enforcement. Mac keeps its RSS-based guard.

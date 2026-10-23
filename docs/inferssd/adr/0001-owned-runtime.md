# 0001: Own configuration and services in the context

Status: accepted for the private v0.1 candidate.

L3 copies validated settings, registers a host-budget client, opens an immutable
source index and writer, and binds backend providers through ordinary registries.
Backend API tables carry versions and structure sizes; executors borrow services.
No provider lookup uses process symbol visibility. Context lifetime encloses all
borrowers, including constructor failure. This preserves independent backend
builds and prevents one request from selecting another request's sources.

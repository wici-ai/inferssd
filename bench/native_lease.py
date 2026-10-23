"""Compatibility entry for the single shared, declared ownership implementation."""
from native_core import CORE

peer_identity = CORE["lease"].peer_identity
LeaseServer = CORE["lease"].LeaseServer
Lease = CORE["lease"].Lease

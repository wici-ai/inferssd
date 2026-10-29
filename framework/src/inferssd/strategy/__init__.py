# SPDX-License-Identifier: Apache-2.0
"""Launch strategy: the native settings and server arguments for one launch.

A strategy is derived on every launch from the machine characteristics file,
the model's GGUF header and facts read when the launch starts. It only
proposes values: a value the user set always wins. Importing this package
performs no observations.
"""

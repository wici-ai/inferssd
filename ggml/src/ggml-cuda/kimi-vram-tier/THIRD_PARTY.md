# Reference float math notices

`../kimi-q3k-reference-math.cuh` is test2's CUDA translation of glibc 2.35 / fdlibm float routines. The original Sun Microsystems and glibc notices remain in that source file. The glibc expf implementation and table are licensed under LGPL-2.1-or-later; the LGPL 2.1 text is in [COPYING.LIB](COPYING.LIB). These notices apply to that component independently of the surrounding repository license.

The accepted source is supplied together with the standalone component build script. Strict rounding, FTZ-off behavior and the original notices must be preserved when editing or rebuilding the reference implementation.

# Framework tests

Most tests here use synthetic inputs and never start a model. One group compares
the package with **F**, the original implementation recorded before it was
refactored into `framework/src/inferssd`:

- `fixtures/oracle/` holds byte-for-byte copies of the original sources
  (`*_f.py` and others). Each `fixtures/oracle/*-manifest.json` records their
  SHA-256 hashes, and the tests fail if a byte changes.
- `observe_*_F.py` run that recorded code on synthetic inputs and capture every
  effect as a transcript. `fixtures/*-goldens.json` and `fixtures/*-F.json` hold
  the expected transcripts.
- `test_*_F.py` check that the recorded code still reproduces its transcripts;
  the other tests in the same area check that the package produces the same
  transcripts or results.

Do not edit a file whose hash appears in a manifest. Change the package and its
own tests instead. The recorded code still uses the project's historical
environment names (`SSDASH_*`); the package uses `INFERSSD_*`.

The transcripts were recorded on Python 3.9 and 3.12. On other versions these
checks skip, so run the full suite on one of those two versions.

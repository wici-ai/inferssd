# The macOS package

On an Apple silicon Mac with macOS 15 or later, the installer on `main` installs a prebuilt
package of `main` rather than building from source, so the Mac needs no developer tools: no
Xcode or Command Line Tools, Git, Python, CMake or compiler.

```sh
curl -fsSL https://raw.githubusercontent.com/wici-ai/inferssd/main/scripts/install.sh | sh
```

`INFERSSD_FROM_SOURCE=1` or a branch or tag in `INFERSSD_REF` builds from source instead, and
an installer from a release builds that release from source, as before. So does the installer on
`main` while no package is published, when the package list answers 404.

## What it is

One archive, `inferssd-<version>-macos-arm64.tar.gz`, about 30 MB, 80 MB unpacked, named by the
build's [version](engine-resources.md), `<yymmdd>.<build>` (`inferssd-261119.301-macos-arm64`). A
package built outside a release, by hand, is never published:

```text
inferssd-<version>-macos-arm64/
  bin/inferssd               the command, on the package's own Python
  bin/inferssd-server        the engine, its libraries beside it
  bin/inferssd-cpuprobe, bin/inferssd-ssdprobe
  python/                    CPython 3.12 with the framework and Hugging Face support
  templates/                 the models' chat templates
  engine-resources.json      the engine descriptor, with paths relative to it
  activate                   makes this directory the installed engine
  licenses/
```

It runs wherever it is unpacked: every binary loads only macOS's libraries and the package's
own, found beside it. The engine is built for the M1, which every Apple silicon CPU runs, rather
than for the building Mac's CPU as a source build is. On an M5, the machine profile's CPU
measurements of the two builds match at one token, which is what decode runs; in batches they
differ by format, one way or the other. The Python packages are pinned by hash in
`scripts/package-macos-requirements.txt`.

`activate` does what the installer's last step does for a source build: it points
`$INFERSSD_HOME/current` at the package and writes the `inferssd` command into
`$INFERSSD_BIN_DIR`. The installer keeps the installation it replaces and removes older ones.

## How it is made

```sh
scripts/package-macos.sh dist          # on an Apple silicon Mac with Xcode, CMake and Ninja
scripts/check-package-macos.sh dist/inferssd-*-macos-arm64.tar.gz
```

The first needs a clone with `main`'s history, since the build number is its commit count. It
writes the archive and a `.json` that describes it. The second unpacks the archive under a path
with a space, checks every binary's libraries, search paths, signature and oldest macOS, then
installs it with `scripts/install.sh` from a package list of its own, with every developer tool
replaced by a stub that fails, and profiles the machine.

## Releases

Every published package is a release's. **Actions > Release**, run by hand on `main`, releases
its head as its version, `<yymmdd>.<build>`: the commit's day (UTC) and `main`'s commit count, as
every build is named. It calls **Package macOS**, and **Package Windows** once for Vulkan and once
for CUDA, at that commit, with the release's tag, which is the version, and each package is checked
against it. macOS builds on macOS 15 and checks on macOS 15 and 26; Windows also checks
installation without developer tools, and signs (see [the Windows package](windows-package.md)).
Each package, once its own checks pass, reaches the CDN and its package list points at it
(`scripts/publish-package.py`, which publishes nothing but a release's package): none waits for
the slowest, so the macOS and Vulkan packages reach the app while the CUDA one still builds. Once
all three are published, the release job attaches the archives, manifests and pinned installers to
the GitHub release named by the version, which tags the commit when it is published. A version
that is already a tag stops the run before anything is built. `APP_VERSIONING_TOKEN` is required.

The release's notes list the pull requests merged since the last release, by their titles; a
release with highlights of its own has a section in [CHANGELOG.md](../CHANGELOG.md), which comes
first.

With `github_release` off, a run builds, checks and signs the same packages and points the package
lists at them, which the installers from `main` and the desktop app follow, but makes no GitHub
release and no tag, so no notes or pinned installers are published. The CDN then holds the
version, and a release built again at the same commit would be refused there: the next GitHub
release needs a newer commit on `main`. The same holds for a run whose later package fails: the
packages already published stay, and the failed one's list keeps its earlier version.

**Package macOS** also runs by hand on any branch, building and checking without publishing; run it for
a change to how the package is built. Pull requests and merges to `main` build no package. A run
produces the `package-macos` Actions artifact.

Each release's installers pin their package lists to that same GitHub release. The installer
from `main` follows the latest package list; explicit `INFERSSD_REF` or `INFERSSD_FROM_SOURCE`
requests a source build.

**Package macOS** also runs by hand on any branch, building and checking without publishing; run
it for a change to how the package is built. Pull requests and merges to `main` build no package. The publisher uploads to
`https://cdn.wici.ai/apps/inferssd/`, checks the served bytes before updating the list, and
refuses a list older than the one already published.

The CDN keeps a file for a year, so a name never gets other bytes: a version already uploaded is
not uploaded again with other bytes. If publishing fails, re-run the release's failed job: it
publishes the same files, and the service skips the ones it holds.

## The package list

The installer reads `https://appapi.wici.ai/versioning/inferssd/config/latest-macos.json`
(`INFERSSD_PACKAGES` names another), with `plutil`, as macOS has it:

```json
{
  "schema": "inferssd-package 1",
  "name": "inferssd-261119.301-macos-arm64",
  "version": "261119.301",
  "tag": "261119.301",
  "build": 301,
  "revision": "…",
  "date": "2026-11-19",
  "branch": "main",
  "label": "261119.301.61a66d5",
  "platform": "darwin-arm64",
  "minimum_macos": "15.0",
  "file": "inferssd-261119.301-macos-arm64.tar.gz",
  "size": 27751636,
  "sha256": "…",
  "url": "https://cdn.wici.ai/apps/inferssd/inferssd-261119.301-macos-arm64.tar.gz"
}
```

It downloads `url`, refuses it unless its SHA-256 is `sha256`, unpacks it into a new directory
under `releases` and runs `activate`. A failed download or check leaves the installed engine as
it was. Whatever runs the engine can read the same list to learn of a newer `build`.

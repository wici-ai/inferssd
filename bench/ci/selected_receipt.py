"""Observe current typed engines, immutable inputs and real loaded images.

No old-release mapping or declared-only runtime receipt is accepted here.
Linux uses root /proc observations; macOS uses the root executor's exact launch,
birth identity and lsof image inventory under its continuously enforced policy.
"""

import os
import csv
import io
from pathlib import Path
import re

from ci.contract import digest, load, save
from ci.staged_build import sealed_file
from ci.staged_checkpoint import file_identity, mapped_files, image_identity, native_settings
from ci.staged_benchmark import verify_runtime


def normalize(value, frozen):
    if isinstance(value, str):
        return value.replace(frozen + "/", "<sealed-build>/")
    if isinstance(value, list):
        return [normalize(item, frozen) for item in value]
    if isinstance(value, dict):
        return {key: normalize(item, frozen) for key, item in value.items()}
    return value


def mac_images(raw, pid):
    current_pid, descriptor, paths = None, None, set()
    for line in raw.splitlines():
        if line.startswith("p"):
            current_pid, descriptor = int(line[1:]), None
        elif line.startswith("f"):
            descriptor = line[1:]
        elif (line.startswith("n/") and current_pid == pid
              and descriptor is not None and (descriptor == "txt" or descriptor.startswith("mem"))):
            paths.add(line[1:])
    return paths


def receipt(config, manifest, side, model, spec, build, output, *, destination=None):
    output = Path(output)
    before, after = (load(output / (name + ".json")) for name in ("runtime-before", "runtime-after"))
    started = load(output / "started.json")
    for observed in (before, after):
        if any(observed.get(key) != started[key] for key in (
                "worker_pid", "worker_uid", "birth_identity", "scope_identity")):
            raise ValueError("runtime PID, birth, UID or resource scope changed")
        # The recorder already checks exact listener ownership, retained here too.
        class Identity:
            pass
        identity = Identity()
        for key, value in started.items():
            setattr(identity, key, value)
        verify_runtime(identity, observed)
    platform = config["paths"]["platform"]
    host = config["host"]
    uname = os.uname()
    if host["os"] != dict(sysname=uname.sysname, release=uname.release, machine=uname.machine):
        raise ValueError("operator OS identity differs from actual host")
    if not host["hardware"]:
        raise ValueError("root-observed device hardware identity required")
    resources = host["resources"]
    data = config["sides"][side]["models"][model]
    model_identity = data["model_identity"]
    readonly_data = {item["runtime_path"] for item in model_identity["shards"]}
    readonly_data.update(data[key]["runtime_path"] for key in ("settings", "template"))
    paths = []
    if platform == "linux":
        ambient = None
        for observed in (before, after):
            if (observed["cmdline"] != spec["argv"]
                    or observed["executable"] != build["binary"]["runtime_path"]
                    or observed["executable_sha256"] != build["binary"]["sha256"]
                    or observed["cpus_allowed_list"] != resources["cpus_allowed_list"]
                    or observed["cgroup_limits"] != resources["cgroup_limits"]):
                raise ValueError("actual argv, executable or enforced resources differ")
            limit = re.search(r"^Max locked memory\s+(\S+)\s+(\S+)\s+bytes\s*$", observed["proc_limits"], re.M)
            if not limit or limit.groups() != (str(resources["memlock_bytes"]),) * 2:
                raise ValueError("actual memory lock limit differs")
            environment = dict(observed["environment"])
            if environment.pop("PWD", None) != spec["cwd"]:
                raise ValueError("actual working directory differs")
            for key, value in spec["environment"].items():
                if environment.pop(key, None) != value:
                    raise ValueError("actual explicit environment differs")
            if ambient is not None and ambient != environment:
                raise ValueError("ambient environment changed during the request")
            ambient = environment
            paths.append(mapped_files(observed))
    else:
        # Root executor policy is in the hash-bound job manifest, not supplied by
        # candidate output. A legacy manifest without it cannot pass performance.
        if manifest["selection_resource_policy"] != resources:
            raise ValueError("Mac root enforcement policy differs from the compared envelope")
        ambient = dict(spec["environment"])
        for observed in (before, after):
            if observed["argv"] != spec["argv"]:
                raise ValueError("root executor launch differs from the reviewed native argv")
            if (observed.get("members") is None or observed.get("resource_policy") != resources
                    or observed["rss_bytes"] > resources["memory_budget_bytes"]):
                raise ValueError("Mac process memory inventory missing or over budget")
            # lsof labels mmap'ed GGUF inputs as txt too. They are separately
            # verified readonly data, never an executable image or system DSO.
            paths.append({path: None for path in mac_images(observed["lsof"], started["worker_pid"])
                          if path not in readonly_data})
    if set(paths[0]) != set(paths[1]):
        raise ValueError("actual executable image set changed during completion")
    frozen = config["sides"][side]["paths"]["frozen"]
    all_images = [build["binary"], *build["images"]]
    loaded = []
    for image in all_images:
        allowed = {image["runtime_path"]}
        if platform == "linux":
            root = Path(build["freeze"]["host_path"])
            allowed |= {str(Path(frozen) / Path(item).relative_to(root)) for item in image.get("aliases", [])}
        actual = allowed & set(paths[0])
        if len(actual) > 1:
            raise ValueError("multiple mappings for one sealed image")
        if not actual:
            if image is build["binary"]:
                raise ValueError("actual server image missing from root runtime inventory")
            continue
        path = actual.pop()
        if platform == "linux":
            for observed in paths:
                image_identity(image["path"], observed[path])
        loaded.append(dict(path=path, sha256=image["sha256"]))
    known = {item["path"] for item in loaded}
    systems = []
    roots = ("/usr/", "/lib/", "/lib64/", "/System/Library/", "/Library/Apple/", "/opt/inferssd-ci-toolchains/")
    for path in sorted(set(paths[0]) - known):
        if not path.startswith(roots):
            raise ValueError("unreviewed loaded image outside sealed build/system roots")
        if platform == "linux":
            from ci.staged_checkpoint import system_image
            systems.append(system_image(path, paths[0][path]))
        else:
            # Apple shared-cache entries may lack a standalone file. Their path
            # inventory is bound to the exact root-observed OS build above.
            systems.append(dict(path=path, sha256=digest(path) if Path(path).is_file() else "os-shared-cache"))
    for shard in model_identity["shards"]:
        if file_identity(shard["host_path"]) != shard["stat"]:
            raise ValueError("root-bound readonly model shard changed")
    template = sealed_file(data["template"]["host_path"])
    settings, settings_raw = sealed_file(data["settings"]["host_path"], contents=True)
    if (template["sha256"] != data["template"]["sha256"]
            or settings["sha256"] != data["settings"]["sha256"]):
        raise ValueError("typed settings or template changed")
    from inferssd.configuration.settings import parse
    expected = parse(settings_raw.decode(), data["settings"]["runtime_path"], platform=platform).document()["values"]
    native = native_settings(output / "stderr.log")
    if native["values"] != expected:
        raise ValueError("actual native settings differ from root-sealed configuration")
    shard_sizes = {item["runtime_path"]: item["stat"]["bytes"] for item in model_identity["shards"]}
    model_flags = [flag for flag in ("-m", "--model") if flag in spec["argv"]]
    if (len(model_flags) != 1 or spec["argv"].count(model_flags[0]) != 1
            or spec["argv"][spec["argv"].index(model_flags[0]) + 1] not in shard_sizes):
        raise ValueError("actual model argument differs from the root-bound shard set")
    for source in expected["expert.source"]:
        if source in shard_sizes:
            continue
        relative = Path(source).relative_to(frozen)
        _, alias_raw = sealed_file(Path(build["freeze"]["host_path"]) / relative, contents=True)
        rows = list(csv.DictReader(io.StringIO(alias_raw.decode()), delimiter="\t"))
        if not rows:
            raise ValueError("empty expert source alias")
        for row in rows:
            size, offset = int(row["nbytes"]), int(row["offset"])
            if (row["source_path"] not in shard_sizes or int(row["expert_idx"]) < 0
                    or size <= 0 or offset < 0 or offset + size > shard_sizes[row["source_path"]]):
                raise ValueError("expert source alias escapes the root-bound model shards")
    args, index = [], 1
    argv = spec["argv"]
    substitutions = {"--ssd-settings": data["settings"]["runtime_path"],
                     "--chat-template-file": data["template"]["runtime_path"]}
    for flag, path in substitutions.items():
        if argv.count(flag) != 1 or argv[argv.index(flag) + 1] != path:
            raise ValueError("native launch must consume the exact protected asset")
    while index < len(argv):
        if argv[index] in substitutions or argv[index] == "--ssd-telemetry":
            args += [argv[index], "<protected-asset>" if argv[index] != "--ssd-telemetry" else "<private-events>"]
            index += 2
        else:
            args.append(argv[index])
            index += 1
    result = dict(model_identity=model_identity,
                  build=dict(revision=config["sides"][side]["revision"], loaded_libraries=loaded),
                  server=dict(pid=started["worker_pid"], start_identity=started["birth_identity"]),
                  host=dict(hardware=host["hardware"], os=host["os"],
                            memory_budget_bytes=resources.get("memory_budget_bytes", int(resources.get("cgroup_limits", {}).get("memory_max", 0))),
                            os_page_cache_treatment=spec["os_page_cache_treatment"],
                            effective_settings=dict(native_launch_policy=normalize(dict(args=args, typed_values=expected,
                                template_sha256=template["sha256"]), frozen), native_resource_policy=resources,
                                ambient_environment=ambient, system_images=systems)))
    save(destination if destination is not None else output / "comparison-receipt.json", result)
    return result

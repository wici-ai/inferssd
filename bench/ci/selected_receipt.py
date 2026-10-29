"""Observe current typed engines, immutable inputs and real loaded images.

No old-release mapping or declared-only runtime receipt is accepted here.
Linux uses root /proc observations; macOS uses the root executor's exact launch,
birth identity and lsof image inventory under its continuously enforced policy.
"""

import os
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
from types import SimpleNamespace

from ci.contract import digest, load, save
from ci.staged_build import sealed_file
from ci.staged_checkpoint import mapped_files, image_identity, native_settings
from ci.model_admission import checked_model
from ci.staged_benchmark import verify_runtime, verify_stopped


MAC_MEMORY_METRIC = "physical-footprint"
MAC_RESOURCE_ENFORCEMENT = "root-sampled-worker-physical-footprint-processes-host-reserve"


def mac_job_root(manifest_path):
    match = re.fullmatch(r"/Library/InferSSD-CI/private/job-manifests/([0-9a-f]{32})\.json", str(manifest_path))
    if match is None:
        raise ValueError("Mac receipt requires the fixed current root job manifest")
    return "/Library/InferSSD-CI/jobs/" + match[1]


def mac_stopped(config, manifest, stopped):
    """A clean requested stop still needs this job's UID/domain/lease receipt."""
    verify_stopped(stopped)
    root = mac_job_root(config["manifest_path"])
    cleanup = stopped.get("cleanup", {})
    if not isinstance(cleanup, dict):
        raise ValueError("Mac cleanup inventory is missing")
    retired = cleanup.get("worker_user_domain_retired")
    if (stopped.get("job_id") != PurePosixPath(root).name
            or stopped.get("job_root") != root or stopped.get("scope_identity") != root
            or stopped.get("worker_uid") != 62002
            or stopped.get("manifest_sha256") != config["manifest_sha256"]
            or stopped.get("lock_dev_inode") != manifest["selection_lock_identity"]
            or cleanup.get("confirmed") is not True or cleanup.get("remaining") != []
            or cleanup.get("error")
            or not isinstance(retired, dict) or retired.get("returncode") != 0):
        raise ValueError("Mac stop lacks exact job/lease and empty worker-domain cleanup")


def mac_memory_sample(observed, resources):
    """Verify the root executor's live footprint sum and exited-PID partition."""
    members = observed.get("members")
    samples = observed.get("memory_members")
    exited = observed.get("memory_exited_pids")
    footprint = observed.get("physical_footprint_bytes")
    if (observed.get("memory_metric") != MAC_MEMORY_METRIC
            or "memory_accounting_error" in observed
            or observed.get("resource_violation") is not None
            or type(footprint) is not int or footprint < 0
            or observed.get("resource_policy") != resources
            or not isinstance(members, list) or not members or len(members) > resources["max_processes"]
            or not isinstance(samples, list) or not isinstance(exited, list)):
        raise ValueError("Mac physical footprint accounting is missing or invalid")
    inventory = {}
    for row in members:
        if (not isinstance(row, dict) or type(row.get("pid")) is not int or row["pid"] <= 1
                or row["pid"] in inventory
                or type(row.get("rss_bytes")) is not int or row["rss_bytes"] < 0):
            raise ValueError("Mac worker memory inventory differs")
        inventory[row["pid"]] = row
    if (type(observed.get("rss_bytes")) is not int
            or observed["rss_bytes"] != sum(row["rss_bytes"] for row in members)):
        raise ValueError("Mac diagnostic RSS sum differs")
    live, total = {}, 0
    for row in samples:
        if not isinstance(row, dict):
            raise ValueError("Mac footprint member is invalid")
        pid, birth = row.get("pid"), row.get("birth_identity")
        if (type(pid) is not int or pid not in inventory or pid in live
                or type(row.get("rss_bytes")) is not int or row["rss_bytes"] != inventory[pid]["rss_bytes"]
                or not isinstance(birth, dict) or birth.get("pid") != pid or birth.get("uid") != 62002
                or any(type(birth.get(key)) is not int for key in ("pid", "uid", "start_sec", "start_usec"))
                or birth["start_sec"] <= 0 or not 0 <= birth["start_usec"] < 1000000
                or any(type(row.get(key)) is not int or row[key] < 0 for key in (
                    "physical_footprint_bytes", "resident_bytes", "start_abstime", "exit_abstime"))
                or row["start_abstime"] == 0 or row["exit_abstime"] != 0):
            raise ValueError("Mac live footprint member identity or accounting differs")
        live[pid] = row
        total += row["physical_footprint_bytes"]
    if (any(type(pid) is not int or pid not in inventory for pid in exited)
            or len(set(exited)) != len(exited) or set(exited) & live.keys()
            or live.keys() | set(exited) != inventory.keys()
            or observed.get("worker_pid") not in live
            or live[observed["worker_pid"]]["birth_identity"] != observed.get("birth_identity")):
        raise ValueError("Mac live/exited memory membership differs")
    if footprint != total or footprint > resources["memory_budget_bytes"]:
        raise ValueError("Mac physical footprint sum or budget differs")
    return footprint


def mac_runtime_images(config, manifest, side, model, spec, started, observations):
    """Recheck root Mac samples without inventing a Linux hard memory limit.

    The executor monitors worker physical footprint/process count and host reserve.
    RSS is retained only as a diagnostic, including reclaimable file mappings.
    These snapshots attest that sampled policy, not memory.max or swap=0.
    """
    root = mac_job_root(config["manifest_path"])
    resources = config["host"]["resources"]
    keys = {"max_processes", "memory_budget_bytes", "host_reserve_bytes"}
    lease = manifest.get("selection_lock_identity")
    if (set(resources) != keys or any(type(resources[k]) is not int or resources[k] <= 0 for k in keys)
            or manifest.get("selection_resource_policy") != resources
            or not isinstance(lease, list) or len(lease) != 2
            or any(type(v) is not int or v < 0 for v in lease)
            or config["sides"][side]["paths"]["job_root"] != root):
        raise ValueError("Mac root resource policy or original lease is missing")
    data = config["sides"][side]["models"][model]
    readonly = {row["runtime_path"] for row in data["model_identity"]["shards"]}
    readonly.update(data[key]["runtime_path"] for key in ("settings", "template"))
    paths = []
    for observed in observations:
        verify_runtime(SimpleNamespace(**started), observed)
        birth = observed["birth_identity"]
        members = observed.get("members")
        if (observed.get("worker_uid") != 62002 or not isinstance(birth, dict)
                or birth.get("uid") != 62002 or birth.get("pid") != started["worker_pid"]
                or any(type(birth.get(k)) is not int or birth[k] < 0 for k in ("start_sec", "start_usec"))
                or birth["start_usec"] >= 1000000 or not birth["start_sec"]
                or observed.get("job_id") != PurePosixPath(root).name
                or observed.get("job_root") != root or observed["scope_identity"] != root
                or observed.get("manifest_sha256") != config["manifest_sha256"]
                or observed.get("lock_dev_inode") != lease
                or observed.get("argv") != spec["argv"]
                or observed.get("resource_policy") != resources
                or not isinstance(members, list) or not members
                or len(members) > resources["max_processes"]):
            raise ValueError("Mac PID/birth/job/lease/launch/resource sample differs")
        mac_memory_sample(observed, resources)
        if (observed.get("listener_pids") != [started["worker_pid"]]
                or not isinstance(observed.get("lsof"), str)):
            raise ValueError("Mac worker listener/image inventory differs")
        current, owner = None, []
        for line in observed["lsof"].splitlines():
            if line.startswith("p"):
                current = int(line[1:])
            elif line.startswith("u") and current == started["worker_pid"]:
                owner.append(line[1:])
        if owner != ["62002"]:
            raise ValueError("Mac lsof inventory does not bind the execution UID")
        paths.append({path: None for path in mac_images(observed["lsof"], started["worker_pid"])
                      if path not in readonly})
    if not paths or any(paths[0] != observed for observed in paths[1:]):
        raise ValueError("Mac executable image set changed during completion")
    return paths


def normalize(value, frozen):
    if isinstance(value, str):
        return value.replace(frozen + "/", "<sealed-build>/")
    if isinstance(value, list):
        return [normalize(item, frozen) for item in value]
    if isinstance(value, dict):
        return {key: normalize(item, frozen) for key, item in value.items()}
    return value


def retained_typed_settings(native, expected, side, model, build, freeze_manifest, manifest_path):
    """Compare a native snapshot to its receipt without broad path rewriting.

    Only a source in the fixed base/candidate sandbox may share its receipt's sealed
    spelling, and only when that exact asset is in the root freezer's hash-bound
    inventory. Other typed values and direct model-shard paths remain exact.
    """
    if isinstance(native, dict) and isinstance(expected, dict) and native == expected:
        return True
    if (not isinstance(native, dict) or not isinstance(expected, dict) or set(native) != set(expected)
            or any(native[key] != expected[key] for key in native if key != "expert.source")):
        return False
    if "expert.source" not in native:
        return True
    actual, reviewed = native["expert.source"], expected["expert.source"]
    if (not isinstance(actual, list) or not isinstance(reviewed, list) or not actual
            or len(actual) != len(reviewed)
            or any(not isinstance(value, str) or not value or "\0" in value
                   for value in [*actual, *reviewed])):
        return False
    if any("\\" in value or str(PurePosixPath(value)) != value
           or any(part in (".", "..") for part in value.split("/")) for value in [*actual, *reviewed]):
        return False
    shards = {item["runtime_path"] for item in side["models"][model]["model_identity"]["shards"]}
    assets = None
    for observed, reference in zip(actual, reviewed):
        if observed == reference and observed.startswith("/") and observed in shards:
            continue
        if assets is None:
            frozen = build["freeze"]
            host = frozen.get("host_path", frozen["path"])
            if str(manifest_path).startswith("/Library/InferSSD-CI/"):
                root = mac_job_root(manifest_path)
                runtime_root = side["paths"]["frozen"]
                frozen_side = PurePosixPath(runtime_root).name
                if (frozen_side not in ("base", "candidate")
                        or side["paths"].get("job_root") != root
                        or runtime_root != root + "/frozen/" + frozen_side
                        or frozen["path"] != runtime_root or host != runtime_root
                        or frozen.get("manifest_path") != root + "/frozen/" + frozen_side + ".manifest.json"
                        or frozen.get("readonly") is not True
                        or digest(freeze_manifest) != frozen["manifest_sha256"]):
                    raise ValueError("retained Mac source does not bind the fixed side freeze")
                inventory = load(freeze_manifest)
                if (not isinstance(inventory, dict) or type(frozen.get("file_count")) is not int
                        or not frozen["file_count"] or len(inventory) != frozen["file_count"]):
                    raise ValueError("retained Mac source freezer inventory differs")
                records = [dict(row, path=name) for name, row in inventory.items()]
            else:
                frozen_side = frozen.get("side")
                runtime_root = "/build/" + str(frozen_side)
                if (not isinstance(manifest_path, str) or not re.fullmatch(
                    r"/var/lib/inferssd-ci-broker/jobs/[0-9a-f]{32}/manifest\.json", manifest_path)
                    or frozen_side not in ("base", "candidate")
                    or side["paths"]["frozen"] != runtime_root
                    or frozen["path"] != runtime_root or not isinstance(host, str)
                    or host != str(PurePosixPath(manifest_path).parent / "frozen" / frozen_side)):
                    raise ValueError("retained source does not bind the fixed side freeze")
                records = load(freeze_manifest)["files"]
                encoded = json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
                if (hashlib.sha256(encoded).hexdigest() != frozen["manifest_sha256"]
                        or len(records) != frozen["files"]):
                    raise ValueError("retained source root freezer inventory differs")
            assets = {}
            for row in records:
                name = row["path"]
                if (not isinstance(name, str) or not name or name.startswith("/") or "\\" in name
                        or any(part in ("", ".", "..") for part in name.split("/")) or name in assets):
                    raise ValueError("retained source freezer member is unsafe or duplicated")
                assets[name] = row
        names = []
        for value in (observed, reference):
            prefix = next((p for p in (runtime_root + "/", "<sealed-build>/") if value.startswith(p)), None)
            name = value[len(prefix):] if prefix else ""
            if (not name.startswith("assets/") or "\\" in name
                    or any(part in ("", ".", "..") for part in name.split("/"))):
                return False
            names.append(name)
        if names[0] != names[1] or names[0] not in assets:
            return False
        row = assets[names[0]]
        if (not isinstance(row.get("size"), int) or isinstance(row["size"], bool) or row["size"] <= 0
                or not isinstance(row.get("sha256"), str) or not re.fullmatch("[0-9a-f]{64}", row["sha256"])):
            raise ValueError("retained source asset lacks its sealed file identity")
    return True


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
        ambient = dict(spec["environment"])
        paths = mac_runtime_images(config, manifest, side, model, spec, started, (before, after))
        mac_stopped(config, manifest, load(output / "stopped.json"))
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
    model_identity = checked_model(config, manifest, model, data)
    template = sealed_file(data["template"]["host_path"])
    settings, settings_raw = sealed_file(data["settings"]["host_path"], contents=True)
    if (template["sha256"] != data["template"]["sha256"]
            or settings["sha256"] != data["settings"]["sha256"]):
        raise ValueError("typed settings or template changed")
    from inferssd.configuration.settings import parse
    reviewed = parse(settings_raw.decode(), data["settings"]["runtime_path"], platform=platform).document()
    # A derived launch runs the settings its side's own strategy resolved on the
    # device; the reviewed file is only that resolution's input.
    resolved = data.get("derived")
    expected = reviewed["values"] if resolved is None else resolved["effective"]["values"]
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
        frozen_host = build["freeze"].get("host_path", build["freeze"]["path"])
        _, alias_raw = sealed_file(Path(frozen_host) / relative, contents=True)
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
    substitutions = {"--ssd-settings": data["settings"]["runtime_path"] if resolved is None else resolved["settings"],
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
    policy = dict(args=args, typed_values=expected, template_sha256=template["sha256"])
    if resolved is not None:
        # Equal on both sides: what the launch was asked for, not what it derived.
        policy["request"] = dict(data["launch"], settings={key: reviewed["values"][key] for key, source
                                                           in sorted(reviewed["sources"].items()) if source["origin"] != "default"})
    result = dict(model_identity=model_identity,
                  build=dict(revision=config["sides"][side]["revision"], loaded_libraries=loaded),
                  server=dict(pid=started["worker_pid"], start_identity=started["birth_identity"]),
                  host=dict(hardware=host["hardware"], os=host["os"],
                            memory_budget_bytes=resources.get("memory_budget_bytes", int(resources.get("cgroup_limits", {}).get("memory_max", 0))),
                            os_page_cache_treatment=spec["os_page_cache_treatment"],
                            effective_settings=dict(native_launch_policy=normalize(policy, frozen), native_resource_policy=resources,
                                ambient_environment=ambient, system_images=systems)))
    if resolved is not None:
        from ci.staged_launch import sealed, summary
        result["derived_launch"] = summary(config, side, model, resolved)
        # A PR job's launches derive from the characteristics sealed into the job.
        value = sealed(config)
        if value is not None and result["derived_launch"]["characteristics_sha256"] != value["sha256"]:
            raise ValueError("the launch was not derived from the job's sealed machine characteristics")
    if platform == "darwin":
        result["host"]["effective_settings"]["native_resource_enforcement"] = MAC_RESOURCE_ENFORCEMENT
    save(destination if destination is not None else output / "comparison-receipt.json", result)
    return result

#!/home/jy001/micromamba/envs/virtual-cell/bin/python
"""Validate and operate the Git-backed VCC research graph.

This checks registration, provenance and execution contracts. Scientific claims
still require review; neither source inspection nor a result file proves them.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

AXES = set("TDRALOVIG")
FILES = {name: f"docs/research/{name}.json" for name in
         ("method_space", "experiment_dag", "evidence_ledger")}
STATES = {"signal", "supported", "not_supported", "inconclusive"}
SOURCE_STATES = {"literature": "external", "dataset": "observed", "protocol": "constraint"}
LOCAL_STATUS_EVIDENCE = {"UNTESTED": set(), "SIGNAL": {"signal", "supported"},
                         "CONFIRMED": {"supported"}, "REJECTED": {"not_supported"},
                         "CONTEXT_DEPENDENT": STATES}
NODE_STATES = {"draft", "ready", "completed", "failed", "interrupted"}
PURPOSES = {"screen", "confirm", "interaction", "integrate", "protocol_audit", "baseline"}
SPEC_FIELDS = ("experiment_id", "directory", "method_id", "protocol_id", "comparison_id",
               "controls", "sources", "requires_evidence", "evidence_conditions", "config_ref", "expected_config", "code_refs")


class ResearchError(ValueError):
    """A research contract is missing or inconsistent."""


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def git(root, *args, check=True):
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True)
    if check and result.returncode:
        raise ResearchError(result.stderr.decode().strip() or "git command failed")
    return result


def local_path(root, path):
    if not isinstance(path, str) or not path:
        raise ResearchError("reference path must be a nonempty repository-relative path")
    resolved = (root / path).resolve()
    if Path(path).is_absolute() or not resolved.is_relative_to(root.resolve()):
        raise ResearchError(f"reference escapes repository: {path}")
    return resolved


def read_bytes(root, path, staged=False):
    resolved = local_path(root, path)
    if staged:
        result = git(root, "show", f":{path}", check=False)
        if result.returncode == 0:
            return result.stdout
        # Results are intentionally ignored by Git and remain local evidence.
        if git(root, "ls-files", "--error-unmatch", "--", path, check=False).returncode == 0:
            raise ResearchError(f"tracked reference is absent from index: {path}")
    try:
        return resolved.read_bytes()
    except OSError as exc:
        raise ResearchError(f"cannot read {path}: {exc}") from exc


def load(root, staged=False):
    objects = {}
    for name, path in FILES.items():
        try:
            if staged:
                raw = git(root, "show", f":{path}").stdout
            else:
                raw = read_bytes(root, path)
            objects[name] = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ResearchError(f"invalid JSON in {path}: {exc}") from exc
    return objects


def ref_error(root, ref, staged=False):
    if not isinstance(ref, dict) or not {"path", "sha256"} <= set(ref) or set(ref) - {"path", "sha256", "git_commit"}:
        return "reference must contain path and sha256, with optional git_commit"
    if not isinstance(ref["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", ref["sha256"]):
        return "reference sha256 must be 64 lowercase hexadecimal characters"
    try:
        if "git_commit" in ref:
            local_path(root, ref["path"])
            if not re.fullmatch(r"[0-9a-f]{40,64}", str(ref["git_commit"])):
                return "reference git_commit must be a full commit hash"
            raw = git(root, "show", f"{ref['git_commit']}:{ref['path']}").stdout
        else:
            raw = read_bytes(root, ref["path"], staged)
        actual = hashlib.sha256(raw).hexdigest()
    except ResearchError as exc:
        return str(exc)
    if actual != ref["sha256"]:
        return f"reference hash changed: {ref['path']}"
    return None


def differences(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        result = set()
        for key in left.keys() | right.keys():
            field = f"{prefix}.{key}" if prefix else key
            if not prefix and key == "experiment_id":
                continue
            if key not in left or key not in right:
                result.add(field)
            else:
                result.update(differences(left[key], right[key], field))
        return result
    return {prefix} if left != right else set()


def is_closed_experiment(item):
    return item.get("kind", "experiment") == "experiment" and item.get("state") in STATES


def evidence_parents(objects, evidence_id):
    """Scientific prerequisites depend on all arms of their planned comparison.

    Published findings and dataset facts are inputs to a hypothesis, not results
    of a local comparison. They never introduce execution dependencies.
    """
    item = objects["evidence_ledger"]["evidence"].get(evidence_id, {})
    if item.get("kind", "experiment") != "experiment":
        return []
    comparison = objects["experiment_dag"]["comparisons"].get(item.get("comparison_id"), {})
    return list(dict.fromkeys(comparison.get("controls", []) + comparison.get("candidates", [])))


def incomplete(node, dag):
    """Return missing launch requirements without upgrading a draft to ready."""
    missing = list(node.get("blockers", []))
    for field in ("experiment_id", "directory", "method_id", "protocol_id", "comparison_id",
                  "config_ref", "expected_config", "code_refs"):
        if not node.get(field):
            missing.append(f"missing {field}")
    protocol = dag.get("protocols", {}).get(node.get("protocol_id"), {})
    if protocol.get("status") != "ready":
        missing.append("evaluation protocol is not ready/frozen")
    comparison = dag.get("comparisons", {}).get(node.get("comparison_id"), {})
    if comparison.get("status") != "ready":
        missing.append("comparison is not ready")
    design = node.get("design", {})
    if "fit_scope" in design:
        scope = design["fit_scope"]
        if not isinstance(scope, dict) or any(not scope.get(field) for field in ("outer_split", "target_partition")):
            missing.append("fit_scope requires a concrete outer_split and target_partition for this independent fit")
        if (node.get("expected_config") or {}).get("fit_scope") != scope:
            missing.append("expected_config.fit_scope must equal design.fit_scope")
    return missing


def validate(root, objects, staged=False):
    """Return all discoverable contract violations; draft omissions are allowed."""
    errors = []
    for name, obj in objects.items():
        if not isinstance(obj, dict) or obj.get("schema_version") != 1:
            errors.append(f"{name}: schema_version must be 1")
    if errors:
        return errors
    space, dag, ledger = (objects[name] for name in FILES)
    axes, methods = space.get("axes", {}), space.get("methods", {})
    nodes, protocols, comparisons = (dag.get(key, {}) for key in ("nodes", "protocols", "comparisons"))
    evidence = ledger.get("evidence", {})
    if any(not isinstance(value, dict) for value in (axes, methods, nodes, protocols, comparisons, evidence)):
        return ["axes/methods/nodes/protocols/comparisons/evidence must be objects"]
    if set(axes) != AXES:
        errors.append("method_space.axes must contain exactly T/D/R/A/L/O/V/I/G")
    for axis, spec in axes.items():
        if not isinstance(spec, dict) or not spec.get("name") or not isinstance(spec.get("variants"), dict) or not spec["variants"]:
            errors.append(f"axis {axis}: name and nonempty variants required")
    for mid, method in methods.items():
        if not method.get("description") or not method.get("prior"):
            errors.append(f"method {mid}: description and explicit prior required; use 'none' when absent")
        coords = method.get("axes", {})
        if set(coords) != AXES:
            errors.append(f"method {mid}: all nine axes required")
        for axis, variant in coords.items():
            if variant not in axes.get(axis, {}).get("variants", {}):
                errors.append(f"method {mid}: unknown {axis} variant {variant}")
        local_evidence = []
        for eid in method.get("evidence_ids", []):
            if eid not in evidence:
                errors.append(f"method {mid}: unknown evidence {eid}")
                continue
            item = evidence[eid]
            if item.get("kind", "experiment") == "experiment":
                referenced_nodes = evidence_parents(objects, eid) if item.get("state") == "pending" else item.get("node_ids", [])
                if not any(nodes.get(nid, {}).get("method_id") == mid for nid in referenced_nodes):
                    errors.append(f"method {mid}: experiment evidence {eid} must include a node using this method")
                elif is_closed_experiment(item):
                    local_evidence.append(item)
        local_status = method.get("local_status")
        if local_status is not None and local_status not in LOCAL_STATUS_EVIDENCE:
            errors.append(f"method {mid}: invalid local_status {local_status}")
        required_states = LOCAL_STATUS_EVIDENCE.get(local_status, set())
        # Keep existing lowercase result statuses compatible with the formal
        # local_status field, without allowing literature to upgrade either.
        old_status = method.get("status")
        if old_status in STATES:
            old_allowed = {"signal", "supported"} if old_status == "signal" else {old_status}
            if not any(item["state"] in old_allowed for item in local_evidence):
                errors.append(f"method {mid}: local conclusion status requires matching closed experiment evidence")
        if required_states and not any(item["state"] in required_states for item in local_evidence):
            errors.append(f"method {mid}: local_status {local_status} requires matching closed experiment evidence")

    def refs(label, records):
        if not isinstance(records, list):
            errors.append(f"{label}: references must be a list")
            return
        for ref in records:
            problem = ref_error(root, ref, staged)
            if problem:
                errors.append(f"{label}: {problem}")

    for pid, protocol in protocols.items():
        refs(f"protocol {pid}", protocol.get("source_refs", []))
        if protocol.get("status") == "ready" and (not protocol.get("scope") or not protocol.get("description")):
            errors.append(f"protocol {pid}: ready requires description and scope")
    new_directories = {}
    for nid, node in nodes.items():
        if node.get("status") not in NODE_STATES or not isinstance(node.get("legacy"), bool):
            errors.append(f"node {nid}: valid status and explicit legacy boolean required")
        for field, table in (("method_id", methods), ("protocol_id", protocols), ("comparison_id", comparisons)):
            value = node.get(field)
            if value and value not in table:
                errors.append(f"node {nid}: unknown {field} {value}")
        for field in ("controls", "sources"):
            for target in node.get(field, []):
                if target not in nodes:
                    errors.append(f"node {nid}: unknown {field} node {target}")
        for eid in node.get("requires_evidence", []):
            if eid not in evidence:
                errors.append(f"node {nid}: unknown required evidence {eid}")
            elif evidence[eid].get("kind", "experiment") != "experiment":
                errors.append(f"node {nid}: required evidence {eid} must be a local experiment conclusion")
            elif not evidence_parents(objects, eid):
                errors.append(f"node {nid}: required evidence {eid} has no comparison arms")
        conditions = node.get("evidence_conditions", {})
        if not isinstance(conditions, dict):
            errors.append(f"node {nid}: evidence_conditions must be an object")
        else:
            for eid, allowed in conditions.items():
                if eid not in node.get("requires_evidence", []):
                    errors.append(f"node {nid}: condition {eid} must also be in requires_evidence")
                if not isinstance(allowed, list) or not allowed or not set(allowed) <= STATES:
                    errors.append(f"node {nid}: condition {eid} requires a nonempty list of closed experiment states")
        for field in ("config_ref", "metrics_ref", "report_ref"):
            if node.get(field):
                refs(f"node {nid} {field}", [node[field]])
        refs(f"node {nid} code_refs", node.get("code_refs", []))
        if not node.get("legacy"):
            comparison = comparisons.get(node.get("comparison_id"), {})
            if comparison and nid not in comparison.get("controls", []) + comparison.get("candidates", []):
                errors.append(f"node {nid}: not a member of its registered comparison")
            if nid != node.get("experiment_id"):
                errors.append(f"node {nid}: node_id must equal experiment_id")
            directory = node.get("directory")
            if directory in new_directories:
                errors.append(f"node {nid}: Experiment directory already used by {new_directories[directory]}")
            new_directories[directory] = nid
            if any("git_commit" in ref for ref in [node.get("config_ref") or {}] + node.get("code_refs", [])):
                errors.append(f"node {nid}: active config/code references must bind current contents, not historical git_commit")
            if node.get("directory") != f"experiments/{node.get('experiment_id')}":
                errors.append(f"node {nid}: one Experiment requires directory experiments/<experiment_id>")
            expected = node.get("expected_config")
            if expected and expected.get("experiment_id") != node.get("experiment_id"):
                errors.append(f"node {nid}: expected_config.experiment_id mismatch")
            if node.get("status") == "ready":
                errors.extend(f"node {nid}: {item}" for item in incomplete(node, dag))
    seen, active = set(), set()

    def visit(nid):
        if nid in active:
            errors.append(f"Experiment DAG cycle at {nid}")
            return
        if nid in seen or nid not in nodes:
            return
        active.add(nid)
        parents = nodes[nid].get("controls", []) + nodes[nid].get("sources", [])
        for eid in nodes[nid].get("requires_evidence", []):
            parents += evidence_parents(objects, eid)
        for parent in parents:
            visit(parent)
        active.remove(nid)
        seen.add(nid)

    for nid in nodes:
        visit(nid)
    for cid, comparison in comparisons.items():
        members = comparison.get("controls", []) + comparison.get("candidates", [])
        if comparison.get("purpose") not in PURPOSES or comparison.get("status") not in {"draft", "ready", "historical"}:
            errors.append(f"comparison {cid}: invalid purpose/status")
        for nid in members:
            if nid not in nodes:
                errors.append(f"comparison {cid}: unknown node {nid}")
        valid = [nodes[nid] for nid in members if nid in nodes]
        if comparison.get("status") == "historical" and any(not n.get("legacy") for n in valid):
            errors.append(f"comparison {cid}: historical comparisons require legacy nodes")
        pids = {n.get("protocol_id") for n in valid if n.get("protocol_id")}
        if comparison.get("purpose") != "protocol_audit" and len(pids) > 1:
            errors.append(f"comparison {cid}: incompatible evaluation protocols {sorted(pids)}")
        if comparison.get("status") != "ready":
            continue
        rule = comparison.get("decision_rule", {})
        required = ("primary_metric", "direction", "min_effect", "scope", "on_support", "on_against", "on_inconclusive")
        if not comparison.get("hypothesis") or any(key not in rule or rule[key] is None or rule[key] == "" for key in required):
            errors.append(f"comparison {cid}: hypothesis and complete decision_rule required")
        changed_axes = set(comparison.get("changed_axes", []))
        if not changed_axes <= AXES:
            errors.append(f"comparison {cid}: invalid changed_axes")
        if comparison.get("purpose") in {"screen", "confirm"} and len(changed_axes) != 1:
            errors.append(f"comparison {cid}: screen/confirm requires one changed axis")
        controls = [nodes[n] for n in comparison.get("controls", []) if n in nodes]
        candidates = [nodes[n] for n in comparison.get("candidates", []) if n in nodes]
        if not candidates or (comparison.get("purpose") not in {"baseline", "protocol_audit"} and not controls):
            errors.append(f"comparison {cid}: candidates and appropriate controls required")
        all_fields = set()
        for candidate in candidates:
            cfg = candidate.get("expected_config")
            if not isinstance(cfg, dict):
                errors.append(f"comparison {cid}: candidate expected_config missing")
                continue
            if not controls:
                continue
            paired = [n for n in controls if isinstance(n.get("expected_config"), dict)
                      and n["expected_config"].get("seed") == cfg.get("seed")
                      and n["expected_config"].get("fit_scope") == cfg.get("fit_scope")]
            if "seed" not in cfg or not paired:
                errors.append(f"comparison {cid}: seed and fit_scope must match a control")
                continue
            control = paired[0]
            all_fields.update(differences(control["expected_config"], cfg))
            ca = methods.get(candidate.get("method_id"), {}).get("axes", {})
            ba = methods.get(control.get("method_id"), {}).get("axes", {})
            actual_axes = {axis for axis in AXES if ca.get(axis) != ba.get(axis)}
            if not actual_axes <= changed_axes:
                errors.append(f"comparison {cid}: undeclared method axes {sorted(actual_axes - changed_axes)}")
        if controls and comparison.get("purpose") != "protocol_audit" and all_fields != set(comparison.get("changed_fields", [])):
            errors.append(f"comparison {cid}: actual config changed_fields={sorted(all_fields)} differ from declaration")
    for eid, item in evidence.items():
        kind = item.get("kind", "experiment")
        state = item.get("state")
        if not item.get("claim") or not item.get("next_action") or not item.get("reopen_when"):
            errors.append(f"evidence {eid}: claim, next_action and reopen_when required")
        if kind == "experiment":
            if state not in STATES | {"pending"}:
                errors.append(f"evidence {eid}: invalid experiment state")
            if state == "pending" and not item.get("comparison_id"):
                errors.append(f"evidence {eid}: pending experiment requires comparison_id")
        elif kind in SOURCE_STATES:
            if state != SOURCE_STATES[kind]:
                errors.append(f"evidence {eid}: {kind} state must be {SOURCE_STATES[kind]}")
            if item.get("node_ids") or item.get("comparison_id"):
                errors.append(f"evidence {eid}: external sources cannot claim local comparison or node results")
            urls = item.get("source_urls")
            if not isinstance(urls, list) or not urls or any(
                    not isinstance(url, str) or not re.match(r"https?://[^/\s]+", url) for url in urls):
                errors.append(f"evidence {eid}: external source_urls required")
            if not item.get("source_refs"):
                errors.append(f"evidence {eid}: external evidence requires a hashed local source note")
        else:
            errors.append(f"evidence {eid}: unknown kind {kind}")
        if kind in SOURCE_STATES or state == "pending":
            measured = {"results", "metrics", "local_metrics", "local_effect", "local_effects", "effect",
                        "effect_size", "effect_sizes", "paired_effect", "Overall", "score"}
            if measured & item.keys():
                errors.append(f"evidence {eid}: source facts and pending hypotheses cannot contain local results or effects")
        if item.get("comparison_id") and item["comparison_id"] not in comparisons:
            errors.append(f"evidence {eid}: unknown comparison")
        for nid in item.get("node_ids", []):
            if nid not in nodes:
                errors.append(f"evidence {eid}: unknown node {nid}")
        refs(f"evidence {eid}", item.get("source_refs", []))
    queue = ledger.get("queue", [])
    if not isinstance(queue, list) or len(queue) > 3:
        errors.append("decision queue must contain at most three questions")
    else:
        for item in queue:
            if item.get("comparison_id") not in comparisons:
                errors.append(f"queue {item.get('id')}: unknown comparison")
            if not isinstance(item.get("priority"), int) or not item.get("question") or not item.get("action"):
                errors.append(f"queue {item.get('id')}: priority, question and action required")
            for eid in item.get("requires_evidence", []):
                if eid not in evidence:
                    errors.append(f"queue {item.get('id')}: unknown evidence {eid}")
                elif evidence[eid].get("kind", "experiment") != "experiment":
                    errors.append(f"queue {item.get('id')}: required evidence {eid} must be a local experiment conclusion")
    return errors


def require_valid(root, objects, staged=False):
    problems = validate(root, objects, staged)
    if problems:
        raise ResearchError("\n".join(problems))


def entry_errors(root, node, staged=False):
    directory = node["directory"]
    try:
        launch = read_bytes(root, f"{directory}/reproduce.sh", staged).decode()
        if not re.search(r"research(?:\.py)?[^\n]*\bexecute\b", launch):
            return [f"{directory}/reproduce.sh: research execute call missing"]
        paths = git(root, "ls-files", "--", f"{directory}/src", check=False).stdout.decode().splitlines()
        found = any(re.search(r"\bbind\s*\(", read_bytes(root, p, staged).decode())
                    for p in paths if p.endswith(".py"))
        if not found:
            return [f"{directory}/src: bind(actual_config) call missing"]
    except (ResearchError, UnicodeDecodeError) as exc:
        return [str(exc)]
    return []


def staged_errors(root, objects):
    errors = []
    changed = git(root, "diff", "--cached", "--name-only", "--diff-filter=ACMRD").stdout.decode().splitlines()
    nodes = objects["experiment_dag"]["nodes"]
    for nid, node in nodes.items():
        directory = node.get("directory", "")
        relevant = [p for p in changed if p.startswith(directory + "/")]
        if not relevant:
            continue
        training_changes = [p for p in relevant if p == directory + "/reproduce.sh" or
                            p.startswith(directory + "/src/") or p.startswith(directory + "/configs/") or
                            p in {directory + "/pyproject.toml", directory + "/uv.lock"}]
        if node.get("legacy") and training_changes:
            errors.append(f"legacy node {nid}: training files are frozen; create a new Experiment or restore its original Git commit")
        elif not node.get("legacy") and training_changes:
            errors.extend(entry_errors(root, node, True))
    registered = {n.get("directory") for n in nodes.values()}
    for path in changed:
        training = re.fullmatch(r"(experiments/[^/]+)/(?:src/.+|configs/.+|reproduce\.sh|pyproject\.toml|uv\.lock)", path)
        if training and training.group(1) not in registered:
            errors.append(f"{path}: Experiment is not registered in the DAG")
    ledger = objects["evidence_ledger"]
    for cid, comparison in objects["experiment_dag"]["comparisons"].items():
        members = comparison.get("controls", []) + comparison.get("candidates", [])
        if comparison.get("status") != "ready" or not members or not all(nodes[n]["status"] == "completed" for n in members):
            continue
        if not any(is_closed_experiment(item) and item.get("comparison_id") == cid and set(members) <= set(item.get("node_ids", [])) and
                   all(nodes[n].get("metrics_ref") in item.get("source_refs", []) for n in members)
                   for item in ledger["evidence"].values()):
            errors.append(f"comparison {cid}: all arms completed; close evidence with all metrics refs and next decision before committing")
    return errors


def committed(root, paths, revision="HEAD"):
    for path in sorted(set(paths)):
        actual = read_bytes(root, path)
        recorded = git(root, "show", f"{revision}:{path}", check=False)
        if recorded.returncode or recorded.stdout != actual:
            raise ResearchError(f"uncommitted or changed registered file: {path}")


def metadata(root, objects, nid, commit=None):
    node = objects["experiment_dag"]["nodes"][nid]
    registered_method = objects["method_space"]["methods"][node["method_id"]]
    method = {field: registered_method.get(field) for field in ("axes", "prior")}
    dag = objects["experiment_dag"]
    protocol = {key: value for key, value in dag["protocols"][node["protocol_id"]].items()
                if key not in {"description", "status"}}
    comparison = {key: value for key, value in dag["comparisons"][node["comparison_id"]].items()
                  if key != "status"}
    spec = {field: node.get(field) for field in SPEC_FIELDS}
    return {"node_id": nid, "experiment_id": node["experiment_id"], "method_id": node["method_id"],
            "comparison_id": node["comparison_id"], "protocol_id": node["protocol_id"],
            "protocol_sha256": digest(protocol), "effective_config_sha256": digest(node["expected_config"]),
            "registration_sha256": digest({"node": spec, "method": method, "protocol": protocol, "comparison": comparison}),
            "git_commit": commit or git(root, "rev-parse", "HEAD").stdout.decode().strip()}


def gate(root, node_id, *, resume=False, config_path=None):
    """Check a standard launch, without starting or modifying an Experiment."""
    root = Path(root).resolve()
    objects = load(root)
    require_valid(root, objects)
    node = objects["experiment_dag"]["nodes"].get(node_id)
    if node is None:
        raise ResearchError(f"unknown node {node_id}")
    if node.get("legacy"):
        raise ResearchError(f"{node_id}: legacy results are read-only; create a new Experiment")
    allowed = {"failed", "interrupted"} if resume else {"ready"}
    if node.get("status") not in allowed:
        raise ResearchError(f"{node_id}: status {node.get('status')} cannot {'resume' if resume else 'start'}")
    if not resume and any((local_path(root, node["directory"]) / name).exists()
                          for name in ("metrics.json", "cache/research-binding.json")):
        raise ResearchError(f"{node_id}: existing execution results require explicit recovery or a new Experiment")
    missing = incomplete(node, objects["experiment_dag"])
    if missing:
        raise ResearchError(f"{node_id}: " + "; ".join(missing))
    ledger = objects["evidence_ledger"]["evidence"]
    for eid in node.get("requires_evidence", []):
        item = ledger.get(eid, {})
        if not is_closed_experiment(item) or not item.get("next_action") or not item.get("reopen_when"):
            raise ResearchError(f"{node_id}: prerequisite evidence {eid} is not closed")
        if eid in node.get("evidence_conditions", {}) and item.get("state") not in node["evidence_conditions"][eid]:
            raise ResearchError(f"{node_id}: prerequisite evidence {eid} state {item.get('state')} does not meet evidence_conditions")
    if config_path is not None and local_path(root, config_path) != local_path(root, node["config_ref"]["path"]):
        raise ResearchError(f"{node_id}: actual config path differs from config_ref")
    paths = list(FILES.values()) + [node["config_ref"]["path"]] + [r["path"] for r in node["code_refs"]]
    paths += [f"{node['directory']}/reproduce.sh", f"{node['directory']}/PLAN.md"]
    required_code = {f"{node['directory']}/{name}" for name in ("reproduce.sh", "pyproject.toml", "uv.lock")}
    required_code.update(git(root, "ls-files", "--", f"{node['directory']}/src").stdout.decode().splitlines())
    uncovered = required_code - {ref["path"] for ref in node["code_refs"]}
    if uncovered:
        raise ResearchError(f"{node_id}: code_refs do not cover training files: {sorted(uncovered)}")
    committed(root, paths)
    problems = entry_errors(root, node)
    if problems:
        raise ResearchError("\n".join(problems))
    if node.get("git_commit") and node["git_commit"] != git(root, "rev-parse", "HEAD").stdout.decode().strip():
        # Unrelated commits are allowed: only registered files must match.
        committed(root, paths[3:], node["git_commit"])
    if resume:
        directory = local_path(root, node["directory"])
        if not any((directory / name).is_file() for name in ("config.yaml", "cache/research-binding.json")) or not node.get("metrics_ref"):
            raise ResearchError(f"{node_id}: resume requires frozen binding/config, metrics and full-state checkpoint restoration")
        previous = json.loads(read_bytes(root, node["metrics_ref"]["path"])).get("research", {})
        if not previous.get("git_commit") or previous != metadata(root, objects, node_id, previous["git_commit"]):
            raise ResearchError(f"{node_id}: resume registration differs from original bound research identity")
        return previous
    return metadata(root, objects, node_id)


def bind(root, node_id, actual_config, *, resume=False):
    """Call after resolving configuration and before W&B init; embed return value
    in config.yaml and wandb.config['research']. Resume state restoration remains
    the training implementation's responsibility, not this metadata binding.
    """
    root = Path(root).resolve()
    result = gate(root, node_id, resume=resume)
    expected = load(root)["experiment_dag"]["nodes"][node_id]["expected_config"]
    if canonical(actual_config) != canonical(expected):
        fields = sorted(differences(expected, actual_config))
        raise ResearchError(f"{node_id}: actual_config differs from registered expected_config: {fields or ['experiment_id']}")
    marker = root / "experiments" / node_id / "cache/research-binding.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    # Durable before any training output: an interrupted execution may no longer
    # use the pre-binding retry path, even when metrics have not been written.
    if not resume:
        with marker.open("x") as stream:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
    return result


def status(objects):
    dag, ledger = objects["experiment_dag"], objects["evidence_ledger"]
    evidence = ledger["evidence"]
    closed = {item.get("comparison_id") for item in evidence.values() if is_closed_experiment(item)}
    unclosed = [cid for cid, comparison in dag["comparisons"].items()
                if comparison.get("status") == "ready" and cid not in closed and
                all(dag["nodes"][n].get("status") == "completed" for n in
                    comparison.get("controls", []) + comparison.get("candidates", []))]
    queue = []
    for item in sorted(ledger.get("queue", []), key=lambda item: item["priority"]):
        queue.append(dict(item, unmet_evidence=[eid for eid in item.get("requires_evidence", [])
                                               if not is_closed_experiment(evidence.get(eid, {}))]))
    pending = [{"id": eid, "comparison_id": item.get("comparison_id"),
                "hypothesis": item.get("hypothesis", item.get("claim")), "next_action": item.get("next_action")}
               for eid, item in evidence.items()
               if item.get("kind", "experiment") == "experiment" and item.get("state") == "pending"]
    return {"initialization": {"axes": len(objects["method_space"]["axes"]),
                               "methods": len(objects["method_space"]["methods"]),
                               "planned_nodes": sum(n.get("status") == "draft" for n in dag["nodes"].values()),
                               "source_evidence": sum(item.get("kind") in SOURCE_STATES for item in evidence.values()),
                               "closed_experiment_evidence": sum(is_closed_experiment(item) for item in evidence.values()),
                               "pending_hypotheses": len(pending)},
            "needs_evidence": unclosed, "pending_questions": pending, "queue": queue,
            "nodes": {nid: {"status": node["status"], "legacy": node["legacy"],
                            "blockers": [] if node["legacy"] else incomplete(node, dag),
                            "next_command": f"./scripts/research.py gate {nid}" if node["status"] == "ready" and not node["legacy"] else None}
                      for nid, node in dag["nodes"].items()}}


@contextmanager
def registry_lock(root):
    key = hashlib.sha256(str(root.resolve()).encode()).hexdigest()
    with (Path(tempfile.gettempdir()) / f"vcc-research-registry-{key}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def write_object(root, name, obj):
    path = root / FILES[name]
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".research-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def record(root, node_id, metrics_path):
    with registry_lock(root):
        return _record(root, node_id, metrics_path)


def _record(root, node_id, metrics_path):
    objects = load(root)
    node = objects["experiment_dag"]["nodes"].get(node_id)
    if not node or node.get("legacy") or node.get("status") not in {"ready", "failed", "interrupted"}:
        raise ResearchError(f"{node_id}: record requires a nonlegacy, unfinished Experiment")
    path = local_path(root, metrics_path)
    if not path.is_relative_to(local_path(root, node["directory"])):
        raise ResearchError("metrics must belong to the Experiment directory")
    result = json.loads(path.read_text())
    node["metrics_ref"] = {"path": str(path.relative_to(root)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    require_valid(root, objects)
    if result.get("status") not in {"completed", "failed", "interrupted"}:
        raise ResearchError("metrics.status must be completed, failed or interrupted")
    research = result.get("research", {})
    commit = research.get("git_commit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40,64}", commit):
        raise ResearchError("metrics.research.git_commit missing or invalid")
    git(root, "cat-file", "-e", commit + "^{commit}")
    if research != metadata(root, objects, node_id, commit):
        raise ResearchError("metrics.research differs from bound registration")
    recorded_objects = {name: json.loads(git(root, "show", f"{commit}:{path}").stdout) for name, path in FILES.items()}
    if research != metadata(root, recorded_objects, node_id, commit):
        raise ResearchError("metrics.research registration was not present at recorded Git commit")
    if result["status"] == "completed":
        if result.get("evaluation_completed") is not True:
            raise ResearchError("completed requires evaluation_completed=true")
        primary = objects["experiment_dag"]["comparisons"][node["comparison_id"]]["decision_rule"]["primary_metric"]
        value = result.get(primary)
        if value is None:
            value = result
            for part in primary.split("."):
                value = value.get(part) if isinstance(value, dict) else None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ResearchError(f"completed requires finite primary metric: {primary}")
        for field in ("checkpoint_ref", "predictions_ref"):
            problem = ref_error(root, result.get(field))
            if problem:
                raise ResearchError(f"{field}: {problem}")
    wandb_url = result.get("wandb_url")
    if wandb_url and wandb_url != f"https://wandb.ai/yjcyxky/virtual-cell-challenge/runs/{node['experiment_id']}":
        raise ResearchError("metrics.wandb_url does not match the Experiment identity")
    node["status"] = result["status"]
    node["git_commit"] = commit
    for field in ("wandb_url", "artifact", "wandb_sync"):
        if field in result:
            node[field] = result[field]
    write_object(root, "experiment_dag", objects["experiment_dag"])
    return {"node_id": node_id, "status": result["status"], "next": "close comparison evidence after all arms finish, then commit results and decision together"}


def execute(root, node_id, command, *, resume=False):
    """Hold an Experiment lock through execution and automatically register its result."""
    if not command:
        raise ResearchError("execute requires -- COMMAND [ARG ...]")
    gate(root, node_id, resume=resume)
    identity = hashlib.sha256(f"{root}:{node_id}".encode()).hexdigest()
    lock_path = Path(tempfile.gettempdir()) / f"vcc-research-{identity}.lock"
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ResearchError(f"{node_id}: another execution holds the Experiment lock") from exc
        launch_identity = gate(root, node_id, resume=resume)
        node = load(root)["experiment_dag"]["nodes"][node_id]
        directory = local_path(root, node["directory"])
        metrics_path = str((directory / "metrics.json").relative_to(root))
        environment = dict(os.environ, VCC_RESEARCH_NODE=node_id, VCC_RESEARCH_ROOT=str(root),
                           VCC_RESEARCH_RESUME="1" if resume else "0")
        failure, failure_status = None, "failed"
        try:
            result = subprocess.run(command, cwd=directory, env=environment)
            if result.returncode:
                failure = f"execution exited {result.returncode}; completed status rejected"
            elif not (directory / "metrics.json").is_file():
                failure = "execution produced no Experiment metrics.json"
            else:
                return record(root, node_id, metrics_path)
        except (OSError, ResearchError, json.JSONDecodeError) as exc:
            failure = f"execution/result contract failed: {exc}"
        except KeyboardInterrupt:
            failure, failure_status = "execution interrupted; checkpoint restoration remains required", "interrupted"
        with registry_lock(root):
            objects = load(root)
            registered = objects["experiment_dag"]["nodes"][node_id]
            registered["status"] = failure_status
            registered["blockers"] = list(dict.fromkeys(registered.get("blockers", []) + [failure]))
            registered["last_attempt"] = {"status": failure_status, "research": launch_identity, "error": failure}
            if not (directory / "metrics.json").exists() and (directory / "cache/research-binding.json").exists():
                (directory / "metrics.json").write_text(json.dumps({
                    "status": failure_status, "research": launch_identity,
                    "evaluation_completed": False, "execution_error": failure,
                }, indent=2) + "\n")
            if (directory / "metrics.json").is_file():
                registered["metrics_ref"] = {"path": metrics_path, "sha256": hashlib.sha256((directory / "metrics.json").read_bytes()).hexdigest()}
            write_object(root, "experiment_dag", objects["experiment_dag"])
        raise ResearchError(f"{node_id}: {failure}; {failure_status} status recorded, inspect and resolve blocker before resume")


def retry(root, node_id):
    """Re-arm an unchanged execution that failed before training was bound."""
    with registry_lock(root):
        objects = load(root)
        require_valid(root, objects)
        node = objects["experiment_dag"]["nodes"].get(node_id, {})
        if node.get("legacy") or node.get("status") not in {"failed", "interrupted"}:
            raise ResearchError("retry requires a failed/interrupted nonlegacy Experiment")
        directory = local_path(root, node["directory"])
        if any((directory / name).exists() for name in
               ("config.yaml", "metrics.json", "cache/research-binding.json")) or any(
                   p.is_file() for name in ("checkpoints", "predictions") for p in (directory / name).rglob("*")):
            raise ResearchError("training already bound or produced results; restore its complete state with --resume")
        attempt = node.get("last_attempt", {})
        previous = attempt.get("research", {})
        if not previous.get("git_commit") or previous != metadata(root, objects, node_id, previous["git_commit"]):
            raise ResearchError("retry requires unchanged conditions from the recorded failed attempt")
        node["blockers"] = [problem for problem in node.get("blockers", []) if problem != attempt.get("error")]
        node["status"] = "ready"
        require_valid(root, objects)
        write_object(root, "experiment_dag", objects["experiment_dag"])
        return {"node_id": node_id, "status": "ready", "next": "commit the retry registration, then execute the same pipeline; the failed attempt remains recorded"}


def close(root, evidence_path):
    with registry_lock(root):
        return _close(root, evidence_path)


def _close(root, evidence_path):
    objects = load(root)
    require_valid(root, objects)
    item = json.loads(Path(evidence_path).read_text())
    eid = item.pop("id", item.pop("evidence_id", None))
    previous = objects["evidence_ledger"]["evidence"].get(eid)
    if not eid or (previous and (previous.get("kind", "experiment") != "experiment" or previous.get("state") != "pending")):
        raise ResearchError("new unique evidence id or pending hypothesis required; retain closed evidence")
    if not is_closed_experiment(item):
        raise ResearchError("close requires a local experiment conclusion, not a source fact or pending hypothesis")
    if previous:
        if previous.get("comparison_id") != item.get("comparison_id"):
            raise ResearchError("closing pending hypothesis must preserve its comparison_id")
        original_hypothesis = previous.get("hypothesis", previous["claim"])
        if item.get("hypothesis", original_hypothesis) != original_hypothesis:
            raise ResearchError("closing pending hypothesis must preserve its original hypothesis")
        item["hypothesis"] = original_hypothesis
    if not item.get("node_ids") or not item.get("source_refs"):
        raise ResearchError("closing evidence requires node_ids and hashed source_refs")
    if not item.get("limitations"):
        raise ResearchError("closing evidence requires explicit limitations")
    cid = item.get("comparison_id")
    if cid not in objects["experiment_dag"]["comparisons"]:
        raise ResearchError("closing new evidence requires a registered comparison_id")
    comparison = objects["experiment_dag"]["comparisons"][cid]
    required_nodes = set(comparison.get("controls", []) + comparison.get("candidates", []))
    if not required_nodes <= set(item["node_ids"]):
        raise ResearchError(f"closing {cid} requires all comparison arms: {sorted(required_nodes)}")
    for nid in item["node_ids"]:
        node = objects["experiment_dag"]["nodes"].get(nid, {})
        if node.get("status") != "completed":
            raise ResearchError(f"{nid}: failed or unfinished execution is not method evidence")
        if not node.get("metrics_ref") or node["metrics_ref"] not in item["source_refs"]:
            raise ResearchError(f"{nid}: closing evidence must include its exact metrics_ref")
    objects["evidence_ledger"]["evidence"][eid] = item
    require_valid(root, objects)
    write_object(root, "evidence_ledger", objects["evidence_ledger"])
    return {"evidence_id": eid, "state": item["state"], "next_action": item["next_action"],
            "notice": "Recorded a reviewable scientific judgment, not an automatic proof."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="repository root")
    commands = parser.add_subparsers(dest="command", required=True)
    check_parser = commands.add_parser("check", help="validate registry, hashes and attribution contracts")
    check_parser.add_argument("--staged", action="store_true", help="read registry/index and check changed training entrypoints")
    status_parser = commands.add_parser("status", help="show missing evidence, at most three questions, and launch blockers")
    status_parser.add_argument("--json", action="store_true")
    commands.add_parser("graph", help="print the actual registered Experiment DAG as Mermaid Markdown")
    gate_parser = commands.add_parser("gate", help="check registration before environment setup; does not launch training")
    gate_parser.add_argument("node")
    gate_parser.add_argument("--config", help="repository-relative actual configuration file")
    gate_parser.add_argument("--resume", action="store_true", help="resume only failed/interrupted registered identity")
    execute_parser = commands.add_parser("execute", help="gate, lock, execute COMMAND in Experiment directory, and register metrics.json")
    execute_parser.add_argument("node")
    execute_parser.add_argument("--resume", action="store_true", help="continue failed/interrupted identity; COMMAND must restore checkpoint state")
    execute_parser.add_argument("argv", nargs=argparse.REMAINDER, help="-- COMMAND [ARG ...]; invoke training, not reproduce.sh recursively")
    record_parser = commands.add_parser("record", help="attach completed/failed/interrupted metrics to a node")
    record_parser.add_argument("node")
    record_parser.add_argument("--metrics", required=True, help="repository-relative metrics.json")
    retry_parser = commands.add_parser("retry", help="re-arm unchanged pre-binding failure; does not retrain or resume checkpoints")
    retry_parser.add_argument("node")
    close_parser = commands.add_parser("close", help="append a local conclusion or close its pending hypothesis ID, retaining earlier conclusions")
    close_parser.add_argument("--file", required=True, help="path to evidence JSON")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        if args.command == "gate":
            output = gate(root, args.node, resume=args.resume, config_path=args.config)
        elif args.command == "execute":
            child = args.argv
            if child and child[0] == "--resume":
                args.resume, child = True, child[1:]
            if child and child[0] == "--":
                child = child[1:]
            output = execute(root, args.node, child, resume=args.resume)
        elif args.command == "record":
            output = record(root, args.node, args.metrics)
        elif args.command == "retry":
            output = retry(root, args.node)
        elif args.command == "close":
            output = close(root, args.file)
        else:
            objects = load(root, getattr(args, "staged", False))
            require_valid(root, objects, getattr(args, "staged", False))
            if args.command == "check":
                problems = staged_errors(root, objects) if args.staged else []
                if problems:
                    raise ResearchError("\n".join(problems))
                output = {"valid": True, "nodes": len(objects["experiment_dag"]["nodes"]), "staged": args.staged}
            elif args.command == "status":
                output = status(objects)
                if not args.json:
                    summary = output["initialization"]
                    print(f"Research: {summary['axes']} axes, {summary['methods']} methods, {summary['planned_nodes']} draft nodes; "
                          f"{summary['source_evidence']} source records, {summary['pending_hypotheses']} pending hypotheses, "
                          f"{summary['closed_experiment_evidence']} local conclusions")
                    for cid in output["needs_evidence"]:
                        print(f"FIRST: close completed comparison {cid} with evidence and next decision")
                    for item in output["queue"]:
                        print(f"{item['priority']}. {item['id']}: {item['question']}\n   {item['action']}")
                        if item["unmet_evidence"]:
                            print("   Await local evidence: " + ", ".join(item["unmet_evidence"]))
                    for item in output["pending_questions"]:
                        print(f"PENDING {item['id']} ({item['comparison_id']}): {item['hypothesis']}")
                    for nid, node in output["nodes"].items():
                        if not node["legacy"]:
                            print(f"{nid} [{node['status']}]: " + ("; ".join(node["blockers"]) or node["next_command"] or "await result/evidence"))
                    return 0
            else:
                print("# Experiment DAG\n\nGenerated from the Git registry. Solid edges distinguish controls and reused sources; "
                      "dashed edges require a local comparison conclusion. Draft nodes are plans, not completed experiments.\n\n```mermaid\nflowchart TD")
                nodes = objects["experiment_dag"]["nodes"]
                identifiers = {nid: f"n{i}" for i, nid in enumerate(nodes)}
                for nid, node in nodes.items():
                    label = f"{nid} | {node['status']}".replace('"', "'").replace("\n", " ")
                    print(f'  {identifiers[nid]}["{label}"]')
                for nid, node in nodes.items():
                    for field, label in (("controls", "control"), ("sources", "source")):
                        for target in node.get(field, []):
                            print(f"  {identifiers[target]} -->|{label}| {identifiers[nid]}")
                    for eid in node.get("requires_evidence", []):
                        label = eid.replace('"', "'").replace("\n", " ")
                        for target in evidence_parents(objects, eid):
                            print(f'  {identifiers[target]} -.->|"evidence: {label}"| {identifiers[nid]}')
                print("```")
                return 0
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0
    except (ResearchError, OSError, json.JSONDecodeError, KeyError, TypeError, AttributeError) as exc:
        print(f"research: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

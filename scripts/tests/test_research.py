"""Exercise the research controls through their public CLI and training binding.

Every test owns a temporary Git repository. No real experiment, training process,
dataset, environment, or W&B service is modified.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "research.py"
AXES = "T D R A L O V I G".split()
REGISTRY_DIR = Path("docs/research")


class ResearchControlsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="vcc-research-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.git("init", "--quiet")
        self.git("config", "user.email", "research-test@example.invalid")
        self.git("config", "user.name", "Research control tests")
        self.space = {
            "schema_version": 1,
            "axes": {
                axis: {"name": axis, "variants": {"default": "Control"}}
                for axis in AXES
            },
            "methods": {
                "base": {
                    "description": "Minimal baseline",
                    "axes": dict.fromkeys(AXES, "default"),
                    "prior": "No biological prior",
                    "evidence_ids": [],
                }
            },
        }
        self.dag = {
            "schema_version": 1,
            "protocols": {
                "p": {
                    "description": "Frozen fixture evaluation",
                    "scope": "Development; two held-out backgrounds",
                    "status": "ready",
                    "source_refs": [],
                }
            },
            "nodes": {},
            "comparisons": {},
        }
        self.ledger = {"schema_version": 1, "evidence": {}, "queue": []}
        self.add_node("baseline", "base", "baseline-comparison")
        self.write_registries()
        self.commit()

    def git(self, *args):
        result = subprocess.run(
            ["git", "-C", str(self.root), *args],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout.strip()

    def write(self, path, contents):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(contents, encoding="utf-8")
        return target

    def write_json(self, path, value):
        return self.write(path, json.dumps(value, indent=2) + "\n")

    def file_ref(self, path):
        return {
            "path": str(path),
            "sha256": hashlib.sha256((self.root / path).read_bytes()).hexdigest(),
        }

    def write_registries(self):
        for name, value in (
            ("method_space", self.space),
            ("experiment_dag", self.dag),
            ("evidence_ledger", self.ledger),
        ):
            self.write_json(REGISTRY_DIR / f"{name}.json", value)

    def commit(self):
        self.git("add", "--all")
        self.git("commit", "--quiet", "--allow-empty", "-m", "Fixture registration")

    def cli(self, *args, ok=True):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--root", str(self.root), *args],
            cwd=self.root,
            text=True,
            capture_output=True,
            check=False,
        )
        output = result.stdout + result.stderr
        if ok:
            self.assertEqual(result.returncode, 0, output)
        else:
            self.assertNotEqual(result.returncode, 0, output)
        return output

    def bind(self, node_id, config=None, *, ok=True, resume=False):
        if config is None:
            config = self.dag["nodes"][node_id]["expected_config"]
        code = (
            "import json, pathlib, runpy, sys; "
            "api = runpy.run_path(sys.argv[1]); "
            "result = api['bind'](pathlib.Path(sys.argv[2]), sys.argv[3], "
            "json.loads(sys.argv[4]), resume=json.loads(sys.argv[5])); "
            "print(json.dumps(result))"
        )
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                code,
                str(SCRIPT),
                str(self.root),
                node_id,
                json.dumps(config),
                json.dumps(resume),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        output = result.stdout + result.stderr
        if ok:
            self.assertEqual(result.returncode, 0, output)
            return json.loads(result.stdout.strip().splitlines()[-1])
        self.assertNotEqual(result.returncode, 0, output)
        return output

    def add_node(self, node_id, method, comparison_id, *, controls=()):
        directory = Path("experiments") / node_id
        config = {
            "experiment_id": node_id,
            "seed": 17,
            "data": {"qc": False},
            "model": {"width": 4},
        }
        self.write_json(directory / "configs/train.json", config)
        self.write(directory / "PLAN.md", "# Plan\nTrain one full epoch.\n")
        self.write(
            directory / "reproduce.sh",
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            f"python ../../scripts/research.py execute {node_id} -- python src/main.py\n",
        )
        self.write(
            directory / "src/main.py",
            "import hashlib, json, os, runpy, sys\n"
            "from pathlib import Path\n"
            "bind = runpy.run_path(sys.argv[1])['bind']\n"
            "root = Path(sys.argv[2])\n"
            f"node_id = {node_id!r}\n"
            "directory = root / 'experiments' / node_id\n"
            "config = json.loads((directory / 'configs/train.json').read_text())\n"
            "research = bind(root, node_id, config, resume=os.environ.get('VCC_RESEARCH_RESUME') == '1')\n"
            "mode = sys.argv[3]\n"
            "if mode == 'no-result':\n"
            "    raise SystemExit(0)\n"
            "(directory / 'config.yaml').write_text(json.dumps(dict(config, research=research)))\n"
            "(directory / 'checkpoint.bin').write_bytes(b'fixture checkpoint')\n"
            "(directory / 'predictions.json').write_text('[0.5]')\n"
            "(directory / 'REPORT.md').write_text('# Result\\nFixture protocol only.\\n')\n"
            "def ref(name):\n"
            "    path = directory / name\n"
            "    return {'path': str(path.relative_to(root)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}\n"
            "result = {'status': 'completed', 'research': research, 'evaluation_completed': True,\n"
            "          'checkpoint_ref': ref('checkpoint.bin'), 'predictions_ref': ref('predictions.json'),\n"
            "          'Overall': 0.5}\n"
            "(directory / 'metrics.json').write_text(json.dumps(result))\n"
            "raise SystemExit(7 if mode == 'fail' else 0)\n",
        )
        self.write(directory / "pyproject.toml", "[project]\nname = 'fixture'\nversion = '0.0.0'\n")
        self.write(directory / "uv.lock", "version = 1\n")
        self.dag["nodes"][node_id] = {
            "experiment_id": node_id,
            "directory": str(directory),
            "legacy": False,
            "method_id": method,
            "protocol_id": "p",
            "status": "ready",
            "comparison_id": comparison_id,
            "controls": list(controls),
            "sources": [],
            "requires_evidence": [],
            "config_ref": self.file_ref(directory / "configs/train.json"),
            "metrics_ref": None,
            "report_ref": None,
            "wandb_url": None,
            "artifact": None,
            "git_commit": None,
            "expected_config": config,
            "code_refs": [
                self.file_ref(directory / filename)
                for filename in ("reproduce.sh", "src/main.py", "pyproject.toml", "uv.lock")
            ],
            "blockers": [],
        }
        self.dag["comparisons"][comparison_id] = {
            "purpose": "baseline",
            "status": "ready",
            "hypothesis": "The baseline yields a valid comparable result.",
            "controls": list(controls),
            "candidates": [node_id],
            "changed_axes": [],
            "changed_fields": [],
            "decision_rule": {
                "primary_metric": "Overall",
                "direction": "maximize",
                "min_effect": 0,
                "scope": "Fixture protocol only",
                "on_support": "Confirm on independent backgrounds",
                "on_against": "Keep the baseline",
                "on_inconclusive": "Collect more independent backgrounds",
            },
        }

    def add_qc_comparison(self):
        self.space["axes"]["D"]["variants"]["optional-qc"] = "Optional QC"
        self.space["methods"]["qc"] = copy.deepcopy(self.space["methods"]["base"])
        self.space["methods"]["qc"]["axes"]["D"] = "optional-qc"
        self.add_node("qc", "qc", "qc-comparison", controls=["baseline"])
        candidate = self.dag["nodes"]["qc"]
        candidate["expected_config"]["data"]["qc"] = True
        self.write_json(candidate["config_ref"]["path"], candidate["expected_config"])
        candidate["config_ref"] = self.file_ref(candidate["config_ref"]["path"])
        comparison = self.dag["comparisons"]["qc-comparison"]
        comparison.update(
            purpose="screen",
            changed_axes=["D"],
            changed_fields=["data.qc"],
            hypothesis="Optional QC improves Overall under the same evaluation.",
        )
        self.write_registries()
        self.commit()

    def pending_evidence(self, comparison_id="baseline-comparison"):
        return {
            "kind": "experiment", "state": "pending", "comparison_id": comparison_id,
            "claim": "The registered comparison can falsify the baseline hypothesis.",
            "node_ids": [], "source_refs": [],
            "next_action": "Run the complete registered comparison",
            "reopen_when": "The evaluation protocol changes",
        }

    def source_evidence(self, kind="literature"):
        self.write("docs/source-note.md", "# Source note\nPublished finding; not a local result.\n")
        return {
            "kind": kind, "state": {"literature": "external", "dataset": "observed", "protocol": "constraint"}[kind],
            "claim": "The public source motivates a testable local hypothesis.",
            "comparison_id": None, "node_ids": [],
            "source_urls": ["https://example.org/primary-source"],
            "source_refs": [self.file_ref("docs/source-note.md")],
            "next_action": "Test the hypothesis in a registered comparison",
            "reopen_when": "Source version or scientific scope changes",
        }

    def test_source_evidence_types_are_valid_but_not_local_conclusions(self):
        for kind in ("literature", "dataset", "protocol"):
            self.ledger["evidence"][kind] = self.source_evidence(kind)
        self.space["methods"]["base"]["evidence_ids"] = ["literature"]
        self.write_registries()
        self.cli("check")
        state = json.loads(self.cli("status", "--json"))
        self.assertEqual(state["initialization"]["source_evidence"], 3)
        self.assertEqual(state["initialization"]["closed_experiment_evidence"], 0)

    def test_external_record_cannot_be_promoted_to_local_evidence(self):
        for kind in ("literature", "dataset", "protocol"):
            with self.subTest(kind=kind):
                item = self.source_evidence(kind)
                item["state"] = "supported"
                self.ledger["evidence"] = {"source": item}
                self.write_registries()
                self.cli("check", ok=False)

    def test_external_reference_cannot_upgrade_method_to_local_supported(self):
        self.ledger["evidence"]["published"] = self.source_evidence()
        self.space["methods"]["base"].update(status="supported", evidence_ids=["published"])
        self.write_registries()
        self.assertIn("closed experiment evidence", self.cli("check", ok=False))

    def test_formal_local_status_requires_the_matching_local_conclusion(self):
        self.ledger["evidence"]["published"] = self.source_evidence()
        method = self.space["methods"]["base"]
        method["evidence_ids"] = ["published"]
        for local_status in ("UNTESTED", "SIGNAL", "CONFIRMED", "REJECTED", "CONTEXT_DEPENDENT", "SUPPORTED"):
            with self.subTest(source_only=local_status):
                method["local_status"] = local_status
                self.write_registries()
                self.cli("check", ok=local_status == "UNTESTED")
        method["local_status"] = "UNTESTED"
        self.write_registries()
        self.commit()
        self.execute("complete")
        self.write_json("proposed-evidence.json", self.completed_evidence())
        self.cli("close", "--file", "proposed-evidence.json")
        self.ledger = json.loads((self.root / REGISTRY_DIR / "evidence_ledger.json").read_text())
        self.dag = json.loads((self.root / REGISTRY_DIR / "experiment_dag.json").read_text())
        method["evidence_ids"] = ["E-baseline"]
        cases = (("SIGNAL", "signal", True), ("SIGNAL", "supported", True),
                 ("CONFIRMED", "supported", True), ("CONFIRMED", "signal", False),
                 ("REJECTED", "not_supported", True), ("REJECTED", "supported", False),
                 ("CONTEXT_DEPENDENT", "inconclusive", True))
        for local_status, evidence_state, accepted in cases:
            with self.subTest(local_status=local_status, evidence_state=evidence_state):
                method["local_status"] = local_status
                self.ledger["evidence"]["E-baseline"]["state"] = evidence_state
                self.write_registries()
                self.cli("check", ok=accepted)

    def test_method_cannot_borrow_another_methods_closed_evidence(self):
        self.add_qc_comparison()
        self.execute("complete")
        self.write_json("proposed-evidence.json", self.completed_evidence())
        self.cli("close", "--file", "proposed-evidence.json")
        self.ledger = json.loads((self.root / REGISTRY_DIR / "evidence_ledger.json").read_text())
        self.dag = json.loads((self.root / REGISTRY_DIR / "experiment_dag.json").read_text())
        self.space["methods"]["base"].update(local_status="CONFIRMED", evidence_ids=["E-baseline"])
        self.write_registries()
        self.cli("check")
        for local_status in ("UNTESTED", "CONFIRMED"):
            with self.subTest(local_status=local_status):
                self.space["methods"]["qc"].update(local_status=local_status, evidence_ids=["E-baseline"])
                self.write_registries()
                self.assertIn("node using this method", self.cli("check", ok=False))

    def test_external_record_cannot_attach_local_nodes_or_close(self):
        item = self.source_evidence()
        self.ledger["evidence"]["published"] = dict(item, node_ids=["baseline"])
        self.write_registries()
        self.assertIn("local comparison or node", self.cli("check", ok=False))
        self.ledger["evidence"] = {}
        self.write_registries()
        self.write_json("proposed-evidence.json", dict(item, id="published"))
        self.assertIn("local experiment conclusion", self.cli("close", "--file", "proposed-evidence.json", ok=False))

    def test_external_evidence_requires_public_url_and_hashed_note(self):
        for field in ("source_urls", "source_refs"):
            with self.subTest(field=field):
                item = self.source_evidence()
                item[field] = []
                self.ledger["evidence"] = {"source": item}
                self.write_registries()
                self.cli("check", ok=False)

    def test_external_and_pending_records_cannot_claim_local_results(self):
        for item in (self.source_evidence(), self.pending_evidence()):
            for field in ("results", "metrics", "local_effect"):
                with self.subTest(kind=item["kind"], field=field):
                    self.ledger["evidence"] = {"claim": dict(item, **{field: {"Overall": 0.5}})}
                    self.write_registries()
                    self.cli("check", ok=False)

    def test_external_evidence_cannot_satisfy_execution_prerequisite(self):
        self.ledger["evidence"]["published"] = self.source_evidence()
        self.dag["nodes"]["baseline"]["requires_evidence"] = ["published"]
        self.write_registries()
        self.commit()
        self.assertIn("local experiment", self.cli("gate", "baseline", ok=False))

    def test_pending_hypothesis_does_not_close_a_completed_comparison(self):
        self.ledger["evidence"]["E-baseline"] = self.pending_evidence()
        self.write_registries()
        self.commit()
        self.execute("complete")
        state = json.loads(self.cli("status", "--json"))
        self.assertEqual(state["needs_evidence"], ["baseline-comparison"])
        self.assertEqual(state["initialization"]["closed_experiment_evidence"], 0)
        self.assertEqual(state["pending_questions"][0]["id"], "E-baseline")
        self.git("add", "--all")
        self.cli("check", "--staged", ok=False)

    def test_pending_comparison_can_close_same_id_as_signal_once(self):
        pending = self.pending_evidence()
        self.ledger["evidence"]["E-baseline"] = pending
        self.write_registries()
        self.commit()
        self.execute("complete")
        result = self.completed_evidence()
        result["state"] = "signal"
        self.write_json("proposed-evidence.json", result)
        self.cli("close", "--file", "proposed-evidence.json")
        ledger = json.loads((self.root / REGISTRY_DIR / "evidence_ledger.json").read_text())
        self.assertEqual(ledger["evidence"]["E-baseline"]["state"], "signal")
        self.assertEqual(ledger["evidence"]["E-baseline"]["hypothesis"], pending["claim"])
        self.assertEqual(json.loads(self.cli("status", "--json"))["needs_evidence"], [])
        self.assertIn("retain closed", self.cli("close", "--file", "proposed-evidence.json", ok=False))

    def test_pending_closure_preserves_comparison_and_hypothesis(self):
        self.ledger["evidence"]["E-baseline"] = self.pending_evidence()
        self.write_registries()
        self.commit()
        self.execute("complete")
        for field in ("comparison_id", "hypothesis"):
            with self.subTest(field=field):
                item = dict(self.completed_evidence(), **{field: "changed"})
                self.write_json("proposed-evidence.json", item)
                self.assertIn("preserve", self.cli("close", "--file", "proposed-evidence.json", ok=False))

    def test_prospective_evidence_dependency_blocks_start_and_has_distinct_graph_edge(self):
        self.add_qc_comparison()
        self.ledger["evidence"]["E-baseline"] = self.pending_evidence()
        self.dag["nodes"]["qc"]["requires_evidence"] = ["E-baseline"]
        self.write_registries()
        self.commit()
        self.cli("check")
        self.assertIn("not closed", self.cli("gate", "qc", ok=False))
        graph = self.cli("graph")
        self.assertIn("-->|control|", graph)
        self.assertIn('-.->|"evidence: E-baseline"|', graph)

    def test_prospective_evidence_dependency_cycle_is_rejected(self):
        self.add_node("second", "base", "second-comparison")
        self.ledger["evidence"] = {
            "E-baseline": self.pending_evidence(),
            "E-second": self.pending_evidence("second-comparison"),
        }
        self.dag["nodes"]["second"]["requires_evidence"] = ["E-baseline"]
        self.dag["nodes"]["baseline"]["requires_evidence"] = ["E-second"]
        self.write_registries()
        self.assertIn("DAG cycle", self.cli("check", ok=False))

    def test_not_supported_cannot_unlock_supported_only_candidate(self):
        self.add_qc_comparison()
        self.ledger["evidence"]["E-baseline"] = self.pending_evidence()
        self.dag["nodes"]["qc"]["requires_evidence"] = ["E-baseline"]
        self.dag["nodes"]["qc"]["evidence_conditions"] = {"E-baseline": ["supported"]}
        self.write_registries()
        self.commit()
        self.execute("complete")
        result = self.completed_evidence()
        result["state"] = "not_supported"
        self.write_json("proposed-evidence.json", result)
        self.cli("close", "--file", "proposed-evidence.json")
        self.commit()
        self.assertIn("evidence_conditions", self.cli("gate", "qc", ok=False))

    def test_evidence_conditions_must_name_required_evidence_and_closed_states(self):
        for conditions in ({"unrequired": ["supported"]}, {"unrequired": ["external"]}):
            with self.subTest(conditions=conditions):
                self.dag["nodes"]["baseline"]["evidence_conditions"] = conditions
                self.write_registries()
                self.cli("check", ok=False)

    def test_unregistered_training_file_changes_are_rejected_without_legacy_registry(self):
        for filename in ("src/main.py", "configs/train.json", "reproduce.sh", "pyproject.toml", "uv.lock"):
            with self.subTest(filename=filename):
                path = f"experiments/unregistered/{filename}"
                self.write(path, "unregistered change\n")
                self.git("add", path)
                self.assertIn("not registered", self.cli("check", "--staged", ok=False))
                self.git("reset", "--", path)

    def test_draft_registration_and_graph_do_not_create_experiment_files(self):
        node = copy.deepcopy(self.dag["nodes"]["baseline"])
        node.update(experiment_id="future", directory="experiments/future", status="draft",
                    config_ref=None, expected_config=None, code_refs=[], comparison_id="future-comparison")
        self.dag["nodes"]["future"] = node
        comparison = copy.deepcopy(self.dag["comparisons"]["baseline-comparison"])
        comparison.update(status="draft", candidates=["future"])
        self.dag["comparisons"]["future-comparison"] = comparison
        self.ledger["evidence"]["E-future"] = self.pending_evidence("future-comparison")
        self.write_registries()
        self.cli("check")
        self.cli("graph")
        self.cli("status")
        self.cli("gate", "future", ok=False)
        self.assertFalse((self.root / "experiments/future").exists())

    def test_registered_fixture_is_usable(self):
        self.cli("check")
        self.cli("status", "--json")
        self.cli("graph")
        self.cli("gate", "baseline")
        metadata = self.bind("baseline")
        self.assertEqual(metadata["experiment_id"], "baseline")

    def test_unknown_method_is_rejected(self):
        self.cli("check")
        self.dag["nodes"]["baseline"]["method_id"] = "missing-method"
        self.write_registries()
        self.assertIn("missing-method", self.cli("check", ok=False))

    def test_unknown_axis_variant_is_rejected(self):
        self.cli("check")
        self.space["methods"]["base"]["axes"]["D"] = "unregistered-qc"
        self.write_registries()
        self.cli("check", ok=False)

    def test_unknown_axis_cannot_bypass_method_space(self):
        self.cli("check")
        self.space["methods"]["base"]["axes"]["X"] = "unregistered-axis"
        self.write_registries()
        self.cli("check", ok=False)

    def test_dangling_dependency_is_rejected(self):
        self.cli("check")
        self.dag["nodes"]["baseline"]["sources"] = ["missing-source"]
        self.write_registries()
        self.assertIn("missing-source", self.cli("check", ok=False))

    def test_new_node_must_belong_to_its_declared_comparison(self):
        self.add_node("extra", "base", "extra-comparison")
        del self.dag["comparisons"]["extra-comparison"]
        self.dag["nodes"]["extra"]["comparison_id"] = "baseline-comparison"
        self.write_registries()
        self.commit()
        self.cli("check", ok=False)
        self.cli("gate", "extra", ok=False)

    def test_dependency_cycle_is_rejected(self):
        self.add_qc_comparison()
        self.cli("check")
        self.dag["nodes"]["baseline"]["sources"] = ["qc"]
        self.write_registries()
        self.cli("check", ok=False)

    def test_cross_protocol_comparison_cannot_claim_attribution(self):
        self.add_qc_comparison()
        self.cli("check")
        self.dag["protocols"]["different-panel"] = copy.deepcopy(self.dag["protocols"]["p"])
        self.dag["nodes"]["qc"]["protocol_id"] = "different-panel"
        self.write_registries()
        self.cli("check", ok=False)

    def test_challenge_protocol_cannot_be_ready_without_frozen_artifacts(self):
        self.dag["protocols"]["p"]["contract_kind"] = "vcc2026"
        self.write_registries()
        output = self.cli("check", ok=False)
        self.assertIn("challenge data_audit must have a frozen source reference", output)

    def test_challenge_ready_rejects_nonofficial_scorer_even_with_matching_hash(self):
        scorer = "docs/research/scorer-fixture.json"
        self.write_json(scorer, {"repository": "custom-implementation", "preset": "vcc2026"})
        ref = self.file_ref(scorer)
        self.dag["protocols"]["p"].update(contract_kind="vcc2026", binding={"scorer": ref}, source_refs=[ref])
        self.write_registries()
        output = self.cli("check", ok=False)
        self.assertIn("fixed official cell-eval2 vcc2026 contract required", output)

    def test_challenge_node_cannot_skip_benchmark_binding(self):
        self.dag["protocols"]["p"].update(contract_kind="vcc2026", status="draft",
                                           binding={"data_audit": {"sha256": "a"*64}})
        self.write_registries()
        output = self.cli("gate", "baseline", ok=False)
        self.assertIn("expected_config.benchmark must bind", output)

    def test_single_axis_label_cannot_hide_extra_config_changes(self):
        self.add_qc_comparison()
        self.cli("check")
        candidate = self.dag["nodes"]["qc"]
        candidate["expected_config"]["model"]["width"] = 128
        self.write_json(candidate["config_ref"]["path"], candidate["expected_config"])
        candidate["config_ref"] = self.file_ref(candidate["config_ref"]["path"])
        self.write_registries()
        self.cli("check", ok=False)

    def test_same_seed_cannot_pair_different_fit_scopes(self):
        self.add_qc_comparison()
        for node_id, outer_split in (("baseline", "fold-a"), ("qc", "fold-b")):
            node = self.dag["nodes"][node_id]
            node["expected_config"]["fit_scope"] = {"outer_split": outer_split, "target_partition": "held-out-targets"}
            self.write_json(node["config_ref"]["path"], node["expected_config"])
            node["config_ref"] = self.file_ref(node["config_ref"]["path"])
        self.write_registries()
        self.assertIn("seed and fit_scope must match", self.cli("check", ok=False))
        node = self.dag["nodes"]["qc"]
        node["expected_config"]["fit_scope"]["outer_split"] = "fold-a"
        self.write_json(node["config_ref"]["path"], node["expected_config"])
        node["config_ref"] = self.file_ref(node["config_ref"]["path"])
        self.write_registries()
        self.cli("check")

    def test_ready_node_requires_its_declared_fit_scope_to_be_frozen_in_config(self):
        node = self.dag["nodes"]["baseline"]
        node.update(status="draft", design={"fit_scope": {"outer_split": None, "target_partition": None}})
        self.write_registries()
        self.cli("check")
        node["status"] = "ready"
        self.write_registries()
        self.assertIn("concrete outer_split", self.cli("check", ok=False))
        node["design"]["fit_scope"] = {"outer_split": "fold-a", "target_partition": "held-out-targets"}
        self.write_registries()
        self.assertIn("expected_config.fit_scope", self.cli("check", ok=False))
        node["expected_config"]["fit_scope"] = copy.deepcopy(node["design"]["fit_scope"])
        self.write_json(node["config_ref"]["path"], node["expected_config"])
        node["config_ref"] = self.file_ref(node["config_ref"]["path"])
        self.write_registries()
        self.cli("check")

    def test_unregistered_experiment_cannot_start(self):
        self.cli("gate", "baseline")
        self.cli("gate", "unregistered", ok=False)

    def test_historical_experiment_cannot_restart(self):
        self.dag["nodes"]["baseline"]["legacy"] = True
        self.write_registries()
        self.commit()
        self.cli("gate", "baseline", ok=False)

    def test_completed_experiment_cannot_restart(self):
        self.dag["nodes"]["baseline"]["status"] = "completed"
        self.write_registries()
        self.commit()
        self.cli("gate", "baseline", ok=False)

    def test_gate_requires_committed_registration(self):
        self.cli("gate", "baseline")
        self.space["methods"]["base"]["description"] = "Changed registration"
        self.write_registries()
        self.cli("gate", "baseline", ok=False)

    def test_ready_label_cannot_overwrite_existing_execution(self):
        self.cli("gate", "baseline")
        self.write_json("experiments/baseline/metrics.json", {"status": "completed"})
        self.assertIn("existing execution", self.cli("gate", "baseline", ok=False))

    def test_config_hash_drift_blocks_training(self):
        self.cli("gate", "baseline")
        config = copy.deepcopy(self.dag["nodes"]["baseline"]["expected_config"])
        config["seed"] = 999
        self.write_json("experiments/baseline/configs/train.json", config)
        self.cli("gate", "baseline", ok=False)

    def test_effective_config_override_blocks_binding(self):
        actual = copy.deepcopy(self.dag["nodes"]["baseline"]["expected_config"])
        actual["model"]["width"] = 128
        self.assertIn("actual_config", self.bind("baseline", actual, ok=False))
        self.bind("baseline")

    def test_unclosed_evidence_dependency_blocks_training(self):
        self.dag["nodes"]["baseline"]["requires_evidence"] = ["E-QC"]
        self.write_registries()
        self.commit()
        self.cli("gate", "baseline", ok=False)

    def test_incomplete_training_cannot_be_recorded_as_completed(self):
        metadata = self.bind("baseline")
        self.write_json(
            "experiments/baseline/metrics.json",
            {"status": "completed", "research": metadata, "Overall": 0.5},
        )
        output = self.cli("record", "baseline", "--metrics", "experiments/baseline/metrics.json", ok=False)
        self.assertIn("evaluation_completed", output)
        dag = json.loads((self.root / REGISTRY_DIR / "experiment_dag.json").read_text())
        self.assertNotEqual(dag["nodes"]["baseline"]["status"], "completed")

    def test_evidence_cannot_close_before_experiment_completion(self):
        self.write_json(
            "proposed-evidence.json",
            {
                "id": "E-baseline",
                "comparison_id": "baseline-comparison",
                "node_ids": ["baseline"],
                "state": "supported",
                "claim": "The baseline is a valid comparator",
                "limitations": "Fixture only",
                "source_refs": [self.file_ref("experiments/baseline/PLAN.md")],
                "next_action": "Run QC comparison",
                "reopen_when": "Protocol changes",
            },
        )
        output = self.cli("close", "--file", "proposed-evidence.json", ok=False)
        self.assertIn("unfinished", output)
        ledger = json.loads((self.root / REGISTRY_DIR / "evidence_ledger.json").read_text())
        self.assertNotIn("E-baseline", ledger["evidence"])

    def fixture_command(self, node_id, mode):
        return [
            sys.executable,
            str(self.root / "experiments" / node_id / "src/main.py"),
            str(SCRIPT), str(self.root), mode,
        ]

    def execute(self, mode, *, node_id="baseline", ok=True, resume=False):
        options = ["--resume"] if resume else []
        return self.cli(
            "execute", *options, node_id, "--", *self.fixture_command(node_id, mode), ok=ok,
        )

    def result_state(self, node_id="baseline"):
        dag = json.loads((self.root / REGISTRY_DIR / "experiment_dag.json").read_text())
        return dag["nodes"][node_id]["status"]

    def test_draft_node_cannot_record_a_completed_result(self):
        result = subprocess.run(
            self.fixture_command("baseline", "complete"),
            cwd=self.root,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.dag["nodes"]["baseline"]["status"] = "draft"
        self.write_registries()
        self.commit()
        self.cli("record", "baseline", "--metrics", "experiments/baseline/metrics.json", ok=False)
        self.assertEqual(self.result_state(), "draft")

    def test_execute_records_a_complete_result(self):
        self.execute("complete")
        self.assertEqual(self.result_state(), "completed")
        self.cli("check")
        self.cli("gate", "baseline", ok=False)

    def test_execute_without_results_does_not_complete(self):
        self.execute("no-result", ok=False)
        self.assertNotEqual(self.result_state(), "completed")

    def test_execute_nonzero_exit_cannot_claim_completion(self):
        self.execute("fail", ok=False)
        self.assertNotEqual(self.result_state(), "completed")

    def test_failure_before_binding_can_retry_without_new_identity(self):
        self.cli(
            "execute", "baseline", "--", sys.executable, "-c", "raise SystemExit(7)",
            ok=False,
        )
        self.assertEqual(self.result_state(), "failed")
        self.cli("retry", "baseline")
        self.assertEqual(self.result_state(), "ready")
        self.commit()
        self.execute("complete")
        self.assertEqual(self.result_state(), "completed")

    def test_failure_after_binding_cannot_restart_with_retry(self):
        # Binding happened, but no final config or metrics exists. The binding
        # marker must still distinguish this from a preflight failure.
        self.execute("no-result", ok=False)
        self.assertEqual(self.result_state(), "failed")
        self.cli("retry", "baseline", ok=False)
        self.assertEqual(self.result_state(), "failed")

    def test_bound_failure_without_final_metrics_can_resume(self):
        self.execute("no-result", ok=False)
        self.assertEqual(self.result_state(), "failed")
        self.dag = json.loads((self.root / REGISTRY_DIR / "experiment_dag.json").read_text())
        self.dag["nodes"]["baseline"]["blockers"] = []
        self.write_registries()
        self.commit()
        self.execute("complete", resume=True)
        self.assertEqual(self.result_state(), "completed")

    def test_completed_result_requires_the_declared_primary_metric(self):
        result = subprocess.run(
            self.fixture_command("baseline", "complete"), cwd=self.root,
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        path = "experiments/baseline/metrics.json"
        metrics = json.loads((self.root / path).read_text())
        del metrics["Overall"]
        self.write_json(path, metrics)
        self.cli("record", "baseline", "--metrics", path, ok=False)
        self.assertEqual(self.result_state(), "ready")

    def test_dotted_primary_metric_must_be_finite(self):
        self.dag["comparisons"]["baseline-comparison"]["decision_rule"]["primary_metric"] = "scores.Overall"
        self.write_registries()
        self.commit()
        result = subprocess.run(
            self.fixture_command("baseline", "complete"), cwd=self.root,
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        path = "experiments/baseline/metrics.json"
        metrics = json.loads((self.root / path).read_text())
        del metrics["Overall"]
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                metrics["scores"] = {"Overall": value}
                self.write_json(path, metrics)
                self.cli("record", "baseline", "--metrics", path, ok=False)
                self.assertEqual(self.result_state(), "ready")
        metrics["scores"] = {"Overall": 0.5}
        self.write_json(path, metrics)
        self.cli("record", "baseline", "--metrics", path)
        self.assertEqual(self.result_state(), "completed")

    def completed_evidence(self):
        return {
            "id": "E-baseline",
            "comparison_id": "baseline-comparison",
            "node_ids": ["baseline"],
            "state": "supported",
            "claim": "The baseline is a valid comparator under fixture protocol p.",
            "limitations": "Synthetic fixture only; no biological conclusion.",
            "source_refs": [
                self.file_ref("experiments/baseline/metrics.json"),
                self.file_ref("experiments/baseline/REPORT.md"),
            ],
            "next_action": "Run the registered QC comparison",
            "reopen_when": "The evaluation protocol changes",
        }

    def test_complete_result_can_close_evidence_and_expose_next_action(self):
        self.execute("complete")
        self.write_json("proposed-evidence.json", self.completed_evidence())
        self.cli("close", "--file", "proposed-evidence.json")
        self.cli("check")
        ledger = json.loads((self.root / REGISTRY_DIR / "evidence_ledger.json").read_text())
        self.assertEqual(ledger["evidence"]["E-baseline"]["state"], "supported")
        self.assertEqual(ledger["evidence"]["E-baseline"]["next_action"], "Run the registered QC comparison")

    def test_report_can_be_completed_after_automatic_result_collection(self):
        self.execute("complete")
        self.write(
            "experiments/baseline/REPORT.md",
            "# Result\nPost-training comparison analysis and limitations.\n",
        )
        self.write_json("proposed-evidence.json", self.completed_evidence())
        self.cli("close", "--file", "proposed-evidence.json")
        self.cli("check")
        ledger = json.loads((self.root / REGISTRY_DIR / "evidence_ledger.json").read_text())
        self.assertIn(
            self.file_ref("experiments/baseline/REPORT.md"),
            ledger["evidence"]["E-baseline"]["source_refs"],
        )

    def test_completed_result_without_evidence_provenance_cannot_close(self):
        self.execute("complete")
        evidence = self.completed_evidence()
        evidence["source_refs"] = []
        self.write_json("proposed-evidence.json", evidence)
        self.cli("close", "--file", "proposed-evidence.json", ok=False)
        ledger = json.loads((self.root / REGISTRY_DIR / "evidence_ledger.json").read_text())
        self.assertNotIn("E-baseline", ledger["evidence"])

    def test_evidence_requires_a_registered_comparison(self):
        self.execute("complete")
        evidence = self.completed_evidence()
        evidence["comparison_id"] = None
        self.write_json("proposed-evidence.json", evidence)
        self.cli("close", "--file", "proposed-evidence.json", ok=False)
        ledger = json.loads((self.root / REGISTRY_DIR / "evidence_ledger.json").read_text())
        self.assertNotIn("E-baseline", ledger["evidence"])

    def test_evidence_cannot_omit_a_comparison_arm(self):
        self.add_qc_comparison()
        self.execute("complete")
        self.commit()
        self.execute("complete", node_id="qc")
        self.assertEqual(self.result_state("qc"), "completed")
        evidence = self.completed_evidence()
        evidence.update(
            id="E-qc", comparison_id="qc-comparison", node_ids=["qc"],
            source_refs=[self.file_ref("experiments/qc/metrics.json")],
        )
        self.write_json("proposed-evidence.json", evidence)
        self.cli("close", "--file", "proposed-evidence.json", ok=False)
        ledger = json.loads((self.root / REGISTRY_DIR / "evidence_ledger.json").read_text())
        self.assertNotIn("E-qc", ledger["evidence"])

    def test_resume_replaces_failed_metrics_after_unrelated_evidence_commit(self):
        self.add_node("unrelated", "base", "unrelated-comparison")
        self.write_registries()
        self.commit()
        result = subprocess.run(
            self.fixture_command("baseline", "complete"), cwd=self.root,
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        metrics_path = "experiments/baseline/metrics.json"
        metrics = json.loads((self.root / metrics_path).read_text())
        original_binding = metrics["research"]
        metrics["status"] = "failed"
        self.write_json(metrics_path, metrics)
        self.cli("record", "baseline", "--metrics", metrics_path)
        self.commit()

        self.execute("complete", node_id="unrelated")
        evidence = self.completed_evidence()
        evidence.update(
            id="E-unrelated", comparison_id="unrelated-comparison", node_ids=["unrelated"],
            source_refs=[self.file_ref("experiments/unrelated/metrics.json")],
        )
        self.write_json("proposed-evidence.json", evidence)
        self.cli("close", "--file", "proposed-evidence.json")
        self.commit()

        self.execute("complete", resume=True)
        self.assertEqual(self.result_state(), "completed")
        resumed = json.loads((self.root / metrics_path).read_text())
        self.assertEqual(resumed["research"], original_binding)

    def test_staged_completed_comparison_requires_closed_evidence(self):
        self.execute("complete")
        self.git("add", "--all")
        self.cli("check", "--staged", ok=False)
        self.write_json("proposed-evidence.json", self.completed_evidence())
        self.cli("close", "--file", "proposed-evidence.json")
        self.git("add", "--all")
        self.cli("check", "--staged")

    def test_staged_check_uses_index_instead_of_valid_worktree(self):
        original = self.dag["nodes"]["baseline"]["method_id"]
        self.dag["nodes"]["baseline"]["method_id"] = "missing-staged-method"
        self.write_registries()
        self.git("add", str(REGISTRY_DIR / "experiment_dag.json"))
        self.dag["nodes"]["baseline"]["method_id"] = original
        self.write_registries()
        self.cli("check")
        self.assertIn("missing-staged-method", self.cli("check", "--staged", ok=False))

    def test_staged_check_hashes_staged_config(self):
        path = Path("experiments/baseline/configs/train.json")
        original = (self.root / path).read_text()
        config = copy.deepcopy(self.dag["nodes"]["baseline"]["expected_config"])
        config["seed"] = 999
        self.write_json(path, config)
        self.git("add", str(path))
        self.write(path, original)
        self.cli("check")
        self.cli("check", "--staged", ok=False)


if __name__ == "__main__":
    unittest.main()

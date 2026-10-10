"""Static checks on deployment/monitoring configuration so CI catches broken YAML/JSON."""
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _docs(path):
    return [d for d in yaml.safe_load_all(path.read_text()) if d]


def test_k8s_manifests_parse_and_have_kinds():
    kinds = {d["kind"] for p in (ROOT / "k8s").rglob("*.yaml") for d in _docs(p)}
    assert {"Deployment", "Service", "Ingress", "HorizontalPodAutoscaler", "Namespace"} <= kinds


def test_deployment_matches_container_contract():
    dep = _docs(ROOT / "k8s/base/deployment.yaml")[0]
    c = dep["spec"]["template"]["spec"]["containers"][0]
    assert c["ports"][0]["containerPort"] == 8000
    assert c["readinessProbe"]["httpGet"]["path"] == "/ready"
    assert c["livenessProbe"]["httpGet"]["path"] == "/health"
    assert dep["spec"]["template"]["metadata"]["annotations"]["prometheus.io/path"] == "/metrics"
    assert "EXPOSE 8000" in (ROOT / "Dockerfile").read_text()


def test_monitoring_copies_in_sync():
    """k8s/monitoring holds copies (kustomize can't read outside its dir) - `make sync-monitoring`."""
    assert (ROOT / "k8s/monitoring/alert_rules.yml").read_text() == (ROOT / "monitoring/alert_rules.yml").read_text()
    assert (ROOT / "k8s/monitoring/heart-risk-api.json").read_text() == \
        (ROOT / "monitoring/grafana/dashboards/heart-risk-api.json").read_text()


def test_grafana_dashboard_queries_reference_exported_metrics():
    from cardiorisk.serving import monitoring as mon  # noqa: F401 - registers metrics

    dash = json.loads((ROOT / "monitoring/grafana/dashboards/heart-risk-api.json").read_text())
    exprs = [t["expr"] for p in dash["panels"] for t in p["targets"]]
    assert len(exprs) >= 10
    known = ("cardiorisk_", "up{", "vector(")
    assert all(any(k in e for k in known) for e in exprs)


def test_alert_rules_valid_yaml():
    groups = yaml.safe_load((ROOT / "monitoring/alert_rules.yml").read_text())["groups"]
    names = {r["alert"] for g in groups for r in g["rules"]}
    assert {"HeartRiskApiDown", "HighErrorRate", "PredictionDrift"} <= names


def test_ci_workflow_has_all_stages():
    wf = yaml.safe_load((ROOT / ".github/workflows/ci-cd.yml").read_text())
    jobs = wf["jobs"]
    assert list(jobs) == ["lint", "test", "train", "docker", "deploy-kind"]
    assert jobs["test"]["needs"] == "lint" and jobs["deploy-kind"]["needs"] == "docker"

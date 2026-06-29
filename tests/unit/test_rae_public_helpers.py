"""Focused public-helper tests for the standalone RAE implementation."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F


def _require_module(name: str):
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        pytest.fail(f"{name} must be importable for the standalone RAE pipeline: {exc}")

def _tiny_linear() -> torch.nn.Linear:
    model = torch.nn.Linear(2, 2, bias=True)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    return model


def _ce_loss(model: torch.nn.Module, x: torch.Tensor, target: int) -> torch.Tensor:
    labels = torch.tensor([target], dtype=torch.long, device=x.device)
    return F.cross_entropy(model(x), labels)


def _acceptance_direction(model: torch.nn.Module,
                          x: torch.Tensor,
                          target: int) -> torch.Tensor:
    gradients = _require_module("scripts_my.rae.gradients")
    dense_candidate_directions = getattr(gradients, "dense_candidate_directions")
    params = [p for p in model.parameters() if p.requires_grad]
    candidate_classes = torch.tensor([[target]], dtype=torch.long, device=x.device)
    return dense_candidate_directions(model, x, candidate_classes, params)[0, 0]


def _unwrap_direction(value: Any) -> Any:
    if isinstance(value, dict):
        for key in ("direction", "directions", "rho", "vector", "flat", "values"):
            if key in value:
                return value[key]
        return value

    for attr in ("direction", "directions", "rho", "vector", "flat"):
        if hasattr(value, attr):
            return getattr(value, attr)

    if isinstance(value, (tuple, list)) and value:
        tensor_items = [item for item in value if isinstance(item, torch.Tensor)]
        if len(tensor_items) == len(value):
            return value
        for item in value:
            if isinstance(item, (torch.Tensor, dict, tuple, list)) or hasattr(item, "direction"):
                return _unwrap_direction(item)

    return value


def _direction_tensors(value: Any, model: torch.nn.Module) -> list[torch.Tensor]:
    direction = _unwrap_direction(value)
    named_params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in named_params]

    if isinstance(direction, dict):
        if all(name in direction for name, _ in named_params):
            return [
                torch.as_tensor(direction[name], dtype=p.dtype, device=p.device).reshape_as(p)
                for name, p in named_params
            ]

        values = list(direction.values())
        if len(values) == len(params) and all(isinstance(v, torch.Tensor) for v in values):
            return [
                torch.as_tensor(v, dtype=p.dtype, device=p.device).reshape_as(p)
                for v, p in zip(values, params)
            ]

    if isinstance(direction, torch.Tensor):
        flat = direction.detach().reshape(-1)
        expected = sum(p.numel() for p in params)
        if flat.numel() != expected:
            pytest.fail(
                f"Flat acceptance direction has {flat.numel()} values, "
                f"but the tiny model has {expected} trainable parameters"
            )

        pieces: list[torch.Tensor] = []
        offset = 0
        for param in params:
            size = param.numel()
            pieces.append(flat[offset: offset + size].to(param).reshape_as(param))
            offset += size
        return pieces

    if isinstance(direction, (tuple, list)):
        if len(direction) != len(params):
            pytest.fail(
                f"Acceptance direction returned {len(direction)} tensors, "
                f"but the tiny model has {len(params)} trainable tensors"
            )
        return [
            torch.as_tensor(item, dtype=p.dtype, device=p.device).detach().reshape_as(p)
            for item, p in zip(direction, params)
        ]

    pytest.fail(f"Unsupported acceptance direction result type: {type(direction)!r}")


def _flat_direction(value: Any, model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([item.detach().reshape(-1).cpu() for item in _direction_tensors(value, model)])


@pytest.mark.unit
def test_gradient_space_all_selects_every_trainable_parameter() -> None:
    gradient_space = _require_module("scripts_my.rae.gradient_space")
    select_gradient_parameters = getattr(gradient_space, "select_gradient_parameters")

    class TinyNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block = torch.nn.Linear(2, 3)
            self.fc = torch.nn.Linear(3, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(self.block(x))

    model = TinyNet()

    classifier_names = [name for name, _ in select_gradient_parameters(model, "classifier")]
    last_block_names = [name for name, _ in select_gradient_parameters(model, "last_block")]
    all_names = [name for name, _ in select_gradient_parameters(model, "all")]

    assert classifier_names == ["fc.weight", "fc.bias"]
    assert last_block_names == ["block.weight", "block.bias"]
    assert all_names == ["block.weight", "block.bias", "fc.weight", "fc.bias"]


@torch.no_grad()
def _apply_direction(model: torch.nn.Module, direction: Any, step_size: float) -> None:
    for param, update in zip(
        [p for p in model.parameters() if p.requires_grad],
        _direction_tensors(direction, model),
    ):
        param.add_(step_size * update)


def _pairwise_k(target_direction: Any,
                reference_direction: Any,
                model: torch.nn.Module) -> float:
    target_flat = _flat_direction(target_direction, model)
    reference_flat = _flat_direction(reference_direction, model)
    return float(torch.dot(target_flat, reference_flat).item())


def _as_scalar(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0].item())
    if isinstance(value, np.ndarray):
        return float(value.reshape(-1)[0])
    return float(value)


def _validation_score_from_same_other(same: torch.Tensor,
                                      other: torch.Tensor) -> float:
    score = _require_module("scripts_my.rae.score")
    validation_scores_from_k = getattr(score, "validation_scores_from_k")
    k_values = torch.cat([same, other]).detach().cpu().numpy()[None, :]
    ref_labels = np.concatenate(
        [
            np.zeros(same.numel(), dtype=np.int64),
            np.ones(other.numel(), dtype=np.int64),
        ]
    )
    candidate_classes = np.array([0], dtype=np.int64)
    validation, _, _ = validation_scores_from_k(
        k_values, ref_labels, candidate_classes)
    return float(validation.reshape(-1)[0])


@pytest.mark.unit
def test_acceptance_direction_reduces_ce_on_tiny_linear_model() -> None:
    model = _tiny_linear()
    x = torch.tensor([[1.0, -0.5]], dtype=torch.float32)
    before = _ce_loss(model, x, target=0).item()

    direction = _acceptance_direction(model, x, target=0)
    _apply_direction(model, direction, step_size=0.05)

    after = _ce_loss(model, x, target=0).item()
    assert after < before


@pytest.mark.unit
def test_k_sign_predicts_first_order_reference_loss_change() -> None:
    x = torch.tensor([[1.0, -0.5]], dtype=torch.float32)

    for reference_label, expected_k_sign, expected_delta_sign in (
        (0, 1.0, -1.0),
        (1, -1.0, 1.0),
    ):
        model = _tiny_linear()
        target_direction = _acceptance_direction(model, x, target=0)
        reference_direction = _acceptance_direction(
            model, x, target=reference_label)
        k_value = _pairwise_k(target_direction, reference_direction, model)

        before = _ce_loss(model, x, target=reference_label).item()
        _apply_direction(model, target_direction, step_size=1e-3)
        after = _ce_loss(model, x, target=reference_label).item()
        delta = after - before

        assert k_value * expected_k_sign > 0.0
        assert delta * expected_delta_sign > 0.0


@pytest.mark.unit
def test_v_c_requires_positive_same_class_support_and_ranking() -> None:
    same = torch.tensor([0.2, -0.1], dtype=torch.float32)
    other = torch.tensor([0.1, -0.2], dtype=torch.float32)
    assert _validation_score_from_same_other(same, other) == pytest.approx(0.5)

    negative_but_ranked = _validation_score_from_same_other(
        same=torch.tensor([-0.1], dtype=torch.float32),
        other=torch.tensor([-0.5], dtype=torch.float32),
    )
    positive_but_not_ranked = _validation_score_from_same_other(
        same=torch.tensor([0.1], dtype=torch.float32),
        other=torch.tensor([0.2], dtype=torch.float32),
    )

    assert negative_but_ranked == pytest.approx(0.0)
    assert positive_but_not_ranked == pytest.approx(0.0)


@pytest.mark.unit
def test_vectorized_batch_v_c_matches_scalar_definition() -> None:
    score = _require_module("scripts_my.rae.score")
    batch_validation = getattr(score, "batch_validation_scores_from_k")
    scalar_validation = getattr(score, "validation_scores_from_k")

    torch.manual_seed(0)
    k_values = torch.randn(3, 4, 8, dtype=torch.float32)
    ref_labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long)
    candidate_classes = torch.tensor(
        [[0, 1, 2, 3], [3, 2, 1, 0], [1, 1, 2, 2]],
        dtype=torch.long,
    )

    v_batch, rank_batch, same_pos_batch = batch_validation(
        k_values, ref_labels, candidate_classes)
    expected_v = []
    expected_rank = []
    expected_same_pos = []
    for row_idx in range(k_values.shape[0]):
        v, rank, same_pos = scalar_validation(
            k_values[row_idx].numpy(),
            ref_labels.numpy(),
            candidate_classes[row_idx].numpy(),
        )
        expected_v.append(v)
        expected_rank.append(rank)
        expected_same_pos.append(same_pos)

    np.testing.assert_allclose(v_batch.numpy(), np.stack(expected_v), atol=1e-7)
    np.testing.assert_allclose(
        rank_batch.numpy(), np.stack(expected_rank), atol=1e-7)
    np.testing.assert_allclose(
        same_pos_batch.numpy(), np.stack(expected_same_pos), atol=1e-7)


@pytest.mark.unit
def test_uniform_rejection_evidence_uses_same_class_mean_k() -> None:
    score = _require_module("scripts_my.rae.score")
    rejection_evidence = getattr(score, "class_rejection_evidence_from_k")

    k_values = torch.tensor([[-0.8, 0.2, -0.4, 0.6]], dtype=torch.float32)
    ref_labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    candidate_classes = torch.tensor([[0, 1]], dtype=torch.long)

    id_evidence, ood_evidence, k_mean = rejection_evidence(
        k_values, ref_labels, candidate_classes)

    torch.testing.assert_close(
        k_mean, torch.tensor([[-0.3, 0.1]], dtype=torch.float64))
    torch.testing.assert_close(
        id_evidence, torch.tensor([[0.3, 0.0]], dtype=torch.float64))
    torch.testing.assert_close(
        ood_evidence, torch.tensor([[0.7, 1.0]], dtype=torch.float64))


@pytest.mark.unit
def test_vectorized_classifier_k_matches_dense_directions() -> None:
    score = _require_module("scripts_my.rae.score")
    gradients = _require_module("scripts_my.rae.gradients")
    classifier_pairwise_k = getattr(score, "classifier_pairwise_k")
    classifier_candidate_directions_dense = getattr(
        gradients, "classifier_candidate_directions_dense")
    classifier_direction_dense = getattr(gradients, "classifier_direction_dense")

    torch.manual_seed(1)
    target_features = torch.randn(3, 5)
    target_probs = torch.softmax(torch.randn(3, 4), dim=1)
    candidate_classes = torch.tensor(
        [[0, 1, 2], [3, 2, 1], [1, 1, 0]],
        dtype=torch.long,
    )
    ref_features = torch.randn(6, 5)
    ref_probs = torch.softmax(torch.randn(6, 4), dim=1)
    ref_labels = torch.tensor([0, 1, 2, 3, 0, 1], dtype=torch.long)

    k_compact = classifier_pairwise_k(
        target_features,
        target_probs,
        candidate_classes,
        ref_features,
        ref_probs,
        ref_labels,
        include_bias=True,
    )
    target_dirs = classifier_candidate_directions_dense(
        target_features,
        target_probs,
        candidate_classes,
        include_bias=True,
    )
    ref_dirs = classifier_direction_dense(
        ref_features,
        ref_probs,
        ref_labels,
        include_bias=True,
    )
    k_dense = torch.einsum("bkd,rd->bkr", target_dirs, ref_dirs)

    torch.testing.assert_close(k_compact, k_dense, atol=1e-6, rtol=1e-6)


@pytest.mark.unit
def test_vmap_dense_directions_match_autograd_loop() -> None:
    gradients = _require_module("scripts_my.rae.gradients")
    dense_reference_directions = getattr(gradients, "dense_reference_directions")
    dense_candidate_directions = getattr(gradients, "dense_candidate_directions")
    normalized_grad_vector = getattr(gradients, "normalized_grad_vector")

    class TinyNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block = torch.nn.Linear(3, 4)
            self.fc = torch.nn.Linear(4, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(torch.tanh(self.block(x)))

    torch.manual_seed(2)
    model = TinyNet().eval()
    data = torch.randn(5, 3)
    labels = torch.tensor([0, 1, 0, 1, 1], dtype=torch.long)
    candidate_classes = torch.tensor(
        [[0, 1], [1, 0], [0, 0], [1, 1], [0, 1]],
        dtype=torch.long,
    )

    for params in (
        [model.block.weight, model.block.bias],
        [param for param in model.parameters() if param.requires_grad],
    ):
        reference = dense_reference_directions(model, data, labels, params)
        expected_reference = []
        for idx in range(data.shape[0]):
            loss = F.cross_entropy(model(data[idx:idx + 1]), labels[idx:idx + 1])
            expected_reference.append(normalized_grad_vector(loss, params))
        torch.testing.assert_close(
            reference,
            torch.stack(expected_reference),
            atol=1e-6,
            rtol=1e-6,
        )

        candidates = dense_candidate_directions(
            model, data, candidate_classes, params)
        expected_rows = []
        for idx in range(data.shape[0]):
            row = []
            for class_id in candidate_classes[idx].tolist():
                target = torch.tensor([class_id], dtype=torch.long)
                loss = F.cross_entropy(model(data[idx:idx + 1]), target)
                row.append(normalized_grad_vector(loss, params))
            expected_rows.append(torch.stack(row))
        torch.testing.assert_close(
            candidates,
            torch.stack(expected_rows),
            atol=1e-6,
            rtol=1e-6,
        )


@pytest.mark.unit
def test_eid_ood_scores_are_higher_for_lower_evidence() -> None:
    score = _require_module("scripts_my.rae.score")
    ood_score_from_eid = getattr(score, "ood_score_from_eid")

    high_evidence = torch.tensor([0.8], dtype=torch.float32)
    low_evidence = torch.tensor([0.08], dtype=torch.float32)

    assert ood_score_from_eid(low_evidence, "neglog_eid") > ood_score_from_eid(
        high_evidence, "neglog_eid"
    )
    assert ood_score_from_eid(low_evidence, "neg_eid") > ood_score_from_eid(
        high_evidence, "neg_eid"
    )


@pytest.mark.unit
def test_candidate_mode_pred_uses_argmax_class_only() -> None:
    score = _require_module("scripts_my.rae.score")
    candidate_fn = getattr(score, "candidate_classes_from_probs")

    probs = torch.tensor(
        [[0.1, 0.7, 0.2], [0.9, 0.05, 0.05]],
        dtype=torch.float32,
    )

    np.testing.assert_array_equal(
        candidate_fn(probs, "all").detach().cpu().numpy(),
        np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        candidate_fn(probs, "pred").detach().cpu().numpy(),
        np.array([[1], [0]], dtype=np.int64),
    )


@pytest.mark.unit
def test_candidate_mode_pred_can_be_claim_bearing_variant() -> None:
    diagnostics = _require_module("scripts_my.rae.diagnostics")
    claim_status = getattr(diagnostics, "diagnostics_claim_status")

    claim_bearing, reasons = claim_status(
        {"diagnostics_status": "pass"},
        max_target_samples=None,
        candidate_mode="pred",
    )

    assert claim_bearing is True
    assert reasons == []


@pytest.mark.unit
def test_pred_scores_are_derived_from_all_candidate_artifact(tmp_path: Path) -> None:
    eval_mod = _require_module("scripts_my.rae.eval")
    derive_pred = getattr(eval_mod, "derive_pred_split_scores_from_all")

    path = tmp_path / "all_scores.npz"
    np.savez(
        path,
        pred=np.array([1, 0], dtype=np.int64),
        label=np.array([1, -1], dtype=np.int64),
        q_max=np.array([0.7, 0.8], dtype=np.float64),
        v_pred=np.array([0.5, 0.25], dtype=np.float64),
        eid_pred=np.array([0.35, 0.2], dtype=np.float64),
        candidate_classes=np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int64),
        q_c=np.array([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]], dtype=np.float64),
        v_c=np.array([[0.2, 0.5, 0.4], [0.25, 0.3, 0.1]], dtype=np.float64),
        e_c=np.array([[0.02, 0.35, 0.08], [0.2, 0.03, 0.01]], dtype=np.float64),
        rank_only_scores=np.array(
            [[0.6, 0.7, 0.8], [0.9, 0.2, 0.1]], dtype=np.float64),
        same_positive_rates=np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float64),
        v_c_label_shuffle=np.array(
            [[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]], dtype=np.float64),
        e_c_label_shuffle=np.array(
            [[0.001, 0.014, 0.006], [0.032, 0.005, 0.006]],
            dtype=np.float64),
    )

    with np.load(path, allow_pickle=True) as arrays:
        derived = derive_pred(arrays)

    np.testing.assert_array_equal(
        derived["candidate_classes"], np.array([[1], [0]], dtype=np.int64))
    np.testing.assert_array_equal(derived["best_class"], np.array([1, 0]))
    np.testing.assert_allclose(derived["eid"], np.array([0.35, 0.2]))
    np.testing.assert_allclose(derived["q_c"], np.array([[0.7], [0.8]]))
    np.testing.assert_allclose(derived["v_c"], np.array([[0.5], [0.25]]))
    np.testing.assert_allclose(derived["e_c"], np.array([[0.35], [0.2]]))
    np.testing.assert_allclose(
        derived["rank_only_scores"], np.array([[0.7], [0.9]]))
    np.testing.assert_allclose(
        derived["same_positive_rates"], np.array([[0.2], [0.4]]))
    np.testing.assert_allclose(
        derived["eid_label_shuffle"], np.array([0.014, 0.032]))


@pytest.mark.unit
def test_reference_sampling_requires_full_per_class_quota() -> None:
    reference = _require_module("scripts_my.rae.reference")
    config_mod = _require_module("scripts_my.rae.config")
    sample_reference_indices = getattr(reference, "sample_reference_indices")
    ReferenceConfig = getattr(config_mod, "ReferenceConfig")

    candidates = {
        "scan_index": np.array([0, 1, 2], dtype=np.int64),
        "index": np.array([10, 11, 12], dtype=np.int64),
        "labels": np.array([0, 0, 1], dtype=np.int64),
        "pred": np.array([0, 0, 1], dtype=np.int64),
        "conf": np.array([0.9, 0.8, 0.95], dtype=np.float64),
        "correct": np.array([True, True, True]),
        "image_name": np.array(["a.png", "b.png", "c.png"]),
    }
    config = ReferenceConfig(
        dataset="cifar10",
        per_class=2,
        filter_name="correct",
        min_confidence=0.9,
        seed=0,
    )

    with pytest.raises(RuntimeError, match="Not enough RAE reference samples"):
        sample_reference_indices(candidates, config, num_classes=2)


@pytest.mark.unit
def test_reference_manifest_reuse_requires_current_model_identity() -> None:
    reference = _require_module("scripts_my.rae.reference")
    config_mod = _require_module("scripts_my.rae.config")
    ReferenceConfig = getattr(config_mod, "ReferenceConfig")
    reusable = getattr(reference, "reference_manifest_is_reusable")
    identity_fn = getattr(reference, "train_candidate_metadata_identity")

    config = ReferenceConfig(
        dataset="cifar10",
        per_class=1,
        filter_name="correct",
        min_confidence=0.9,
        seed=0,
    )
    identity = identity_fn(
        "cifar10",
        checkpoint="a.ckpt",
        checkpoint_sha256="sha-a",
        model_arch="TinyNet",
        num_classes=10,
    )
    manifest = {
        "schema_version": getattr(config_mod, "CACHE_SCHEMA_VERSION"),
        "reference_config": {
            "dataset": "cifar10",
            "per_class": 1,
            "filter_name": "correct",
            "min_confidence": 0.9,
            "seed": 0,
        },
        "checkpoint_sha256": "sha-a",
        "model_arch": "TinyNet",
        "num_classes": 10,
        "quota_satisfied": True,
        "quota_per_class": 1,
        "per_class_counts": {"0": 1},
        "train_candidate_metadata": {"identity": identity},
    }

    assert reusable(
        manifest,
        config,
        expected_metadata_identity=identity,
        checkpoint_sha256="sha-a",
        model_arch="TinyNet",
        num_classes=10,
    )
    assert not reusable(
        manifest,
        config,
        expected_metadata_identity=identity_fn(
            "cifar10",
            checkpoint="b.ckpt",
            checkpoint_sha256="sha-b",
            model_arch="TinyNet",
            num_classes=10,
        ),
        checkpoint_sha256="sha-b",
        model_arch="TinyNet",
        num_classes=10,
    )


@pytest.mark.unit
def test_gradient_bank_manifest_reuse_checks_model_and_reference_identity() -> None:
    eval_mod = _require_module("scripts_my.rae.eval")
    reusable = getattr(eval_mod, "ref_grad_bank_manifest_is_reusable")
    split_score_file_stem = getattr(eval_mod, "split_score_file_stem")

    args = types.SimpleNamespace(dataset="cifar10", gradient_space="classifier")
    reference_manifest = {
        "selected_sample_hash": "ref-hash",
        "reference_config_id": "correct_rpc1",
        "selected_count": 10,
    }
    checkpoint = "a.ckpt"
    manifest = {
        "artifact": "rae_gradient_bank",
        "schema_version": getattr(_require_module("scripts_my.rae.config"),
                                  "CACHE_SCHEMA_VERSION"),
        "dataset": "cifar10",
        "gradient_config": {"gradient_space": "classifier"},
        "reference_set_hash": "ref-hash",
        "reference_config_id": "correct_rpc1",
        "checkpoint_resolved": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": "sha-a",
        "model_arch": "ResNet18_32x32",
        "num_classes": 10,
        "bank_type": "classifier_compact",
        "reference_count": 10,
    }

    assert reusable(
        manifest,
        args,
        reference_manifest,
        checkpoint=checkpoint,
        checkpoint_sha256="sha-a",
    )
    assert not reusable(
        {**manifest, "checkpoint_sha256": "sha-b"},
        args,
        reference_manifest,
        checkpoint=checkpoint,
        checkpoint_sha256="sha-a",
    )
    assert split_score_file_stem("nearood", "cifar100") == "nearood_cifar100"
    assert split_score_file_stem("id", "cifar10") == "id"


@pytest.mark.unit
def test_openood_export_conf_is_negative_ood_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluators_stub = types.ModuleType("openood.evaluators")
    evaluators_stub.__path__ = []
    metrics_stub = types.ModuleType("openood.evaluators.metrics")
    metrics_stub.compute_all_metrics = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "openood.evaluators", evaluators_stub)
    monkeypatch.setitem(sys.modules, "openood.evaluators.metrics", metrics_stub)

    metrics = _require_module("scripts_my.rae.metrics")

    pred = np.array([0, 1, 2], dtype=np.int64)
    label = np.array([0, 1, -1], dtype=np.int64)
    ood_score = np.array([-2.0, 0.0, 3.5], dtype=np.float64)

    exported_pred, conf, exported_label = metrics.score_tuple_from_ood(
        pred, ood_score, label)

    np.testing.assert_array_equal(exported_pred, pred)
    np.testing.assert_array_equal(exported_label, label)
    np.testing.assert_allclose(conf, -ood_score)


@pytest.mark.unit
def test_experiment_diagnostics_write_gate8_and_gate9_tables(tmp_path: Path) -> None:
    diagnostics = _require_module("scripts_my.rae.diagnostics")
    run_experiment_diagnostics = getattr(diagnostics, "run_experiment_diagnostics")
    write_diagnostic_results = getattr(diagnostics, "write_diagnostic_results")
    experiment_required_gates = getattr(diagnostics, "experiment_required_gates")

    run_rows = []
    metric_rows = []
    for gradient_space, auroc_offset in (("classifier", 0.0), ("last_block", 1.0)):
        for reference_per_class in (4, 8):
            run_id = f"{gradient_space}_{reference_per_class}"
            run_rows.append(
                {
                    "run_id": run_id,
                    "run_dir": str(tmp_path / run_id),
                    "run_status": "complete",
                    "reference_per_class": reference_per_class,
                    "reference_seed": 0,
                    "gradient_space": gradient_space,
                    "candidate_mode": "all",
                    "claim_scope": "exact_full_candidate_rae",
                }
            )
            metric_rows.append(
                {
                    "run_id": run_id,
                    "run_dir": str(tmp_path / run_id),
                    "reference_per_class": reference_per_class,
                    "reference_seed": 0,
                    "gradient_space": gradient_space,
                    "candidate_mode": "all",
                    "claim_scope": "exact_full_candidate_rae",
                    "score_rule": "neglog_eid",
                    "ood_dataset": "nearood",
                    "FPR@95": 20.0 - auroc_offset,
                    "AUROC": 80.0 + reference_per_class + auroc_offset,
                    "AUPR_IN": 70.0,
                    "AUPR_OUT": 60.0,
                    "ACC": 90.0 + auroc_offset,
                }
            )

    results = run_experiment_diagnostics(
        tmp_path,
        metric_rows,
        run_rows,
        reference_sizes=[4, 8],
        reference_seeds=[0],
        gradient_spaces=["classifier", "last_block"],
        candidate_modes=["all"],
    )
    manifest = write_diagnostic_results(
        tmp_path,
        results,
        required_gates=experiment_required_gates(
            [4, 8], [0], ["classifier", "last_block"]
        ),
    )

    assert manifest["diagnostics_status"] == "pass"
    assert manifest["deferred_gates"] == ["gate05"]
    assert {result["gate_id"] for result in manifest["results"]} == {"gate08", "gate09"}
    assert (tmp_path / "gate08_reference_size_stability.csv").exists()
    assert (tmp_path / "gate08_reference_size_trends.csv").exists()
    assert (tmp_path / "gate09_gradient_space_ablation.csv").exists()
    assert "candidate_mode" in (tmp_path / "gate08_reference_size_stability.csv").read_text()
    assert "candidate_mode" in (tmp_path / "gate09_gradient_space_ablation.csv").read_text()

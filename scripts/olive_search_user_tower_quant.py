import argparse
import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from olive import run as olive_run

from nanoRecSys.config import settings
from nanoRecSys.models.towers import ItemTower, TransformerUserTower

LOWER_DOT_THRESHOLD = 0.95
UPPER_DOT_THRESHOLD = 1.01


@dataclass
class CandidateResult:
    rank: int
    model_id: str
    model_path: str
    latency_ms: float
    dot_min: float
    dot_max: float
    all_within_threshold: bool


class ONNXUserTower:
    def __init__(self, path: Path, intra_threads: int = 1, inter_threads: int = 1):
        sess_options = ort.SessionOptions()  # type: ignore
        sess_options.intra_op_num_threads = intra_threads
        sess_options.inter_op_num_threads = inter_threads
        self.session = ort.InferenceSession(  # type: ignore
            str(path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

    def encode(self, item_seq: torch.Tensor) -> torch.Tensor:
        inputs = {"item_seq": item_seq.detach().cpu().numpy()}
        out = self.session.run(["user_embedding"], inputs)
        return torch.from_numpy(out[0])


def build_torch_user_tower() -> TransformerUserTower:
    item_map = np.load(settings.processed_data_dir / "item_map.npy")
    n_items = len(item_map)

    dummy_item_tower = ItemTower(
        vocab_size=n_items,
        embed_dim=settings.embed_dim,
        output_dim=settings.tower_out_dim,
        hidden_dims=settings.towers_hidden_dims,
        use_projection=settings.use_projection,
    )

    torch_model = TransformerUserTower(
        vocab_size=n_items,
        embed_dim=settings.tower_out_dim,
        output_dim=settings.tower_out_dim,
        max_seq_len=settings.max_seq_len,
        n_heads=settings.transformer_heads,
        n_layers=settings.transformer_layers,
        dropout=settings.transformer_dropout,
        swiglu_hidden_dim=settings.swiglu_hidden_dim,
        shared_embedding=dummy_item_tower,
    )

    user_tower_path = settings.artifacts_dir / "user_tower.pth"
    state_dict = torch.load(user_tower_path, map_location="cpu", weights_only=False)
    torch_model.load_state_dict(state_dict)
    torch_model.eval()
    return torch_model


def sample_user_sequences(num_users: int, seq_len: int, seed: int) -> torch.Tensor:
    data = np.load(settings.processed_data_dir / "seq_test_sequences.npy")
    if num_users > data.shape[0]:
        raise ValueError(
            f"Requested {num_users} users but seq_test_sequences only has {data.shape[0]} rows."
        )
    rng = np.random.default_rng(seed)
    idx = rng.choice(data.shape[0], size=num_users, replace=False)
    sampled = data[idx]
    return torch.tensor(sampled[:, :seq_len], dtype=torch.long)


def benchmark_latency_ms(
    model_path: Path,
    input_batch: torch.Tensor,
    warmup: int,
    repeats: int,
    intra_threads: int,
    inter_threads: int,
) -> float:
    model = ONNXUserTower(
        model_path,
        intra_threads=intra_threads,
        inter_threads=inter_threads,
    )

    for _ in range(warmup):
        _ = model.encode(input_batch)

    start = time.perf_counter()
    for _ in range(repeats):
        _ = model.encode(input_batch)
    end = time.perf_counter()

    return ((end - start) * 1000.0) / repeats


def copy_onnx_with_external_data(src_onnx: Path, dst_onnx: Path) -> None:
    dst_onnx.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_onnx, dst_onnx)

    src_data = src_onnx.with_suffix(src_onnx.suffix + ".data")
    if src_data.exists():
        dst_data = dst_onnx.with_suffix(dst_onnx.suffix + ".data")
        shutil.copy2(src_data, dst_data)


def make_olive_config(
    input_model: Path,
    output_dir: Path,
    seq_len: int,
    max_iter: int,
    sampler: str,
    seed: int,
    intra_threads: int,
    inter_threads: int,
    quant_search_space: str,
    enable_transformer_optimization: bool,
) -> dict[str, Any]:
    if quant_search_space == "extended":
        quant_pass_config: dict[str, Any] = {
            "precision": "SEARCHABLE_VALUES",
            "per_channel": "SEARCHABLE_VALUES",
            "reduce_range": "SEARCHABLE_VALUES",
            "quant_preprocess": "SEARCHABLE_VALUES",
        }
    else:
        quant_pass_config = {
            "precision": "int8",
            "per_channel": "SEARCHABLE_VALUES",
            "reduce_range": "SEARCHABLE_VALUES",
        }

    passes: dict[str, Any] = {}
    if enable_transformer_optimization:
        passes["transformer_opt"] = {
            "type": "OrtTransformersOptimization",
            "config": {
                "model_type": "gpt2",
                "num_heads": settings.transformer_heads,
                "hidden_size": settings.tower_out_dim,
                "opt_level": 99,
                "only_onnxruntime": True,
                "float16": False,
                "input_int32": False,
                "use_gpu": False,
            },
        }

    passes["quantize"] = {
        "type": "OnnxDynamicQuantization",
        "config": quant_pass_config,
    }

    search_strategy_config: dict[str, Any] = {
        "execution_order": "joint",
        "sampler": sampler,
        "max_iter": max_iter,
    }
    if sampler in {"random", "tpe"}:
        search_strategy_config["sampler_config"] = {"seed": seed}

    return {
        "input_model": {
            "type": "ONNXModel",
            "model_path": str(input_model),
        },
        "systems": {
            "local_system": {
                "type": "LocalSystem",
                "accelerators": [
                    {
                        "device": "cpu",
                        "execution_providers": ["CPUExecutionProvider"],
                    }
                ],
            }
        },
        # Use ONE as input, torch.ones(input_shape, dtype=input_type)
        # https://github.com/microsoft/Olive/blob/main/olive/data/component/dataset.py
        "data_configs": [
            {
                "name": "latency_dummy_data",
                "type": "DummyDataContainer",
                "load_dataset_config": {
                    "params": {
                        "input_shapes": [[1, seq_len]],
                        "input_names": ["item_seq"],
                        "input_types": ["int64"],
                    }
                },
            }
        ],
        "evaluators": {
            "latency_eval": {
                "metrics": [
                    {
                        "name": "latency",
                        "type": "latency",
                        "data_config": "latency_dummy_data",
                        "sub_types": [
                            {
                                "name": "avg",
                                "priority": 1,
                            }
                        ],
                        "user_config": {
                            "inference_settings": {
                                "onnx": {
                                    "execution_provider": ["CPUExecutionProvider"],
                                    "session_options": {
                                        "intra_op_num_threads": intra_threads,
                                        "inter_op_num_threads": inter_threads,
                                    },
                                }
                            }
                        },
                    }
                ]
            }
        },
        "engine": {
            "search_strategy": search_strategy_config,
            "evaluator": "latency_eval",
            "host": "local_system",
            "target": "local_system",
            "clean_cache": True,
            "cache_dir": str(output_dir / "cache"),
            "plot_pareto_frontier": False,
        },
        "passes": passes,
        "output_dir": str(output_dir),
        "log_severity_level": 1,
    }


def run_search(args: argparse.Namespace) -> None:
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    seq_data = sample_user_sequences(
        num_users=args.num_users,
        seq_len=args.seq_len,
        seed=args.eval_seed,
    )

    torch_model = build_torch_user_tower()
    with torch.inference_mode():
        torch_output = torch_model.encode(seq_data)

    results: list[CandidateResult] = []
    latency_probe = seq_data[:1]

    run_name = (
        f"with_transformer_opt_{args.sampler}_{args.max_iter}"
        if args.enable_transformer_optimization
        else f"quant_only_{args.sampler}_{args.max_iter}"
    )
    run_dir = work_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    olive_config = make_olive_config(
        input_model=args.input_model.resolve(),
        output_dir=run_dir,
        seq_len=args.seq_len,
        max_iter=args.max_iter,
        sampler=args.sampler,
        seed=args.search_seed,
        intra_threads=args.intra_threads,
        inter_threads=args.inter_threads,
        quant_search_space=args.quant_search_space,
        enable_transformer_optimization=args.enable_transformer_optimization,
    )

    config_path = run_dir / "olive_user_tower_quant_config.json"
    config_path.write_text(json.dumps(olive_config, indent=2), encoding="utf-8")

    workflow_output = olive_run(str(config_path))
    if workflow_output.has_output_model():
        candidates = workflow_output.get_output_models()
        for rank, candidate in enumerate(candidates, start=1):
            if not candidate.model_path:
                continue
            candidate_path = Path(candidate.model_path)
            onnx_model = ONNXUserTower(
                candidate_path,
                intra_threads=args.intra_threads,
                inter_threads=args.inter_threads,
            )
            onnx_output = onnx_model.encode(seq_data)
            dots = (torch_output * onnx_output).sum(dim=-1).detach().cpu().numpy()

            dot_min = float(np.min(dots))
            dot_max = float(np.max(dots))
            all_within = bool(
                np.all((dots >= LOWER_DOT_THRESHOLD) & (dots <= UPPER_DOT_THRESHOLD))
            )

            latency_ms = benchmark_latency_ms(
                candidate_path,
                latency_probe,
                warmup=args.latency_warmup,
                repeats=args.latency_repeats,
                intra_threads=args.intra_threads,
                inter_threads=args.inter_threads,
            )

            results.append(
                CandidateResult(
                    rank=rank,
                    model_id=candidate.model_id,
                    model_path=str(candidate_path),
                    latency_ms=latency_ms,
                    dot_min=dot_min,
                    dot_max=dot_max,
                    all_within_threshold=all_within,
                )
            )

    accepted = [result for result in results if result.all_within_threshold]
    if not accepted:
        report_path = work_dir / "olive_user_tower_quant_report.json"
        report_path.write_text(
            json.dumps(
                {
                    "selected_model": None,
                    "constraint": {
                        "dot_product_min": LOWER_DOT_THRESHOLD,
                        "dot_product_max": UPPER_DOT_THRESHOLD,
                    },
                    "candidates": [asdict(result) for result in results],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        raise RuntimeError(
            "No Olive candidate met the accuracy gate after all retry search rounds: "
            f"all 128 dot products must be in [{LOWER_DOT_THRESHOLD}, {UPPER_DOT_THRESHOLD}]. "
            f"See report: {report_path}"
        )

    best = min(accepted, key=lambda result: result.latency_ms)
    copy_onnx_with_external_data(Path(best.model_path), args.output_model.resolve())

    report_path = work_dir / "olive_user_tower_quant_report.json"
    report_path.write_text(
        json.dumps(
            {
                "selected_model": asdict(best),
                "output_model_path": str(args.output_model.resolve()),
                "constraint": {
                    "dot_product_min": LOWER_DOT_THRESHOLD,
                    "dot_product_max": UPPER_DOT_THRESHOLD,
                },
                "candidates": [asdict(result) for result in results],
                "threading": {
                    "intra_op_num_threads": args.intra_threads,
                    "inter_op_num_threads": args.inter_threads,
                },
                "search_options": {
                    "quant_search_space": args.quant_search_space,
                    "enable_transformer_optimization": args.enable_transformer_optimization,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Selected model copied to: {args.output_model.resolve()}")
    print(f"Selection report: {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search ONNX dynamic INT8 quantization configs for user tower with Olive, "
            "then select the lowest-latency model that satisfies dot-product accuracy constraints."
        )
    )
    parser.add_argument(
        "--input-model",
        type=Path,
        default=settings.artifacts_dir / "user_tower.onnx",
        help="Input FP32 ONNX user tower model.",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=settings.artifacts_dir / "user_tower.quant.onnx",
        help="Final selected quantized ONNX model path.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=settings.artifacts_dir / "olive_user_tower_search",
        help="Directory for Olive outputs and reports.",
    )
    parser.add_argument(
        "--num-users",
        type=int,
        default=128,
        help="Number of sampled users for accuracy-gate evaluation.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=settings.max_seq_len,
        help="Sequence length used for evaluation input.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=12,
        help="Maximum Olive search iterations.",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="sequential",
        choices=["sequential", "random", "tpe"],
        help="Olive search sampler.",
    )
    parser.add_argument(
        "--quant-search-space",
        type=str,
        default="extended",
        choices=["basic", "extended"],
        help="Quantization search-space size. 'extended' adds more dynamic quantization knobs.",
    )
    parser.add_argument(
        "--enable-transformer-optimization",
        action="store_true",
        help="Enable Olive OrtTransformersOptimization pass before quantization.",
    )
    parser.add_argument(
        "--search-seed",
        type=int,
        default=42,
        help="Search seed.",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=42,
        help="Sampling seed for 128-user accuracy set.",
    )
    parser.add_argument(
        "--latency-warmup",
        type=int,
        default=20,
        help="Warmup runs for final latency benchmark.",
    )
    parser.add_argument(
        "--latency-repeats",
        type=int,
        default=200,
        help="Measured runs for final latency benchmark.",
    )
    parser.add_argument(
        "--intra-threads",
        type=int,
        default=1,
        help="ONNX Runtime intra-op threads (CPU).",
    )
    parser.add_argument(
        "--inter-threads",
        type=int,
        default=1,
        help="ONNX Runtime inter-op threads (CPU).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_search(parse_args())
    # pip install olive-ai[cpu]
    # Must be python <= 3.13 for olive compatibility
    # python .\scripts\olive_search_user_tower_quant.py --max-iter 20 --sampler tpe

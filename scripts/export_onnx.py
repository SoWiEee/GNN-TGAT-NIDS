"""Export trained static GNN models to ONNX format with optional quantization.

Usage:
    uv run python scripts/export_onnx.py --model graphsage
    uv run python scripts/export_onnx.py --model gat --quantize
    uv run python scripts/export_onnx.py --model graphsage --output exports/graphsage.onnx

Exports static models (GraphSAGE, GAT) only — temporal models (TGAT, TGN)
have stateful memory that cannot be captured in a single ONNX graph.

The exported model takes the same inputs as the PyG Data object:
    x           : (num_nodes, n_node_features)
    edge_index  : (2, num_edges)
    edge_attr   : (num_edges, n_edge_features)

Output:
    logits      : (num_edges, num_classes)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINTS_DIR = Path("checkpoints")
EXPORT_DIR = Path("exports")
STATIC_DIR = Path("data/processed/static")


class StaticModelWrapper(torch.nn.Module):
    """Wrapper that accepts raw tensors instead of PyG Data objects."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        from torch_geometric.data import Data
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return self.model(data)


def export_model(model_name: str, output_path: Path, quantize: bool = False) -> None:
    ckpt_path = CHECKPOINTS_DIR / f"{model_name}_best.pt"
    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        raise SystemExit(1)

    model = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.eval()
    logger.info("Loaded %s from %s", model_name, ckpt_path)

    import json
    meta_path = STATIC_DIR / "meta.json"
    if not meta_path.exists():
        logger.error("meta.json not found at %s — run static_builder.py first", meta_path)
        raise SystemExit(1)

    with open(meta_path) as f:
        meta = json.load(f)

    n_features = meta["n_features"]

    num_nodes = 10
    num_edges = 20
    dummy_x = torch.randn(num_nodes, n_features)
    dummy_edge_index = torch.randint(0, num_nodes, (2, num_edges))
    dummy_edge_attr = torch.randn(num_edges, n_features)

    wrapper = StaticModelWrapper(model)
    wrapper.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting to ONNX: %s", output_path)
    torch.onnx.export(
        wrapper,
        (dummy_x, dummy_edge_index, dummy_edge_attr),
        str(output_path),
        input_names=["x", "edge_index", "edge_attr"],
        output_names=["logits"],
        dynamic_axes={
            "x": {0: "num_nodes"},
            "edge_index": {1: "num_edges"},
            "edge_attr": {0: "num_edges"},
            "logits": {0: "num_edges"},
        },
        opset_version=17,
    )
    size_kb = output_path.stat().st_size / 1024
    logger.info("ONNX export complete: %s (%.1f KB)", output_path, size_kb)

    if quantize:
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            quant_path = output_path.with_suffix(".quant.onnx")
            quantize_dynamic(
                str(output_path),
                str(quant_path),
                weight_type=QuantType.QUInt8,
            )
            orig_size = output_path.stat().st_size / 1024
            quant_size = quant_path.stat().st_size / 1024
            logger.info(
                "Quantized: %s (%.1f KB → %.1f KB, %.0f%% reduction)",
                quant_path, orig_size, quant_size,
                (1 - quant_size / orig_size) * 100,
            )
        except ImportError:
            logger.warning("onnxruntime not installed — skipping quantization. "
                           "Install with: uv run pip install onnxruntime")

    # Validate
    try:
        import onnx
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model validation passed")
    except ImportError:
        logger.info("onnx package not installed — skipping validation")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export GNN models to ONNX")
    parser.add_argument("--model", required=True, choices=["graphsage", "gat"],
                        help="Model to export (static models only)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: exports/{model}.onnx)")
    parser.add_argument("--quantize", action="store_true",
                        help="Apply dynamic uint8 quantization after export")
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else EXPORT_DIR / f"{args.model}.onnx"
    export_model(args.model, output_path, quantize=args.quantize)


if __name__ == "__main__":
    main()

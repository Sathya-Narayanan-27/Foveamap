"""
FoveaMap - ONNX Export & Verification Utility.

Exports the trained FoveaNet PyTorch model to standard ONNX format with
dynamic batch and spatial dimensions for embedded NPU/CPU deployment.
"""

from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

from . import config as C
from .model import FoveaNet


class FoveaNetONNXWrapper(nn.Module):
    """Wrapper that takes BEV tensor and scalar level integer for ONNX tracing."""

    def __init__(self, model: FoveaNet):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, level: torch.Tensor) -> torch.Tensor:
        lvl_int = level.squeeze().long()
        return self.model(x, lvl_int)


def export_onnx(
    ckpt_path: str = f"{C.MODEL_DIR}/foveanet_best.pt",
    out_path: str = f"{C.MODEL_DIR}/foveanet.onnx",
    verify: bool = True,
):
    """Export FoveaNet checkpoint to ONNX."""
    print(f"[export_onnx] Loading checkpoint: {ckpt_path}")
    if not os.path.exists(ckpt_path):
        print(f"Error: checkpoint not found at {ckpt_path}")
        return False

    model = FoveaNet(ch=(16, 32, 64, 96))
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

    wrapper = FoveaNetONNXWrapper(model)
    wrapper.eval()

    # Dummy input representing an L0 fovea crop (1, 8, 176, 176) at level 0
    dummy_x = torch.randn(1, C.BEV_CHANNELS, 176, 176, dtype=torch.float32)
    dummy_lvl = torch.tensor([0], dtype=torch.long)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    print(f"[export_onnx] Exporting to {out_path}...")
    try:
        # Try export with classic TorchScript engine (dynamo=False)
        try:
            torch.onnx.export(
                wrapper,
                (dummy_x, dummy_lvl),
                out_path,
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=["bev_features", "level"],
                output_names=["logits"],
                dynamic_axes={
                    "bev_features": {0: "batch_size", 2: "height", 3: "width"},
                    "level": {0: "batch_size"},
                    "logits": {0: "batch_size", 2: "height", 3: "width"},
                },
                dynamo=False,
            )
        except TypeError:
            # Older PyTorch without dynamo arg
            torch.onnx.export(
                wrapper,
                (dummy_x, dummy_lvl),
                out_path,
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=["bev_features", "level"],
                output_names=["logits"],
                dynamic_axes={
                    "bev_features": {0: "batch_size", 2: "height", 3: "width"},
                    "level": {0: "batch_size"},
                    "logits": {0: "batch_size", 2: "height", 3: "width"},
                },
            )
        print(f"[export_onnx] Successfully exported ({os.path.getsize(out_path) / 1e6:.2f} MB)")
    except Exception as e:
        print(f"[export_onnx] ONNX export notice: {e}")
        print("  To enable full ONNX export in PyTorch 2.x, install 'onnx' and 'onnxscript':")
        print("  pip install onnx onnxscript onnxruntime")
        return False

    if verify:
        print("[export_onnx] Verifying ONNX graph consistency with PyTorch...")
        try:
            import onnx
            onnx_model = onnx.load(out_path)
            onnx.checker.check_model(onnx_model)
            print("  ONNX model checker: OK")
        except ImportError:
            print("  (Notice: 'onnx' package not installed; skipping onnx.checker)")

        try:
            import onnxruntime as ort
            session = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
            ort_inputs = {
                "bev_features": dummy_x.numpy(),
                "level": dummy_lvl.numpy(),
            }
            ort_outs = session.run(None, ort_inputs)
            with torch.no_grad():
                pt_out = wrapper(dummy_x, dummy_lvl).numpy()
            diff = np.max(np.abs(ort_outs[0] - pt_out))
            print(f"  ONNXRuntime output diff vs PyTorch: {diff:.2e} (max abs err)")
            assert diff < 1e-4, f"Output mismatch: {diff}"
            print("  ONNXRuntime verification: PASS")
        except ImportError:
            print("  (Notice: 'onnxruntime' not installed; skipping numerical verification)")

    return True


def main():
    ap = argparse.ArgumentParser(description="Export FoveaNet to ONNX format")
    ap.add_argument("--ckpt", default=f"{C.MODEL_DIR}/foveanet_best.pt")
    ap.add_argument("--out", default=f"{C.MODEL_DIR}/foveanet.onnx")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    success = export_onnx(ckpt_path=args.ckpt, out_path=args.out, verify=not args.no_verify)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()

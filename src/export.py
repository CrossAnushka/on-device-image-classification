"""Export trained PyTorch models to on-device formats: ONNX (universal) and TFLite
(Android-native, via Google's ai-edge-torch converter)."""
from __future__ import annotations

import os

import torch


def export_onnx(model, path: str, input_shape=(1, 3, 32, 32), opset: int = 17) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    model = model.eval().cpu()
    dummy = torch.randn(*input_shape)
    kwargs = dict(
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
    )
    # torch>=2.10 defaults to the dynamo exporter (needs onnxscript). Use the stable
    # TorchScript exporter, which is well-tested for CNNs and dependency-light.
    try:
        torch.onnx.export(model, dummy, path, dynamo=False, **kwargs)
    except TypeError:  # older torch without the `dynamo` kwarg
        torch.onnx.export(model, dummy, path, **kwargs)
    print(f"[export] ONNX written to {path}")
    return path


def export_tflite(model, path: str, input_shape=(1, 3, 32, 32), int8: bool = False, calib_loader=None) -> str | None:
    """Convert to .tflite via ai-edge-torch. Returns None (with a hint) if unavailable."""
    try:
        import ai_edge_torch
    except ImportError:
        print("[export] ai-edge-torch not installed — skipping PyTorch->TFLite. "
              "Install with `pip install ai-edge-torch` (needs a supported torch version), "
              "or use the TensorFlow-native variant in tf/ for .tflite artifacts.")
        return None

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    model = model.eval().cpu()
    sample = (torch.randn(*input_shape),)

    quant_config = None
    if int8:
        try:
            from ai_edge_torch.quantize import pt2e_quantizer, quant_config as qc
            from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
            from torch.export import export_for_training

            quantizer = pt2e_quantizer.PT2EQuantizer().set_global(
                pt2e_quantizer.get_symmetric_quantization_config(is_per_channel=True)
            )
            exported = export_for_training(model, sample).module()
            prepared = prepare_pt2e(exported, quantizer)
            if calib_loader is not None:
                with torch.no_grad():
                    for x, _ in calib_loader:
                        prepared(x.cpu())
            model = convert_pt2e(prepared)
            quant_config = qc.QuantConfig(pt2e_quantizer=quantizer)
        except Exception as e:  # noqa: BLE001
            print(f"[export] INT8 TFLite path unavailable ({type(e).__name__}); exporting float TFLite.")

    edge_model = ai_edge_torch.convert(model, sample, quant_config=quant_config)
    edge_model.export(path)
    print(f"[export] TFLite written to {path}")
    return path

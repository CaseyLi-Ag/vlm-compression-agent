"""Apply Olive's block-wise INT4 RTN pass to a captured multimodal decoder."""

import argparse
import json
from pathlib import Path
from time import monotonic

import onnx
from olive.model import ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.rtn_quantization import OnnxBlockWiseRtnQuantization

from vlm_compression_agent.controller import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args()
    model_dir = args.model_dir.resolve()
    config_path = model_dir / "genai_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    decoder = config["model"]["decoder"]
    source = model_dir / decoder["filename"]
    destination = source.with_name("rtn-int4.onnx")
    if destination.is_file():
        raise FileExistsError(f"An INT4 decoder already exists: {destination}")
    start = monotonic()
    algorithm = create_pass_from_dict(OnnxBlockWiseRtnQuantization, {
        "bits": 4, "block_size": 128, "is_symmetric": True, "save_as_external_data": True,
    }, disable_search=True)
    result = algorithm.run(ONNXModelHandler(model_path=str(source)), str(destination))
    graph = onnx.load(result.model_path, load_external_data=False)
    packed = sum(node.op_type == "MatMulNBits" for node in graph.graph.node)
    if not packed:
        raise RuntimeError("Olive's output has no packed INT4 MatMulNBits nodes")
    output_names = {value.name for value in graph.output}
    logits = decoder["outputs"]["logits"]
    if logits not in output_names:
        if f"{logits}_Q4" not in output_names:
            raise RuntimeError("Quantized decoder is missing its logits output")
        decoder["outputs"]["logits"] = f"{logits}_Q4"
    print(f"Olive RTN packed {packed} decoder nodes in {monotonic() - start:.1f} s", flush=True)
    decoder["filename"] = destination.relative_to(model_dir).as_posix()
    write_json(model_dir / "genai_config.fp16.json", json.loads(config_path.read_text(encoding="utf-8")))
    write_json(config_path, config)
    write_json(model_dir / "rtn-int4.json", {
        "source_decoder": str(source),
        "quantized_decoder": str(destination),
        "algorithm": "Olive OnnxBlockWiseRtnQuantization",
        "bits": 4,
        "block_size": 128,
        "packed_matmul_nodes": packed,
        "seconds": monotonic() - start,
    })


if __name__ == "__main__":
    main()

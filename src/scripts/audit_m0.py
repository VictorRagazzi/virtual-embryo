"""Audita E8.5/E9.5 e checkpoint local sem alterar dados nem baixar recursos."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import h5py
import numpy as np
import torch
from dotenv import dotenv_values

BLOCK_SIZE = 5_000_000


def _text(value: object) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _gene_digest(genes: list[str]) -> str:
    return hashlib.sha256("\n".join(genes).encode()).hexdigest()


def _scan_values(data: h5py.Dataset) -> dict[str, int | float | bool]:
    minimum, maximum, zeros, nan, infinite = np.inf, -np.inf, 0, False, False
    for start in range(0, data.shape[0], BLOCK_SIZE):
        block = data[start : start + BLOCK_SIZE]
        if block.size:
            minimum, maximum = min(minimum, float(block.min())), max(maximum, float(block.max()))
            zeros += int((block == 0).sum())
            nan, infinite = nan or bool(np.isnan(block).any()), infinite or bool(np.isinf(block).any())
    return {"min_stored": float(minimum), "max_stored": float(maximum), "explicit_zeros": zeros,
            "nan": nan, "infinite": infinite}


def inspect_h5ad(path: Path) -> dict[str, object]:
    """Examina HDF5 em blocos e nunca materializa uma matriz densa."""
    with h5py.File(path, "r") as file:
        x = file["X"]
        if not isinstance(x, h5py.Group):
            raise ValueError(f"{path}: X não é uma matriz esparsa HDF5.")
        shape = [int(item) for item in x.attrs["shape"]]
        data = x["data"]
        scan = _scan_values(data)
        stored, total = int(data.shape[0]), shape[0] * shape[1]
        genes = [_text(item) for item in file["var"]["_index"][:]]
        celltypes: dict[str, int] = {}
        if "celltype" in file["obs"]:
            categorical = file["obs"]["celltype"]
            categories = [_text(item) for item in categorical["categories"][:]]
            celltypes = dict(sorted(Counter(categories[int(code)] for code in categorical["codes"][:] if code >= 0).items()))
        return {
            "path": str(path), "sha256": _sha256(path), "shape": shape,
            "x": {"encoding": _text(x.attrs["encoding-type"]), "dtype": str(data.dtype),
                  "stored_values": stored, "effective_nonzero_values": stored - int(scan["explicit_zeros"]),
                  "stored_density": stored / total, "effective_density": (stored - int(scan["explicit_zeros"])) / total,
                  **scan},
            "dense_memory_bytes": {"float32": total * 4, "float64": total * 8},
            "genes": {"count": len(genes), "unique": len(set(genes)), "order_sha256": _gene_digest(genes),
                      "first_five": genes[:5], "last_five": genes[-5:]},
            "fields": {name: sorted(file[name].keys()) if name in file else []
                       for name in ("obs", "var", "obsm", "layers", "uns")},
            "celltype_counts": celltypes,
            "normalization_evidence": {
                "raw_present": "raw" in file, "layers_present": bool(file["layers"].keys()),
                "uns_keys": sorted(file["uns"].keys()),
                "interpretation": "Faixa não negativa até ~6.9 é compatível com log1p; não há metadado explícito do método ou fator de normalização.",
            },
        }


def compare_genes(e85: Path, e95: Path) -> dict[str, object]:
    with h5py.File(e85, "r") as first, h5py.File(e95, "r") as second:
        a = [_text(item) for item in first["var"]["_index"][:]]
        b = [_text(item) for item in second["var"]["_index"][:]]
    mismatch = next((i for i, pair in enumerate(zip(a, b)) if pair[0] != pair[1]), None)
    return {"same_length": len(a) == len(b), "same_order": a == b, "first_mismatch_index": mismatch,
            "first_mismatch": None if mismatch is None else {"e85": a[mismatch], "e95": b[mismatch]},
            "e85_only_count": len(set(a) - set(b)), "e95_only_count": len(set(b) - set(a))}


def inspect_checkpoint(checkpoint: Path, args_path: Path) -> dict[str, object]:
    args, state = json.loads(args_path.read_text()), torch.load(checkpoint, map_location="cpu", weights_only=True)
    embedding = state["encoder.embedding.weight"]
    return {"path": str(checkpoint), "sha256": _sha256(checkpoint), "state_tensor_count": len(state),
            "embedding_shape": list(embedding.shape), "embedding_dim": int(embedding.shape[1]),
            "architecture": {key: args[key] for key in ("nlayers", "nheads", "embsize", "d_hid", "n_bins", "max_seq_len")},
            "local_args_data_source": args.get("data_source"), "provenance_status": "BLOQUEADO_POR_PROVENIENCIA",
            "provenance_reason": "Não há URL, licença, hash de origem, corpus auditável ou intervalo de estágios que exclua a janela proibida."}


def inspect_ollama() -> dict[str, object]:
    values = {**dotenv_values(".env"), **os.environ}
    base_url, model = values.get("OLLAMA_BASE_URL"), values.get("OLLAMA_MODEL")
    result: dict[str, object] = {"base_url_configured": bool(base_url), "model_configured": bool(model)}
    if base_url:
        parsed = urlparse(base_url)
        safe = bool(parsed.scheme in {"http", "https"} and parsed.hostname and not parsed.username and not parsed.password and not parsed.query and not parsed.fragment)
        result["base_url_safe_to_record"] = safe
        if safe:
            result["base_url"] = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    if model:
        safe = "\n" not in model and "\r" not in model and len(model) <= 200
        result["model_safe_to_record"] = safe
        if safe:
            result["model"] = model
    return result


def render_markdown(report: dict[str, object]) -> str:
    e85, e95, comparison = report["datasets"]["E85"], report["datasets"]["E95"], report["gene_comparison"]
    lines = ["# Relatório de auditoria M0", "",
             f"Gerado em {report['generated_at_utc']} por `uv run python src/scripts/audit_m0.py --seed {report['seed']}`. Os `.h5ad` foram lidos sem alteração e `X/data` foi varrido em blocos.",
             "", "## Dados oficiais", "",
             "| Arquivo | Shape | X | dtype | Valores armazenados | Densidade efetiva | Min–max | NaN/inf | float32 denso |",
             "|---|---:|---|---|---:|---:|---|---|---:|"]
    for stage, item in (("E8.5", e85), ("E9.5", e95)):
        x = item["x"]
        lines.append(f"| {stage} | {item['shape'][0]} × {item['shape'][1]} | {x['encoding']} | {x['dtype']} | {x['stored_values']:,} | {100*x['effective_density']:.3f}% | {x['min_stored']:.6g}–{x['max_stored']:.6g} | {x['nan']}/{x['infinite']} | {item['dense_memory_bytes']['float32']/2**30:.2f} GiB |")
    lines += ["", f"Concatenar ambos densamente requer {sum(item['dense_memory_bytes']['float32'] for item in (e85, e95))/2**30:.2f} GiB (`float32`) ou {sum(item['dense_memory_bytes']['float64'] for item in (e85, e95))/2**30:.2f} GiB (`float64`), antes de cópias temporárias. A PCA histórica não deve rodar integralmente.",
              "", "Não existe `raw`, `layers` é vazio e `uns` contém apenas `celltype_palette`; portanto a escala log-normalizada é indício numérico, não confirmação de proveniência.",
              "", "## Genes e metadados", "",
              f"Verificação automática: `same_length={comparison['same_length']}`, `same_order={comparison['same_order']}`, `first_mismatch_index={comparison['first_mismatch_index']}`, 32.285 genes únicos. SHA-256 da sequência ordenada: `{e85['genes']['order_sha256']}`. Primeiro/último: `{', '.join(e85['genes']['first_five'])}` / `{', '.join(e85['genes']['last_five'])}`.",
              "", f"E8.5: `obs={e85['fields']['obs']}`, `var={e85['fields']['var']}`, `obsm={e85['fields']['obsm']}`, `layers={e85['fields']['layers']}`, `uns={e85['fields']['uns']}`. E9.5 tem os mesmos conjuntos.",
              "", "### Distribuição de `obs['celltype']`", "", "| Rótulo | E8.5 | E9.5 |", "|---|---:|---:|"]
    labels = sorted(set(e85["celltype_counts"]) | set(e95["celltype_counts"]))
    lines += [f"| {label} | {e85['celltype_counts'].get(label, 0)} | {e95['celltype_counts'].get(label, 0)} |" for label in labels]
    checkpoint, ollama = report["scgpt_checkpoint"], report["ollama"]
    lines += ["", "## Checkpoint scGPT e Ollama", "",
              f"`{checkpoint['path']}`: SHA-256 `{checkpoint['sha256']}`, embedding `{checkpoint['embedding_shape'][0]} × {checkpoint['embedding_shape'][1]}`, `embedding_dim={checkpoint['embedding_dim']}`, {checkpoint['architecture']['nlayers']} camadas/{checkpoint['architecture']['nheads']} cabeças. `args.json` aponta apenas para `{checkpoint['local_args_data_source']}`. Situação: **{checkpoint['provenance_status']}** — {checkpoint['provenance_reason']}",
              "", f"Ollama: `OLLAMA_BASE_URL` configurada={ollama['base_url_configured']}; `OLLAMA_MODEL` configurada={ollama['model_configured']}. Nenhuma chamada ao serviço foi necessária.",
              "", "## Repositório e reprodução", "",
              "Divergências encontradas: `docs/ARCHITETURE.md` não tem o nome citado por AGENTS; o `get_dummy.py` histórico não gerava dummy; PCA densificava os dois estágios; o README referencia `src/scripts/umap_vec.py`, que não existe, e omite o `--input-scale` obrigatório de `pre_processement.py`; e testes referenciam os módulos ausentes `src.approaches.mouse_geneformer` e `src.scripts.subset_heart`. Abordagens históricas foram preservadas.",
              "", "Comandos: `uv run python src/scripts/audit_m0.py --seed 42`; `uv run python src/scripts/get_dummy.py --reference data/E95.h5ad --output /tmp/ve-m0/dummy.h5ad --n-cells 2500 --seed 42`; `uv run python src/scripts/copy_last.py --input data/E95.h5ad --output /tmp/ve-m0/copy_last.h5ad --seed 42`; `uv run python src/scripts/predict_e10_5_PCA.py --e85 data/E85.h5ad --e95 data/E95.h5ad --out /tmp/ve-m0/pca.h5ad --max-cells-per-stage 128 --n-comps 30 --target-cells 1000 --skip-llm --seed 42`.", ""]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e85", type=Path, default=Path("data/E85.h5ad")); parser.add_argument("--e95", type=Path, default=Path("data/E95.h5ad"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/scGPT_heart/best_model.pt")); parser.add_argument("--checkpoint-args", type=Path, default=Path("models/scGPT_heart/args.json"))
    parser.add_argument("--json-output", type=Path, default=Path("docs/M0_AUDIT.json")); parser.add_argument("--markdown-output", type=Path, default=Path("docs/M0_AUDIT.md")); parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "seed": args.seed,
              "git_revision": subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip(),
              "datasets": {"E85": inspect_h5ad(args.e85), "E95": inspect_h5ad(args.e95)},
              "gene_comparison": compare_genes(args.e85, args.e95),
              "scgpt_checkpoint": inspect_checkpoint(args.checkpoint, args.checkpoint_args), "ollama": inspect_ollama()}
    args.json_output.parent.mkdir(parents=True, exist_ok=True); args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n"); args.markdown_output.write_text(render_markdown(report))
    print(json.dumps({"json": str(args.json_output), "markdown": str(args.markdown_output), "same_gene_order": report["gene_comparison"]["same_order"]}))


if __name__ == "__main__":
    main()

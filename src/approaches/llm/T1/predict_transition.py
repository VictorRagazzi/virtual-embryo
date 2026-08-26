"""
Estagio 4-5 do pipeline LLM (Virtual Embryo Challenge - Task 1).

Le os "cluster cards" gerados por cluster_cards.py (celltype x E8.5/E9.5)
e pede ao LLM um plano de transicao para E10.5, em nivel semantico:

  - proporcao alvo de cada tipo celular em E10.5
  - "trend_factor": o quanto extrapolar o delta E8.5->E9.5 ja observado
    (1.0 = continua no mesmo ritmo, 0.0 = estabiliza/plateau,
     negativo = reverte a tendencia, >1 = acelera)
  - possiveis linhagens NOVAS (nunca vistas em E8.5/E9.5) que a biologia
    do desenvolvimento cardiaco de camundongo sugere que podem emergir
    entre E9.5 e E10.5, com marcadores propostos e o progenitor de origem

O LLM NAO gera nenhum vetor de expressao bruto -- so essas decisoes de
alto nivel. A materializacao das celulas (Estagio 6) e' feita depois por
codigo determinístico, usando esse plano.

Uso:
    uv run python predict_transition.py \
        --cards data/cluster_cards.json \
        --out data/transition_plan.json
"""

import argparse
import json
import os
import sys
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationError

SYSTEM_PROMPT = """\
You are a specialist in mouse embryonic cardiac developmental biology. You \
receive a summary (not raw data) of the cellular composition of the heart \
at two observed stages, E8.5 and E9.5: the proportion of each cell type and \
the main marker genes for each type at E9.5, along with the mean expression \
of each marker at E8.5 and E9.5.

Your task is to extrapolate this trajectory to stage E10.5 (the next \
observable stage, not yet observed). You MUST NOT generate raw gene \
expression values -- only high-level decisions about population \
proportions, trend direction/intensity, and possible new lineages emerging \
through differentiation, using your knowledge of cardiac developmental \
biology (chambers, outflow tract, epicardium, conduction system, etc.).

Respond ONLY with valid JSON in the requested format, with no text before \
or after it."""


OUTPUT_SCHEMA_HINT = """\
Output format (strict JSON):
{
  "predicted_celltypes": [
    {
      "celltype": "<exact cell type name, matching the input card, if origin != novel_lineage>",
      "origin": "persistent" | "continuing_new" | "novel_lineage",
      "parent_celltype": "<name of an input cell type, required if novel_lineage, otherwise null>",
      "target_proportion": <float 0-1, expected proportion of the total population at E10.5>,
      "trend_factor": <float, multiplies the already observed E8.5->E9.5 delta to estimate the E9.5->E10.5 delta; use 0 for novel_lineage types>,
      "key_markers": [ {"gene": "<name>", "direction": "up"|"down", "confidence": "high"|"medium"|"low"} ]
        (required and non-empty if origin == "novel_lineage"; optional/may be empty in other cases)
    }
  ],
  "notes": "<1-2 sentences justifying the most important decisions>"
}

Rules:
- Cover ALL "persistent" and "new_in_e9.5" types from the input (extrapolate them even if trend_factor=0).
- Types "lost_after_e8.5" should normally NOT reappear -- only include them if there is a strong biological reason, and explain it in "notes".
- The target_proportion values across all entries should sum to approximately 1.0.
- New lineages (novel_lineage) should only be proposed when there is a clear biological justification (e.g., expected differentiation of an existing progenitor between E9.5 and E10.5); do not invent lineages without biological justification.
"""


class KeyMarker(BaseModel):
    gene: str
    direction: Literal["up", "down"]
    confidence: Literal["high", "medium", "low"]


class PredictedCelltype(BaseModel):
    celltype: str
    origin: Literal["persistent", "continuing_new", "novel_lineage"]
    parent_celltype: Optional[str] = None
    target_proportion: float = Field(ge=0, le=1)
    trend_factor: float
    key_markers: list[KeyMarker] = Field(default_factory=list)


class TransitionPlan(BaseModel):
    predicted_celltypes: list[PredictedCelltype]
    notes: str

def build_user_prompt(cards: list[dict]) -> str:
    lines = ["Dados observados (E8.5 -> E9.5):\n"]
    for c in cards:
        header = (f"## {c['celltype']} [{c['status']}] | "
                  f"prop E8.5={c['proportion_e8.5']*100:.2f}% -> E9.5={c['proportion_e9.5']*100:.2f}% | "
                  f"n={c['n_cells_e8.5']}->{c['n_cells_e9.5']}")
        lines.append(header)
        if c["top_markers"]:
            lines.append("Top marcadores (gene: media_e8.5 -> media_e9.5, delta, log2fc_vs_resto_e9.5):")
            for m in c["top_markers"]:
                m85 = "NA" if m["mean_e8.5"] is None else f"{m['mean_e8.5']:.3f}"
                delta = "NA" if m["delta_e8.5_to_e9.5"] is None else f"{m['delta_e8.5_to_e9.5']:+.3f}"
                lines.append(f"- {m['gene']}: {m85} -> {m['mean_e9.5']:.3f} "
                             f"(delta={delta}), log2fc={m['log2fc_vs_rest_e9.5']:.2f}")
        lines.append("")
    lines.append(OUTPUT_SCHEMA_HINT)
    return "\n".join(lines)


def call_llm(user_prompt: str, model: str, api_key: str, base_url: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content


def parse_and_validate(raw: str, valid_celltypes: set[str], max_attempts_left: int) -> TransitionPlan:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text[4:] if text.startswith("json") else text
    plan = TransitionPlan.model_validate_json(text)

    total = sum(p.target_proportion for p in plan.predicted_celltypes)
    if not (0.85 <= total <= 1.15):
        raise ValueError(f"target_proportion soma {total:.3f}, esperado ~1.0")

    for p in plan.predicted_celltypes:
        if p.origin == "novel_lineage":
            if not p.parent_celltype or p.parent_celltype not in valid_celltypes:
                raise ValueError(f"'{p.celltype}' e' novel_lineage mas parent_celltype invalido/ausente")
            if not p.key_markers:
                raise ValueError(f"'{p.celltype}' e' novel_lineage mas nao tem key_markers")
        elif p.celltype not in valid_celltypes:
            raise ValueError(f"'{p.celltype}' (origin={p.origin}) nao existe no input -- provavel alucinacao")

    return plan


def main():
    from dotenv import load_dotenv

    load_dotenv()

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cards", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL"))
    ap.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY"))
    ap.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    ap.add_argument("--max-attempts", type=int, default=3)
    args = ap.parse_args()

    with open(args.cards) as f:
        cards = json.load(f)
    valid_celltypes = {c["celltype"] for c in cards}
    user_prompt = build_user_prompt(cards)

    if not args.model or not args.api_key:
        sys.exit("Defina LLM_MODEL e OPENROUTER_API_KEY (.env) ou passe --model/--api-key")

    last_error = None
    prompt = user_prompt
    for attempt in range(1, args.max_attempts + 1):
        raw = call_llm(prompt, args.model, args.api_key, args.base_url)
        try:
            plan = parse_and_validate(raw, valid_celltypes, args.max_attempts - attempt)
            break
        except (ValidationError, ValueError, json.JSONDecodeError) as e:
            last_error = e
            print(f"[tentativa {attempt}] saida invalida: {e}", file=sys.stderr)
            prompt = (user_prompt +
                      f"\n\nSua resposta anterior teve este erro de validacao: {e}\n"
                      "Corrija e responda de novo SOMENTE com o JSON.")
    else:
        sys.exit(f"Falhou apos {args.max_attempts} tentativas. Ultimo erro: {last_error}")

    with open(args.out, "w") as f:
        json.dump(plan.model_dump(), f, indent=2, ensure_ascii=False)

    n_novel = sum(p.origin == "novel_lineage" for p in plan.predicted_celltypes)
    print(f"Plano salvo em {args.out}: {len(plan.predicted_celltypes)} tipos previstos "
          f"({n_novel} linhagens novas propostas)")


if __name__ == "__main__":
    main()
"""Baixa os recursos oficiais do Mouse-Geneformer usados pelo pipeline T2.

Execute da raiz: python3 script/download_model.py
Usa somente a biblioteca padrão do Python.
"""

import argparse
import csv
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import pickle
import shutil
import tempfile
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile


DRIVE_URL = "https://drive.usercontent.google.com/download"
MODEL_ID = "1gM3gcc3DlNGt5bAcqHbeRxtdMktGeDEg"
MODEL_URL = f"{DRIVE_URL}?id={MODEL_ID}&export=download"
MODEL_SHA256 = "183731eb1784798d38323a34c125b9bb761c1332d00a9faf281d88a8a083b2ad"
DICTIONARY_URL = "https://huggingface.co/datasets/MPRG/Mouse-Genecorpus-20M/resolve/main/"
GENE_MAP_PICKLE = "MLM-re_token_dictionary_v1_GeneSymbol_to_EnsemblID.pkl"
# Versões verificadas no experimento T2; mudanças na origem exigem revisão.
HASHES = {
    "config.json": "fb13ea56d59002fd84f899c39c4b824234e79b4dd874e80b2d7a01449059658b",
    "pytorch_model.bin": "cd8f2c361f185e2e599750d6c44a598b9e9455d1d63fdad1c70799068fa8bb32",
    "MLM-re_token_dictionary_v1.pkl": "c105045e68e7f34d314099008d77d9e6af4e9a22e719c27f86204ed293267f19",
    "mouse_gene_median_dictionary.pkl": "39ad647c50bb26b152efb89bb9485d2d821e959ab1e2bbfd173be80c662e2390",
    GENE_MAP_PICKLE: "1607905dc20e7a4403169ccd092b79d3bea7f9eb5fa6c4eb34b2c21eeaec4393",
    "gene_map.csv": "ca0642aa19a49c22efdc248c54a822d1a4fa4e8cbff09992cd58b4b0f35cc7c1",
}


class DriveConfirmation(HTMLParser):
    """Lê os campos da confirmação de download de arquivos grandes do Drive."""

    def __init__(self):
        super().__init__()
        self.fields = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "input" and attrs.get("name") in {"uuid", "confirm"}:
            self.fields[attrs["name"]] = attrs.get("value", "")


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def existing_file(path, expected_hash):
    if not path.exists():
        return False
    if sha256(path) != expected_hash:
        raise ValueError(f"Arquivo existente difere da versão esperada: {path}. "
                         "Use outro --output-dir ou remova esse arquivo para baixar novamente.")
    print(f"Verificado: {path.name}", flush=True)
    return True


def download_file(url, destination, expected_hash):
    """Baixa em blocos e publica o arquivo somente após verificar SHA-256."""
    if existing_file(destination, expected_hash):
        return
    print(f"Baixando: {destination.name}", flush=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        partial = Path(temporary) / "download.part"
        for attempt in range(2):
            request = Request(url, headers={"User-Agent": "Mouse-Geneformer-T2-downloader"})
            with urlopen(request, timeout=60) as response:
                if "text/html" in response.headers.get("Content-Type", ""):
                    if not url.startswith(DRIVE_URL) or attempt:
                        raise ValueError(f"Servidor retornou HTML em vez do arquivo: {url}")
                    confirmation = DriveConfirmation()
                    confirmation.feed(response.read(1024 * 1024).decode("utf-8"))
                    if not confirmation.fields:
                        raise ValueError("Não foi possível obter a confirmação do Google Drive.")
                    fields = {"id": MODEL_ID, "export": "download", "confirm": "t"}
                    fields.update(confirmation.fields)
                    url = DRIVE_URL + "?" + urlencode(fields)
                    continue
                with partial.open("wb") as handle:
                    shutil.copyfileobj(response, handle, length=1024 * 1024)
            if sha256(partial) != expected_hash:
                raise ValueError(f"SHA-256 inesperado no download de {destination.name}.")
            partial.replace(destination)
            return


def download_model(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Verifique tudo antes de baixar, preservando arquivos locais divergentes.
    missing = [name for name, checksum in HASHES.items()
               if not existing_file(output_dir / name, checksum)]
    with tempfile.TemporaryDirectory(dir=output_dir) as temporary:
        temporary = Path(temporary)
        model_files = [name for name in ("config.json", "pytorch_model.bin") if name in missing]
        if model_files:
            archive_path = temporary / "mouse-Geneformer.zip"
            download_file(MODEL_URL, archive_path, MODEL_SHA256)
            with ZipFile(archive_path) as archive:
                # Extraia somente os dois arquivos usados pelo encoder.
                for name in model_files:
                    extracted = temporary / name
                    with archive.open(f"mouse-Geneformer/{name}") as source, extracted.open("wb") as target:
                        shutil.copyfileobj(source, target, length=1024 * 1024)
                    if sha256(extracted) != HASHES[name]:
                        raise ValueError(f"SHA-256 inesperado no arquivo extraído: {name}")
                    extracted.replace(output_dir / name)
        for name in missing:
            if name.endswith(".pkl"):
                download_file(DICTIONARY_URL + name, output_dir / name, HASHES[name])
        if "gene_map.csv" in missing:
            # Abra somente o pickle oficial cujo hash foi conferido acima.
            with (output_dir / GENE_MAP_PICKLE).open("rb") as handle:
                mapping = pickle.load(handle)
            csv_path = temporary / "gene_map.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(["gene_symbol", "ensembl_id"])
                writer.writerows(mapping.items())
            if sha256(csv_path) != HASHES["gene_map.csv"]:
                raise ValueError("O mapeamento gerado difere do utilizado pelo T2.")
            csv_path.replace(output_dir / "gene_map.csv")
    records = []
    for name, checksum in HASHES.items():
        source = DICTIONARY_URL + name if name.endswith(".pkl") else MODEL_URL
        if name == "gene_map.csv":
            source = f"{GENE_MAP_PICKLE} → CSV gene_symbol,ensembl_id"
        records.append({"file": name, "source": source, "sha256": checksum})
    metadata = output_dir / "resources.json"
    if not metadata.exists():
        metadata.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Mouse-Geneformer pronto em {output_dir.resolve()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("models/mouse-Geneformer"))
    args = parser.parse_args()
    try:
        download_model(args.output_dir)
    except (OSError, ValueError, BadZipFile) as error:
        parser.exit(1, f"Erro: {error}\n")


if __name__ == "__main__":
    main()

import argparse
import os
import warnings
import numpy as np
import anndata as ad

warnings.filterwarnings("ignore")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validação de score com split aleatório dos dados.")
    parser.add_argument("--h5ad", required=True, help="Caminho para o arquivo .h5ad de entrada (E95)")
    parser.add_argument("--reference", default="data/E85.h5ad", help="Caminho para o arquivo de referência (E85)")
    parser.add_argument("--seed", type=int, default=42, help="Semente aleatória para reprodutibilidade")
    args = parser.parse_args()

    print(f"Carregando {args.h5ad}...")
    adata = ad.read_h5ad(args.h5ad)
    print(f"  {adata.n_obs} células × {adata.n_vars} genes")

    np.random.seed(args.seed)
    indices = np.arange(adata.n_obs)
    np.random.shuffle(indices)

    half = adata.n_obs // 2
    adata_input = adata[indices[:half]].copy()
    adata_target = adata[indices[half:]].copy()

    print(f"Divisão realizada: {adata_input.n_obs} células (input) e {adata_target.n_obs} células (target)")

    input_path = "data/temp_e95_input.h5ad"
    target_path = "data/temp_e95_target.h5ad"

    try:
        from veckit import score

        print("Salvando partições temporárias...")
        adata_input.write_h5ad(input_path)
        adata_target.write_h5ad(target_path)

        print("Calculando o score...")
        result = score(
            task="T1",
            input=input_path,
            target=target_path,
            reference=args.reference
        )

        print("\nResultado do Score:")
        print(result)

    finally:
        for path in [input_path, target_path]:
            if os.path.exists(path):
                os.remove(path)


if __name__ == "__main__":
    main()

"""Task 1 (Temporal) - abordagem baseada em Optimal Transport + fine-tuning estilo scGPT.

Submódulos:
    data_utils        - carregamento e preparação (PCA conjunta) de E8.5/E9.5
    ot_pairing         - construção e solução do problema de OT (moscot), CLI principal
    pairing_sampler    - amostragem de pares (source, target) a partir da matriz de transporte
    build_dataset      - monta o AnnData "pareado" (X=E8.5, layers['target']=E9.5) e salva em disco
"""
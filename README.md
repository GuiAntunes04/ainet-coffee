# AINET Coffee

Projeto para a competicao Kaggle AINET Coffee:
https://www.kaggle.com/competitions/ainet-coffee/overview

O objetivo e classificar imagens de graos de cafe em cinco classes:

- `Verde`
- `Verde cana`
- `Cereja`
- `Passa`
- `Seco`

## Estrutura

- `Training set-kaggle/`: imagens de treino organizadas por classe.
- `test-kaggle/`: imagens usadas para avaliacao local e geracao da submissao.
- `extra-tests/`: imagens externas para testes locais adicionais.
- `experiments/`: pastas de experimentos, com `metadata.json`, `model.keras` local e submissao quando gerada.
- `config.py`: caminhos principais, classes, rotulos e configuracoes globais.
- `utils.py`: carregamento dos datasets, normalizacao e data augmentation.
- `experiment_utils.py`: criacao de pastas de experimento e salvamento de metadados.
- `train_cnn.py`: treino de CNN com parametros configuraveis por comando.
- `train_mlp.py`: treino de MLP com parametros configuraveis por comando.
- `train_mobilenet.py`: treino com MobileNetV2 pre-treinada.
- `evaluate.py`: avaliacao local de um modelo treinado.
- `predict.py`: geracao de `submission.csv` para o Kaggle.
- `predict_extra_tests.py`: predicao das imagens em `extra-tests/`.

## O Que Entra No Git

Entram no repositorio:

- codigo Python;
- `README.md`;
- `requirements.txt`;
- imagens de treino e teste;
- imagens de `extra-tests/`;
- `metadata.json` dos experimentos;
- `submission.csv` dos experimentos.

Ficam fora do Git:

- `.venv/`;
- `.vscode/`;
- `__pycache__/`;
- `.pytest_cache/`;
- arquivos `*.keras`, incluindo `experiments/*/model.keras`.

Os modelos treinados ficam no computador local. Para compartilhar um modelo especifico
com um colaborador, envie o arquivo `model.keras` por outro meio ou use uma solucao
propria para arquivos grandes.

## Ambiente

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Fluxo De Experimentos

Cada treino deve receber um nome unico:

```bash
python train_cnn.py --experiment-name cnn_exp001_baseline
```

O treino cria uma pasta como:

```text
experiments/cnn_exp001_baseline/
```

Dentro dela ficam:

- `model.keras`: melhor modelo salvo pelo `ModelCheckpoint`, ignorado pelo Git.
- `metadata.json`: parametros e metricas do experimento.
- `submission.csv`: arquivo gerado depois pelo `predict.py`, se voce rodar a predicao.

Os metadados e submissoes podem ser versionados para registrar o historico dos
experimentos sem subir os modelos treinados.

## Treinar CNN

Comando base:

```bash
python train_cnn.py --experiment-name cnn_exp001_baseline
```

Alterar numero de epocas:

```bash
python train_cnn.py --experiment-name cnn_exp_epochs60 --epochs 60
```

Alterar batch size:

```bash
python train_cnn.py --experiment-name cnn_exp_batch16 --batch-size 16
```

Alterar tamanho da imagem:

```bash
python train_cnn.py --experiment-name cnn_exp_img96 --image-size 96
```

Alterar learning rate:

```bash
python train_cnn.py --experiment-name cnn_exp_lr0003 --learning-rate 0.0003
```

Alterar dropout:

```bash
python train_cnn.py --experiment-name cnn_exp_dropout04 --dropout 0.4
```

Alterar filtros convolucionais:

```bash
python train_cnn.py --experiment-name cnn_exp_filters8_16_32 --filters 8,16,32
```

O parametro `--filters` tambem define a quantidade de blocos convolucionais. Por
exemplo, `8,16,32` cria tres blocos `Conv2D + MaxPooling2D`.

Alterar quantidade de neuronios da camada densa:

```bash
python train_cnn.py --experiment-name cnn_exp_dense64 --dense-units 64
```

Alterar regularizacao L2:

```bash
python train_cnn.py --experiment-name cnn_exp_l2_0001 --l2 0.0001
```

Alterar pooling final:

```bash
python train_cnn.py --experiment-name cnn_exp_gap --pooling gap
```

Valores aceitos:

- `flatten`
- `gap`

Alterar augmentation:

```bash
python train_cnn.py --experiment-name cnn_exp_aug_light --augmentation light
```

```bash
python train_cnn.py --experiment-name cnn_exp_aug_strong --augmentation strong
```

Valores aceitos:

- `none`
- `light`
- `medium`
- `strong`

Combinar varios parametros:

```bash
python train_cnn.py --experiment-name cnn_exp_custom --image-size 96 --batch-size 16 --learning-rate 0.0003 --dropout 0.4 --filters 16,32,64 --dense-units 32 --l2 0.0001 --pooling gap --augmentation medium
```

## Treinar MLP

Comando base:

```bash
python train_mlp.py --experiment-name mlp_exp001_baseline
```

Alterar numero de epocas:

```bash
python train_mlp.py --experiment-name mlp_exp_epochs60 --epochs 60
```

Alterar batch size:

```bash
python train_mlp.py --experiment-name mlp_exp_batch8 --batch-size 8
```

Alterar tamanho da imagem:

```bash
python train_mlp.py --experiment-name mlp_exp_img32 --image-size 32
```

Alterar learning rate:

```bash
python train_mlp.py --experiment-name mlp_exp_lr0003 --learning-rate 0.0003
```

Alterar dropout:

```bash
python train_mlp.py --experiment-name mlp_exp_dropout05 --dropout 0.5
```

Combinar varios parametros:

```bash
python train_mlp.py --experiment-name mlp_exp_custom --image-size 32 --batch-size 8 --learning-rate 0.0003 --dropout 0.5 --epochs 60
```

## Treinar MobileNetV2

```bash
python train_mobilenet.py
```

## Avaliar Modelo

O jeito recomendado e avaliar pelo experimento:

```bash
python evaluate.py --experiment experiments/cnn_exp002_img96
```

Nesse modo, o script le o `metadata.json`, carrega `model.keras` e usa automaticamente
o tamanho de imagem correto. Para isso, o `model.keras` precisa existir localmente,
mesmo estando fora do Git.

Tambem e possivel informar o modelo manualmente:

```bash
python evaluate.py --model cnn --model-path experiments/cnn_exp001_baseline/model.keras
```

A avaliacao usa as imagens rotuladas dentro de `test-kaggle/` e mostra:

- `loss`
- `accuracy`
- `classification_report`
- `confusion_matrix`

## Gerar Submissao Kaggle

O jeito recomendado e gerar pelo experimento:

```bash
python predict.py --experiment experiments/cnn_exp002_img96
```

Nesse modo, o script le o `metadata.json`, usa o tamanho correto da imagem e salva:

```text
experiments/cnn_exp002_img96/submission.csv
```

Tambem e possivel informar o modelo manualmente:

```bash
python predict.py --model cnn --model-path experiments/cnn_exp001_baseline/model.keras
```

O arquivo final usa as colunas:

```csv
id,class
```

## Testes Extras

A pasta `extra-tests/` e o arquivo `predict_extra_tests.py` fazem parte do repositorio.
Eles servem para testar imagens externas sem misturar com o dataset principal da
competicao.

Exemplo:

```bash
python predict_extra_tests.py --experiment experiments/cnn_exp002_img96
```

A saida padrao e:

```text
extra-tests/predictions.csv
```

Se esse arquivo estiver aberto ou bloqueado, o script salva um novo arquivo com
timestamp.

## Mapeamento Das Classes

O Kaggle espera os seguintes rotulos numericos:

| Classe | Rotulo |
| --- | ---: |
| Verde | 1 |
| Verde cana | 2 |
| Cereja | 3 |
| Passa | 4 |
| Seco | 5 |

## Observacoes

Como o conjunto de dados e pequeno, os resultados variam bastante entre treinos mesmo
com parametros parecidos. Sempre compare:

- metricas do `metadata.json`;
- resultado do `evaluate.py`;
- resultado publico/privado do Kaggle, quando houver.

Evite sobrescrever experimentos bons. Use sempre um `--experiment-name` novo para cada
rodada importante.

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
- `experiments/`: pastas de experimentos, com `metadata.json`, `model.keras` local e submissao quando gerada.
- `config.py`: caminhos principais, classes, rotulos e configuracoes globais.
- `utils.py`: carregamento dos datasets, normalizacao e data augmentation.
- `experiment_utils.py`: criacao de pastas de experimento e salvamento de metadados.
- `train_cnn.py`: treino de CNN com parametros configuraveis por comando.
- `optimize_cnn.py`: busca automatica de hiperparametros da CNN com Optuna.
- `train_mlp.py`: treino de MLP com parametros configuraveis por comando.
- `train_mobilenet.py`: treino com MobileNetV2 pre-treinada.
- `evaluate.py`: avaliacao local de um modelo treinado.
- `predict.py`: geracao de `submission.csv` para o Kaggle.

## O Que Entra No Git

Entram no repositorio:

- codigo Python;
- `README.md`;
- `requirements.txt`;
- imagens de treino e teste;
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

O jeito recomendado para muitos parametros e usar um arquivo YAML:

```bash
python train_cnn.py --config configs/cnn_example.yaml
```

Voce tambem pode sobrescrever qualquer valor do YAML pela linha de comando:

```bash
python train_cnn.py --config configs/cnn_example.yaml --experiment-name cnn_teste_bs4 --batch-size 4
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

Outros parametros aceitos pela CNN:

- `seed`: semente usada para inicializacao, embaralhamento e augmentation.
- `kernel_size`: tamanho do kernel convolucional.
- `conv_dropout`: dropout depois dos blocos convolucionais.
- `batch_normalization`: `true` ou `false`.
- `activation`: `relu`, `elu` ou `swish`.
- `optimizer`: `adam`, `adamw`, `rmsprop` ou `sgd`.
- `weight_decay`: usado pelo `adamw`.
- `label_smoothing`: suavizacao dos rotulos na loss.
- `class_weights`: usa pesos automaticos por classe no treino.
- `manual_class_weights`: pesos manuais por nome de classe, usados quando `class_weights` esta ativo.
- `patience`: paciencia do early stopping.
- `reduce_lr_factor`: fator do `ReduceLROnPlateau`.
- `reduce_lr_patience`: paciencia do `ReduceLROnPlateau`.
- `min_lr`: learning rate minimo.
- `random_brightness`: variacao de brilho no augmentation.
- `random_contrast`: variacao de contraste no augmentation.
- `random_translation`: deslocamento no augmentation.
- `random_zoom`: zoom no augmentation.

Combinar varios parametros:

```bash
python train_cnn.py --experiment-name cnn_exp_custom --image-size 96 --batch-size 16 --learning-rate 0.0003 --dropout 0.4 --filters 16,32,64 --dense-units 32 --l2 0.0001 --pooling gap --augmentation medium
```

Exemplo de pesos manuais no YAML:

```yaml
class_weights: true
manual_class_weights:
  Cereja: 1.0
  Passa: 1.0
  Seco: 1.0
  Verde: 1.2
  Verde cana: 1.3
```

Repetir uma rodada com a mesma semente:

```bash
python train_cnn.py --experiment-name cnn_exp_reprodutivel_a --seed 42
python train_cnn.py --experiment-name cnn_exp_reprodutivel_b --seed 42
```

Isso reduz bastante a variacao entre execucoes iguais. Ainda assim, com poucas imagens,
pequenas mudancas de treino ou validacao podem alterar muito as metricas.

## Otimizar CNN Com Optuna

O Optuna testa combinacoes de hiperparametros automaticamente e salva o melhor
resultado como um experimento normal:

```bash
python optimize_cnn.py --experiment-name cnn_optuna_best --trials 20 --epochs 30
```

O melhor modelo fica em:

```text
experiments/cnn_optuna_best/model.keras
```

E os metadados ficam em:

```text
experiments/cnn_optuna_best/metadata.json
```

Cada tentativa tambem recebe uma pasta propria:

```text
experiments/cnn_optuna_best_trial_000/
experiments/cnn_optuna_best_trial_001/
...
```

Por padrao, essas pastas guardam o `metadata.json`, mas nao guardam o
`model.keras` de todas as tentativas. O modelo completo fica salvo apenas no melhor
experimento para economizar espaco em disco. Para manter o modelo de cada tentativa:

```bash
python optimize_cnn.py --experiment-name cnn_optuna_best --trials 20 --epochs 30 --keep-trial-models
```

Por padrao, o estudo fica salvo em SQLite para poder continuar depois:

```text
experiments/optuna/cnn_optuna.db
```

Para continuar o mesmo estudo, rode novamente com o mesmo `--study-name`:

```bash
python optimize_cnn.py --study-name cnn_optuna --experiment-name cnn_optuna_best --trials 10 --epochs 30
```

O script otimiza por `val_accuracy`. Para otimizar por menor `val_loss`:

```bash
python optimize_cnn.py --direction minimize --experiment-name cnn_optuna_loss --trials 20 --epochs 30
```

Quando `--direction minimize` e usado, o checkpoint tambem salva o modelo com menor
`val_loss`. No modo padrao, ele salva o modelo com maior `val_accuracy`.

Para reduzir os logs de treino:

```bash
python optimize_cnn.py --experiment-name cnn_optuna_best --trials 20 --epochs 30 --verbose 0
```

Depois da busca, avalie ou gere submissao usando o experimento salvo:

```bash
python evaluate.py --experiment experiments/cnn_optuna_best
python predict.py --experiment experiments/cnn_optuna_best
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

Tambem e possivel escolher o nome do experimento:

```bash
python train_mobilenet.py --experiment-name mobilenet_exp001_baseline
```

O jeito recomendado para controlar os parametros e usar um YAML:

```bash
python train_mobilenet.py --config configs/mobilenet_example.yaml
```

No MobileNetV2, `train_base_layers` controla quantas camadas finais da base
pre-treinada ficam treinaveis. Use `0` para manter a base congelada, que costuma ser
mais estavel com poucos dados. Depois, teste valores pequenos como `20` com
`learning_rate` menor, por exemplo `0.00003`, para fine-tuning.

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
python evaluate.py --model cnn --model-path experiments/cnn_exp_img_96_bs_16_ln_0.001_f_16_32_du_32_ep_100/model.keras
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
python predict.py --model cnn --model-path experiments/cnn_exp_img_96_bs_16_ln_0.001_f_16_32_du_32_ep_100/model.keras
```

O arquivo final usa as colunas:

```csv
id,class
```

## Mapeamento Das Classes

O Kaggle espera os seguintes rotulos numericos:

| Classe | Rotulo |
| --- | ---: |
| Verde | 0 |
| Verde cana | 1 |
| Cereja | 2 |
| Passa | 3 |
| Seco | 4 |

## Observacoes

Como o conjunto de dados e pequeno, os resultados variam bastante entre treinos mesmo
com parametros parecidos. Sempre compare:

- metricas do `metadata.json`;
- resultado do `evaluate.py`;
- resultado publico/privado do Kaggle, quando houver.

Evite sobrescrever experimentos bons. Use sempre um `--experiment-name` novo para cada
rodada importante.

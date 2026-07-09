# AINET Coffee

Projeto para a competicao Kaggle AINET Coffee:
https://www.kaggle.com/competitions/ainet-coffee/overview

O objetivo é classificar imagens de grãos de café em cinco classes:

- `Verde`
- `Verde cana`
- `Cereja`
- `Passa`
- `Seco`

## Estrutura atual

- `Training set-kaggle/`: imagens de treino organizadas por classe.
- `test-kaggle/`: imagens de teste.
- `config.py`: caminhos, nomes de modelos, tamanho das imagens e mapeamento das classes.
- `utils.py`: carregamento dos datasets e aumento de dados.
- `train_cnn.py`: treino de uma CNN simples.
- `train_mlp.py`: treino de uma MLP simples.
- `train_mobilenet.py`: treino com MobileNetV2 pré-treinada.
- `evaluate.py`: avaliacao local em uma pasta rotulada.
- `predict.py`: geração do `submission.csv` no formato do Kaggle.
- `experiments/`: modelos e metadados de cada experimento.

## Como preparar o ambiente

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Treinar CNN

Comando base:

```bash
python train_cnn.py --experiment-name cnn_exp001_baseline
```

Alterar número de épocas:

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

Alterar quantidade de neurônios da camada densa:

```bash
python train_cnn.py --experiment-name cnn_exp_dense64 --dense-units 64
```

Alterar regularização L2:

```bash
python train_cnn.py --experiment-name cnn_exp_l2_0001 --l2 0.0001
```

Alterar pooling final:

```bash
python train_cnn.py --experiment-name cnn_exp_gap --pooling gap
```

Alterar augmentation:

```bash
python train_cnn.py --experiment-name cnn_exp_aug_light --augmentation light
```

```bash
python train_cnn.py --experiment-name cnn_exp_aug_strong --augmentation strong
```

Combinar vários parâmetros:

```bash
python train_cnn.py --experiment-name cnn_exp_custom --image-size 96 --batch-size 16 --learning-rate 0.0003 --dropout 0.4 --filters 16,32,64 --dense-units 32 --l2 0.0001 --pooling gap --augmentation medium
```

## Treinar MLP

Comando base:

```bash
python train_mlp.py --experiment-name mlp_exp001_baseline
```

Alterar número de épocas:

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

Combinar varios parâmetros:

```bash
python train_mlp.py --experiment-name mlp_exp_custom --image-size 32 --batch-size 8 --learning-rate 0.0003 --dropout 0.5 --epochs 60
```

## Treinar MobileNetV2

```bash
python train_mobilenet.py
```

## Avaliar localmente

Avaliar um experimento:

```bash
python evaluate.py --experiment experiments/cnn_exp002_img96
```

Nesse modo, o script lê o `metadata.json` e usa automaticamente o tamanho correto da
imagem.

Tambem e possível informar o modelo manualmente:

```bash
python evaluate.py --model cnn --model-path experiments/cnn_exp001_baseline/model.keras
```

## Gerar submissao

Gerar a submissão de um experimento:

```bash
python predict.py --experiment experiments/cnn_exp002_img96
```

Nesse modo, o script le o `metadata.json`, usa automaticamente o tamanho correto da
imagem e salva o resultado em `experiments/cnn_exp002_img96/submission.csv`.

Também e possível informar o modelo manualmente:

```bash
python predict.py --model cnn --model-path experiments/cnn_exp001_baseline/model.keras
```

O arquivo final usa as colunas:

```csv
id,class
```

## Mapeamento das classes

O Kaggle espera os seguintes rótulos numéricos:

| Classe | Rotulo |
| --- | ---: |
| Verde | 1 |
| Verde cana | 2 |
| Cereja | 3 |
| Passa | 4 |
| Seco | 5 |

## Notas de projeto

Os dados, modelos `.keras`, ambiente virtual e arquivo `submission.csv` ficam fora do
Git por padrão. Os arquivos `metadata.json` dentro de `experiments/` podem ser
versionados, porque registram os parâmetros e resultados sem armazenar modelos pesados.

# AINET Coffee

Projeto para a competicao Kaggle AINET Coffee:
https://www.kaggle.com/competitions/ainet-coffee/overview

O objetivo e classificar imagens de graos de cafe em cinco classes:

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
- `train_mobilenet.py`: treino com MobileNetV2 pre-treinada.
- `evaluate.py`: avaliacao local em uma pasta rotulada.
- `predict.py`: geracao do `submission.csv` no formato do Kaggle.
- `experiments/`: modelos e metadados de cada experimento.

## Como preparar o ambiente

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Treinar modelos

CNN simples:

```bash
python train_cnn.py --experiment-name cnn_exp001_baseline
```

Outro exemplo, mudando parametros:

```bash
python train_cnn.py --experiment-name cnn_exp006_lr0003 --learning-rate 0.0003 --dropout 0.5
```

Parametros principais da CNN:

```bash
python train_cnn.py --experiment-name cnn_exp013_aug_medium --image-size 96 --filters 16,32,64 --dropout 0.5 --learning-rate 0.0003 --batch-size 16 --l2 0.0001 --pooling gap --augmentation medium
```

Roteiro inicial recomendado:

```bash
python train_cnn.py --experiment-name cnn_exp001_baseline
python train_cnn.py --experiment-name cnn_exp002_img96 --image-size 96
python train_cnn.py --experiment-name cnn_exp006_lr0003 --image-size 96 --learning-rate 0.0003
python train_cnn.py --experiment-name cnn_exp010_l2_0001 --image-size 96 --learning-rate 0.0003 --l2 0.0001
python train_cnn.py --experiment-name cnn_exp013_aug_medium --image-size 96 --learning-rate 0.0003 --augmentation medium
python train_cnn.py --experiment-name cnn_exp019_gap --image-size 96 --learning-rate 0.0003 --pooling gap
```

MLP simples:

```bash
python train_mlp.py --experiment-name mlp_exp001_baseline
```

Exemplo mudando MLP:

```bash
python train_mlp.py --experiment-name mlp_exp002_img32 --image-size 32 --batch-size 8 --learning-rate 0.0003 --dropout 0.5
```

MobileNetV2:

```bash
python train_mobilenet.py
```

## Avaliar localmente

```bash
python evaluate.py --model mobilenet
```

Tambem e possivel avaliar a CNN:

```bash
python evaluate.py --model cnn --model-path experiments/cnn_exp001_baseline/model.keras
```

## Gerar submissao

Por padrao, a submissao usa o modelo `modelo_mobilenet.keras` e respeita a ordem do
`submission_template.csv`.

```bash
python predict.py --model mobilenet
```

Para usar a CNN:

```bash
python predict.py --model cnn --model-path experiments/cnn_exp001_baseline/model.keras
```

Para gerar a submissao de um experimento, o jeito recomendado e:

```bash
python predict.py --experiment experiments/cnn_exp002_img96
```

Nesse modo, o script le o `metadata.json`, usa automaticamente o tamanho correto da
imagem e salva o resultado em `experiments/cnn_exp002_img96/submission.csv`.

O arquivo final sera salvo como `submission.csv`, com as colunas:

```csv
id,class
```

## Mapeamento das classes

O Kaggle espera os seguintes rotulos numericos:

| Classe | Rotulo |
| --- | ---: |
| Verde | 1 |
| Verde cana | 2 |
| Cereja | 3 |
| Passa | 4 |
| Seco | 5 |

## Notas de projeto

Os dados, modelos `.keras`, ambiente virtual e arquivo `submission.csv` ficam fora do
Git por padrao. Os arquivos `metadata.json` dentro de `experiments/` podem ser
versionados, porque registram os parametros e resultados sem armazenar modelos pesados.

# textgen

Starter project for loading document datasets, iterating over batches with a torch `DataLoader`, and plugging in custom data transforms.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install dependencies:

```bash
uv sync
```

## Run

```bash
uv run python src/textgen/main.py --name funsd
```

## Project structure

```
src/textgen/
    main.py        # dataset loading, DataLoader construction, and iteration loop
    transforms.py  # TransformV1 — passthrough transform to extend with your logic
```

## Transforms

`TransformV1` in `src/textgen/transforms.py` is registered as `"transform_v1"` and accepts a `train` flag so you can apply different logic at train vs. eval time. It currently returns the document unchanged — replace the body of `__call__` with your preprocessing.

A separate transform instance is assigned to each dataset split:

- `train` split → `TransformV1(train=True)`
- `validation` / `test` splits → `TransformV1(train=False)`

## Run the starter pipeline

This starter command loads a dataset such as `funsd`, applies a passthrough transform, builds torch batches, logs a short preview for each sample, and saves image and text previews into the output directory.

```bash
uv run python src/textgen/main.py funsd
```
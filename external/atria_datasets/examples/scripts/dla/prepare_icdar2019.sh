#!/bin/bash

uv run prepare_dataset icdar2019/trackA_modern --max-samples 100
uv run prepare_dataset icdar2019/trackA_archival --max-samples 100
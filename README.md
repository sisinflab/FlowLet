---
license: mit
library_name: pytorch
pipeline_tag: unconditional-image-generation
tags:
  - medical-imaging
  - mri
  - brain
  - neuroimaging
  - 3d
  - flow-matching
  - wavelets
  - generative
  - rectified-flow
arxiv: 2601.05212
---

# FlowLet: Conditional 3D Brain MRI Synthesis using Wavelet Flow Matching

FlowLet is a conditional generative framework that synthesizes age-conditioned 3D brain MRI
volumes. It performs flow matching directly in an invertible 3D Haar wavelet domain, which gives
multi-scale generation without any learned latent compression and avoids the reconstruction
artifacts that latent diffusion models can introduce. Sampling is a deterministic Euler ODE, so
high-fidelity volumes are produced in few steps. Age is injected through two complementary
mechanisms (FiLM in the residual blocks for global modulation, and spatial cross-attention in the
transformer blocks for spatially adaptive control). A motivating application is Brain Age
Prediction (BAP): training BAP models with FlowLet-generated data improves performance for
under-represented age groups, while region-based analysis confirms preservation of anatomical
structure.

> Status: the four checkpoints listed below are currently in training.

![FlowLet architecture](assets/FlowLet_Architecture.png)


## Links

- Hugging Face paper page: https://huggingface.co/papers/2601.05212
- arXiv: https://arxiv.org/abs/2601.05212
- Code (GitHub): https://github.com/sisinflab/FlowLet
- Project page: https://danesed.github.io/flowlet-page/
- Model repository (this page): https://huggingface.co/danesed/FlowLet

## Model description

| Component | Value |
| --- | --- |
| Representation | Single-level 3D Haar DWT, producing 8 wavelet subbands (1 LLL approximation plus 7 detail), each at half spatial resolution |
| Network I/O | Conditional 3D U-Net, 8 input and 8 output channels (one per subband), 3D convolutions throughout |
| Backbone | 3D U-Net with `model_channels=128`, `num_res_blocks=2`, GroupNorm-32, and `SpatialTransformerConditional` attention blocks. Two configurations are released (see [Models](#models)). |
| Conditioning | Age (a single scalar), via FiLM in the residual blocks plus cross-attention in the transformer blocks. Condition embedding dimension 512. |
| Age normalization | Min-max to the [0, 1] interval using `condition_ranges.json`, then clamped to [0, 1] so values outside the training range do not extrapolate. |
| Objective | Rectified Flow Matching (straight-line interpolation between noise and data, constant target velocity). |
| Sampling | Euler ODE integration, deterministic given the seed. High quality in few steps (100 steps for the highest-fidelity results). |
| Output | NIfTI (`.nii.gz`), intensities rescaled to [0, 1], identity affine. |

The codebase also implements other flow formulations (`cfm`, `vp_diffusion`, `trigonometric`), but
only the Rectified Flow Matching checkpoints are released here.

## Models

Four checkpoints: two spatial resolutions, each in two U-Net configurations. All four
use Rectified Flow Matching (`rfm`) and age conditioning. The "base" and "large" configurations
differ in the U-Net channel multipliers and attention resolutions, and therefore in parameter
count.

| Model | Resolution (saved volume) | Config | U-Net params | Planned file | Status |
| --- | --- | --- | --- | --- | --- |
| FlowLet-RFM-91-base   | 91 x 109 x 91   | base (channel_mult 1,2,3,4 / attn 16,8) | 356.4 M | `rfm-91-base/flowlet_rfm_91_base.pth`     | In training, coming soon |
| FlowLet-RFM-91-large  | 91 x 109 x 91   | large (channel_mult 1,2,4,8 / attn 4,8) | 1.00 B  | `rfm-91-large/flowlet_rfm_91_large.pth`   | In training, coming soon |
| FlowLet-RFM-182-base  | 182 x 218 x 182 | base (channel_mult 1,2,3,4 / attn 16,8) | 356.4 M | `rfm-182-base/flowlet_rfm_182_base.pth`   | In training, coming soon |
| FlowLet-RFM-182-large | 182 x 218 x 182 | large (channel_mult 1,2,4,8 / attn 4,8) | 1.00 B  | `rfm-182-large/flowlet_rfm_182_large.pth` | In training, coming soon |

Each variant folder will also contain its `config.json` (the architecture the generation script
rebuilds the model from) and its `condition_ranges.json` (the age range used for normalization).
The 91 resolution uses a padded model input of 112 x 112 x 112, and the 182 resolution uses
224 x 224 x 224.



## How to use (ready for when the weights are released)

FlowLet uses a custom 3D architecture, so it is loaded with the repository code plus the released
`.pth`, not with `transformers` or `PyTorchModelHubMixin`. Once a checkpoint is available, download
it with its sidecar JSON files, then run the repository generation script.

```bash
# Code and environment
git clone https://github.com/sisinflab/FlowLet && cd FlowLet
conda create -n flowlet_env python=3.11 && conda activate flowlet_env
pip install -r requirements.txt   # torch==2.6.0, xformers optional
```

```python
# Download one variant (weights, config, age ranges). Available once Status shows released.
from huggingface_hub import hf_hub_download

repo_id = "danesed/FlowLet"
variant = "rfm-91-base"   # rfm-91-base | rfm-91-large | rfm-182-base | rfm-182-large
fname   = "flowlet_rfm_91_base.pth"

ckpt   = hf_hub_download(repo_id, f"{variant}/{fname}",               revision="main")
config = hf_hub_download(repo_id, f"{variant}/config.json",           revision="main")
ranges = hf_hub_download(repo_id, f"{variant}/condition_ranges.json", revision="main")
print(ckpt, config, ranges)
```

```bash
# Generate. The script rebuilds the model from config.json and normalizes age with
# condition_ranges.json. Arguments are a flat argparse (no subcommands), so flag order is free.
PYTHONPATH=. python3 -u scripts/generate.py \
    --checkpoint_path        "$CKPT" \
    --config_path            "$CONFIG" \
    --condition_ranges_path  "$RANGES" \
    --output_dir             ./generated/rfm-91-base \
    --generation_conditions  "Age=45" "Age=70.5" \
    --num_synthetic 5 \
    --num_flow_steps 100 \
    --save_size 91 109 91
```

For the 182 resolution variants pass `--save_size 182 218 182` (the padded input size is read from
the variant's `config.json`).

Notes:
- Attention uses `xformers` when available and falls back to native PyTorch attention automatically
  if it is not installed (a warning is logged). To force the fallback, set `"use_xformers": false`
  in the variant `config.json` before generating.
- Loading: the released `.pth` files are slimmed (weights under `model_state_dict` plus a small
  config block). The generation script calls `torch.load(..., map_location=device)` without setting
  `weights_only`. On torch 2.6 (pinned here) the default is `weights_only=True`, and the slimmed
  files contain only tensors and JSON-serializable config, so they load under that default.

## Training data

FlowLet was trained on preprocessed T1-weighted brain MRI from public research cohorts:

- OpenBHB: https://baobablab.github.io/bhb/dataset
- ADNI: https://adni.loni.usc.edu/
- OASIS-3: https://sites.wustl.edu/oasisbrains/

No imaging data is redistributed in this repository. Because of patient-privacy regulations and
data-use agreements, the scans cannot be shared here. Access must be requested from the original
providers under their respective agreements. Preprocessing (per the paper and the code repository):
N4ITK bias-field correction (ANTs), affine registration to MNI152 (FSL FLIRT), skull stripping
(FSL BET), resampling to 91 x 109 x 91, and z-score intensity normalization. The conditioning
variable is the subject Age, and the released `condition_ranges.json` covers Age in [5.90, 95.46].

## Intended use and limitations

Intended use: research on generative modeling of brain MRI, data augmentation for downstream
research (for example Brain Age Prediction), and benchmarking of flow-matching formulations.

Limitations and out-of-scope use:
- Not a medical device. No diagnostic, screening, or clinical use.
- Synthetic volumes may contain anatomical artifacts and do not correspond to real individuals.
- Outputs reflect the cohort bias of the training data (acquisition sites, scanners, demographics).
- Age is clamped to the training range [5.90, 95.46]. Values outside it are silently clipped, so
  out-of-range ages do not produce reliable extrapolation.
- Generation is conditioned on age only. Other clinical or morphological factors are not controlled.

## Citation

```bibtex
@misc{danese2026flowletconditional3dbrain,
      title={FlowLet: Conditional 3D Brain MRI Synthesis using Wavelet Flow Matching},
      author={Danilo Danese and Angela Lombardi and Matteo Attimonelli and Giuseppe Fasano and Tommaso Di Noia},
      year={2026},
      eprint={2601.05212},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.05212},
}

@article{danese2026flowlet,
  title   = {FlowLet: Conditional 3D Brain MRI Synthesis using Wavelet Flow Matching},
  author  = {Danese, Danilo and Lombardi, Angela and Attimonelli, Matteo and Fasano, Giuseppe and Di Noia, Tommaso},
  journal = {Medical Image Analysis},
  year    = {2026},
  publisher = {Elsevier},
  DOI     = {TO_BE_ASSIGNED}
}
```

## License

Released under the MIT License. See https://github.com/sisinflab/FlowLet/blob/main/LICENSE

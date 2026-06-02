# Jittor-IS

Jittor-IS is a collection of interactive image segmentation projects. It gathers several representative works and their Jittor implementations or Jittor-related implementations in one repository.

Each method is kept in an independent subdirectory with its own code structure, dependencies, checkpoints, demos, training scripts, and evaluation scripts. This top-level README provides a high-level overview of the repository. For detailed installation, dataset preparation, pretrained weights, training, and evaluation instructions, please refer to the README inside each subproject.

## Included Works

| Directory | Work | Venue / Year | Overview |
| --- | --- | --- | --- |
| `fcanet/` | FCANet: Interactive Image Segmentation with First Click Attention | CVPR 2020 | FCANet highlights the importance of the first user click. It introduces First Click Attention to explicitly use the first interaction point for feature modeling, improving the initial segmentation response. This directory provides the Jittor implementation, evaluation scripts, and an interactive annotation demo. |
| `focuscut/` | FocusCut: Diving into a Focus View in Interactive Segmentation | CVPR 2022 | FocusCut introduces a focus-view strategy that guides the model to concentrate on local regions around user clicks. It is designed for iterative local refinement in interactive segmentation. This directory includes the Jittor implementation, training and evaluation code, and a UI demo. |
| `simpleclick/` | SimpleClick: Interactive Image Segmentation with Simple Vision Transformers | ICCV 2023 | SimpleClick builds an interactive segmentation framework with simple Vision Transformers. It follows a click-based interaction pipeline and provides strong performance on multiple interactive segmentation benchmarks. This directory contains demos, evaluation scripts, training scripts, and model configurations. |
| `segment-anything/` | Segment Anything | ICCV 2023 | Segment Anything is a promptable segmentation foundation model that supports point, box, and mask prompts. It can be used for zero-shot or weakly interactive object segmentation. This directory contains the Jittor-related SAM code, scripts, and examples. |
| `MFP/` | MFP: Making Full Use of Probability Maps for Interactive Image Segmentation | CVPR 2024 | MFP studies how to better use probability maps from previous predictions during interactive segmentation. By feeding historical prediction information back into the model, it improves segmentation quality after later user clicks. This directory follows a SimpleClick/RITM-style framework and includes training and evaluation entry points. |

## Repository Structure

```text
Jittor-IS/
├── fcanet/            # FCANet
├── focuscut/          # FocusCut
├── simpleclick/       # SimpleClick
├── segment-anything/  # Segment Anything
├── MFP/               # MFP
└── README.md          # Top-level overview
```

## Usage

Each method is maintained as a separate project. To use a specific method, enter its directory first:

```bash
cd simpleclick
```

Then follow the README in that directory to install dependencies, download datasets and pretrained weights, configure local paths, and run the corresponding demo, evaluation, or training scripts.

Different methods may require different Python, Jittor, PyTorch, CUDA, or package versions. Creating separate environments for different methods is recommended.

## Notes

- This repository is intended for organizing, reproducing, comparing, and extending interactive image segmentation methods.
- Datasets and pretrained checkpoints are generally not included in the repository. Please download them according to the instructions in each subproject.
- Licenses, citation requirements, and usage restrictions may differ across subprojects. Please follow the original requirements of each work.

## Citation

If you use the code or models from a specific method, please cite the corresponding paper. BibTeX entries are provided in the README files under the respective subdirectories.

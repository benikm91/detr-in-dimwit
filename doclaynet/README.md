# DocLayNet: related work

The papers to read before DocLayNet becomes a dataset of this project, in reading order. All from
arXiv; the venue is where each was published.

| file | paper | venue | why |
|---|---|---|---|
| `01-doclaynet-kdd2022.pdf` | DocLayNet: A Large Human-Annotated Dataset for Document-Layout Analysis | KDD 2022 | the dataset: 80,863 pages, 11 classes, annotator agreement 82–83 mAP, the CNN and YOLO baselines |
| `02-icdar2023-competition-robust-layout-segmentation.pdf` | ICDAR 2023 Competition on Robust Layout Segmentation in Corporate Documents | ICDAR 2023 | the ICDAR competition built on DocLayNet, with a harder test set; 21 teams |
| `03-welayout-icdar2023-winner.pdf` | WeLayout: WeChat Layout Analysis System for the ICDAR 2023 Competition … | ICDAR 2023 | the winner: an ensemble of DINO and YOLO, 70.0 mAP on the competition's test set |
| `04-detr-eccv2020.pdf` | End-to-End Object Detection with Transformers (DETR) | ECCV 2020 | the bridge: object detection as set prediction |
| `05-dino-iclr2023.pdf` | DINO: DETR with Improved DeNoising Anchor Boxes | ICLR 2023 | the DETR variant behind most of the leading DocLayNet results |
| `06-swindocsegmenter-icdar2023.pdf` | SwinDocSegmenter: An End-to-End Unified Domain Adaptive Transformer for Document Instance Segmentation | ICDAR 2023 | a DETR-style model at ICDAR; 76.9 mAP on DocLayNet |
| `07-rodla-cvpr2024.pdf` | RoDLA: Benchmarking the Robustness of Document Layout Analysis Models | CVPR 2024 | the strongest DETR-family result found, 80.5 mAP (as listed by HybriDLA) |
| `08-dlaformer-icdar2024.pdf` | DLAFormer: An End-to-End Transformer For Document Layout Analysis | ICDAR 2024 | an end-to-end transformer at ICDAR; its DocLayNet number is still to be looked up in it |
| `09-doclayout-yolo-2024.pdf` | DocLayout-YOLO: Enhancing Document Layout Analysis through Diverse Synthetic Data … | arXiv 2024 | the non-transformer state of the art, 79.7 mAP with synthetic pre-training |
| `10-hybridla-aaai2026.pdf` | HybriDLA: Hybrid Generation for Document Layout Analysis | AAAI 2026 | the claimed state of the art, 83.5 mAP: diffusion with **autoregressive** decoding, so set against sequence again |

The mAP numbers are COCO mAP@0.5:0.95 on the DocLayNet test split as the papers report them; several
come from other papers' comparison tables and differ by training setup.

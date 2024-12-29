# Multi-modal recommender system for content-streaming platforms

## Description
This repository contains the code, data, and resources for a master thesis project exploring a multi-modal recommender system for content-streaming platforms. The system integrates textual and visual features, leveraging movie subtitles and posters, and employs Disentangled Multimodal Representation Learning (DMRL) to improve recommendation quality.

## Repository structure
Main structure:
```
root/
├── assets/                     # Graphical assets generated for the LaTeX report
├── dataset-helpers/            # Scripts for preprocessing and preparing the dataset
├── demo/                       # Demo web application for testing recommendations
│   ├── README.md               # Instructions for running the NodeJS application
├── *.py                        # Python scripts in the root for model development
├── README.md                   # Main repository README
```
Create the following structure for holding your data:
```
└── data/                       # Directory for holding raw and processed data
│   ├── ml-20m-psm              # Unzip the dataset here
│   ├── processed/              # NPZ numpy arrays with ids and features go here
│       ├── posters             # Processed images will go here
│       ├── subtitles           # Processed subtitles will go here
```

## Source files

Root folder:

- `baseline_coldstart.py` Evaluates baseline models for cold start
- `baseline_global_avg.py`Evaluates baseline GlobalAvg model       
- `baseline_longatil.py` Evaluates baseline models for long tail
- `baseline_mf.py` Evaluates Matrix Factorization model
- `baseline_mf2.py` Evaluates a variant of MF
- `baseline_most_pop.py` Baseline model for TopPop
- `dmrl_example.py.ipynb` Jupyter notebook for DMRL exploration
- `dmrl_images.py` Generates images from vector embeddings
- `evaluation_charts.py` Generates charts with results for the report
- `movielens_1M_vgg16.py` Trains DMRL on VGG16 with MovieLens-1M
- `movielens_1M_vit_H_14.py` Trains DMRL with ViT-14 with MovieLens-1M
- `movielens_1M_vit_L_16.py` Trains DMRL with ViT-16 with MovieLens-1M
- `movielens_1M_vit_L_32.py` Trains DMRL with ViT-32 with MovieLens-1M

The following files overload a method from the DMRL model implementation in Cornac to be able to use external embeddings created by arbitrary models without passing text directly to DMRL. Extremely ugly but it works.
- `movielens_100k_external_text_embeddings.py` Trains using BERT embeddings
- `movielens_100k_external_text_embeddings_coldstart.py` Trains using BERT for cold start
- `movielens_100k_external_text_embeddings_longtail.py` Trains using BERT for long tail


- `movielens_100k_raw.py` Trains DMRL on ML100k using DMRL's internal embeddings
- `movielens_100k_vgg16.py` Trains DMRL on ML100k with VGG16
- `movielens_100k_vit_H_14.py` Trains DMRL on ML100k with ViT-14
- `movielens_100k_vit_L_16.py` Trains DMRL on ML100k with ViT-16
- `movielens_100k_vit_L_16_hyperopt.py` Abandoned test of hyperparam optimization on DMRL.
- `movielens_100k_vit_L_32.py` Trains DMRL on ML100k with ViT-32
- `poster_similarity.ipynb` Obtains poster similarity measure from embeddings.

Dataset-helpers folder:

- `cross_modality_embeddings.py`
- `process_credits_0_ids.py` movieId syncing across datasets
- `process_metadata_0_ids.py` movieId syncing across datasets
- `process_posters_0_size.py` movieId syncing across datasets
- `process_posters_1_extract_vgg16.py` Extract VGG16 features - only CNN
- `process_posters_1_extract_vgg16_v2.py` Extract VGG16 features - up to FC2
- `process_posters_2_extract_vit_h_14.py` Extract ViT-14 embeddings - pooor heads removal
- `process_posters_2_extract_vit_h_14_v2.py` Extract ViT-14 embeddings - better heads removal
- `process_posters_2_extract_vit_l_16.py` Extract ViT-16 embeddings - minus classifier
- `process_posters_2_extract_vit_l_32.py` Extract ViT-32 embeddings - minus classifier
- `process_posters_2_raw.py`
- `process_subtitles_0_ids.py` movieId syncing across datasets
- `process_subtitles_1_clean.py` Remove timings from subtitles
- `process_subtitles_2_load.py` Convert to plaintext
- `subtitles_bert_large_chunking.py` Create external BERT-Base instead of built-in Sentence Transformers
- `subtitles_st_p_miniLM_L6_v2.py` External features of subtitles with the same model as the built-in for reuse. DMRL's own reuse not working properly.

## Dataset files

The dataset is available in Zenodo. 

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14571726.svg)](https://www.doi.org/10.5281/zenodo.14571726)


## Usage

The files inside dataset-helpers are numbered based on their position in the processing pipeline. Non-numbered files mean that they are not part of the main pipeline and they perform auxiliary tasks for different purposes.
Files in the root repository should not need to be executed in any particular order as they are self-contained to execute specific experiments or provide specific results.

## Resources

This repository accompanies the master thesis project and references several key tools and frameworks, including:
- Python 3.11
- CUDA is highly recommended although models can run on CPU if needed.
- Cornac 2.3.0 for the main training and evaluation pipelines.
- Numpy 1.26.4. Higher versions are not yet supported by Cornac.
- MovieLens-100K dataset
- MovieLens-20M Posters Subtitles Multi-modal hosted in Zenodo.
- Pretrained models such as Sentence Transformers and VGG16 will be downloaded automatically by Torchvision as required.


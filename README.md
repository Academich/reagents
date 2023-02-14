# Molecular Transformer for Reagents Prediction
This is the code for the preprint [Reagent Prediction with a Molecular Transformer Improves Reaction Data Quality](https://doi.org/10.26434/chemrxiv-2022-sn2kr).  
The repository is effectively a fork of the [Molecular Transformer](https://github.com/pschwllr/MolecularTransformer).  

Idea:
  * Train a transformer to predict reagents for organic reactions in the way of SMILES-to-SMILES translation.
  ![](reactants_vs_reagents.png "")
  * Infer missing reagents for some reactions in the training set.
  * Train a transformer for reaction product prediction on the dataset with improved reagents.

## Code
### Old files:
`onmt` is a directory with the OpenNMT code.  
`preprocess.py`, `train.py`, `translate.py`, `score_predictions.py` are files used in the Molecular Transformer code.
### New files: 
`src` folder contains preprocessing for reagent prediction.  
`prepare_data.py` is the main script that preprocesses USPTO for reagents prediction with MT.  
`environment.yml` is the conda environment specification.

## Data
The entire USPTO data used to assemble the training set for reagent prediction can be downloaded [here](https://ibm.ent.box.com/v/ReactionSeq2SeqDataset). 

The training set for reagents prediction was obtained from it using the `prepare_data.py` script. It does not overlap with the USPTO MIT test set. The tokenized data can be downloaded [here](https://drive.google.com/drive/folders/1Z-8UsXqScwtV4s9oXYYY1OcbDaYvxLWi?usp=sharing).

The tokenized data for product prediction is stored [here](https://ibm.box.com/v/MolecularTransformerData). For the description of these data, please refer to the README of the original [Molecular Transformer](https://github.com/pschwllr/MolecularTransformer).

The data for product prediction with altered reagents can be downloaded [here](https://drive.google.com/drive/folders/1Cr8AsAIYSGaXJuBnlVflBax3ZbEbq89s?usp=sharing).

## Workflow
 1. Create a conda environment from the specification file:
    ```bash
    conda env create -f environment.yml
    conda activate reagents_pred
    ```
 2. Download the datasets and put them in the `data/tokenized` directory.

 3. Train a reagents prediction model:
    First, preprocess the data for an OpenNMT model:
    ```bash
        python3 preprocess.py -train_src data/tokenized/${DATASET_NAME}/src-train.txt \
                              -train_tgt data/tokenized/${DATASET_NAME}/tgt-train.txt \
                              -valid_src data/tokenized/${DATASET_NAME}/src-val.txt \
                              -valid_tgt data/tokenized/${DATASET_NAME}/tgt-val.txt \
                              -save_data data/tokenized/${DATASET_NAME}/${DATASET_NAME} \
                              -src_seq_length 1000 -tgt_seq_length 1000 \
                              -src_vocab_size 1000 -tgt_vocab_size 1000 -share_vocab
    ```
    
    Then, run a model:
    ```bash
        python3 train.py -data data/tokenized/${DATASET_NAME}/${DATASET_NAME} \
                         -save_model experiments/checkpoints/${DATASET_NAME}/${DATASET_NAME}_model \
                         -seed 42 -gpu_ranks 0 -save_checkpoint_steps 10000 -keep_checkpoint 20 \
                         -train_steps 500000 -param_init 0  -param_init_glorot -max_generator_batches 32 \
                         -batch_size 4096 -batch_type tokens -normalization tokens -max_grad_norm 0  -accum_count 4 \
                         -optim adam -adam_beta1 0.9 -adam_beta2 0.998 -decay_method noam -warmup_steps 8000  \
                         -learning_rate 2 -label_smoothing 0.0 -report_every 10 \
                         -layers 4 -rnn_size 256 -word_vec_size 256 -encoder_type transformer -decoder_type transformer \
                         -dropout 0.1 -position_encoding -share_embeddings \
                         -global_attention general -global_attention_function softmax -self_attn_type scaled-dot \
                         -heads 8 -transformer_ff 2048 -tensorboard
    ```
 5. Train a basline product prediction model:  
    Train a Molecular Transformer on, say, `MIT_separated` data. For this, run the `preprocess.py` script and the `train.py` script  
    as shown above but with `DATASET_NAME=MIT_separated`.

 6. Use a trained reagent model to improve reagents in a dataset for product prediction.   
    The script `reagent_substitution.py` uses a reagents prediction model to change reagents in data which is the input  
    to a product prediction model. To change the reagents in, say, `MIT_separated`, run the script as follows:
    ```bash
        python3 reagent_substitution.py --data_dir data/tokenized/MIT_separated \ 
                                        --reagent_model <MODEL_NAME> \ 
                                        --reagent_model_vocab <MODEL_SRC_VOCAB> \
                                        --beam_size 5 --gpu 0
    ```
    `MODEL_NAME` may be stored, in `experiments/checkpoints/`.
    `MODEL_SRC_VOCAB` (a .json file )may be stored in `data/vocabs/`.  
    Or download the final data [here](https://drive.google.com/drive/folders/1Cr8AsAIYSGaXJuBnlVflBax3ZbEbq89s?usp=sharing).
    
  7. Train product prediction models on datasets cleaned by a reagents prediction model like in step 5.   

The trained reagent and product models in the forms of `.pt` files are stored [here](https://drive.google.com/drive/folders/1weP5wG_wzd9qb3RVnCz0JjyB-TJF-5yQ).

## Inference
To make predictions for reactions supplied in a .txt file as SMILES, use the following script:

    ```bash
        python3 translate.py -model <PATH TO THE MODEL WEIGHTS> \
                    -src <PATH TO THE TEST REACTIONS WITHOUT REAGENTS> \
                    -output <PATH TO THE .TXT FILE WHERE THE PREDICTIONS WILL BE STORED> \
                    -batch_size 64 -replace_unk -max_length 200 -fast -beam_size 5 -n_best 5 -gpu <GPU ID>
    ```
The supplied reactions should be tokenized with tokens separated by spaces, like in files produced by `prepare_data.py`.

## Citation
The preprint: 
```
@article{andronov_voinarovska_andronova_wand_clevert_schmidhuber_2022, 
 place={Cambridge}, 
 title={Reagent Prediction with a Molecular Transformer Improves Reaction Data Quality}, 
 DOI={10.26434/chemrxiv-2022-sn2kr}, 
 journal={ChemRxiv},
 publisher={Cambridge Open Engage}, 
 author={Andronov, Mikhail and Voinarovska, Varvara and Andronova, Natalia and Wand, Michael and Clevert, Djork-Arné and Schmidhuber, Jürgen}, 
 year={2022}
 } 
```

The underlying framework:

```
@inproceedings{opennmt,
  author    = {Guillaume Klein and
               Yoon Kim and
               Yuntian Deng and
               Jean Senellart and
               Alexander M. Rush},
  title     = {Open{NMT}: Open-Source Toolkit for Neural Machine Translation},
  booktitle = {Proc. ACL},
  year      = {2017},
  url       = {https://doi.org/10.18653/v1/P17-4012},
  doi       = {10.18653/v1/P17-4012}
}
```

# Molecular Transformer for Reagents Prediction

The repository is effectively a fork of the [Molecular Transformer](https://github.com/pschwllr/MolecularTransformer).  

### New files: 
`src` folder contains preprocessing for reagent prediction.  
`prepare_data.py` is the main script that preprocesses USPTO for reagents prediction with MT.  
`environment.yml` is the conda environment specification.

### Workflow
 1. Create a conda environment from the specification file:
    ```bash
    conda env create -f environment.yml
    conda activate reagents_pred
    ```
 2. Download the US patents 1976-2016 data from https://ibm.ent.box.com/v/ReactionSeq2SeqDataset.
 3. Store it in the directory `data/raw`.
 4. Merge train, validation and test files into one file:
    ```bash
        sed -i '1,2d' data/raw/US_patents_1976-Sep2016_1product_reactions_train.csv
        sed -i '1,3d' data/raw/US_patents_1976-Sep2016_1product_reactions_valid.csv
        sed -i '1,3d' data/raw/US_patents_1976-Sep2016_1product_reactions_test.csv
        cat data/raw/US_patents_1976-Sep2016_1product_reactions_train.csv data/raw/US_patents_1976-Sep2016_1product_reactions_valid.csv data/raw/US_patents_1976-Sep2016_1product_reactions_test.csv >> data/raw/uspto_stereo.csv
    ```
 5. Split the data into the training set and the validation set:
    ```bash
        python3 train_val_split.py --val_size 5000 --path_data data/raw/uspto_stereo.csv --save_folder data/raw   
    ```
 6. Prepare data for reagents prediction:  
    Training set:
    ```bash
        SUFFIX=simple_tok_with_rep_aug
        python3 prepare_data.py --train --filepath data/raw/uspto_train.csv --output_suffix ${SUFFIX} \
                                --min_reagent_occurances 20 --use_augmentations
    ```
    Validation set:
    ```bash
        python3 prepare_data.py --filepath data/raw/uspto_val.csv --output_suffix ${SUFFIX} \
                                --min_reagent_occurances 20
    ```
    For information about the arguments of the `prepare_data.py` script run `python3 prepare_data.py --help`.
    Logs of this script's work are written into `prepare_data.log`.
 7. Train a reagents prediction model:
    First, preprocess the data for an OpenNMT model:
    ```bash
        python3 preprocess.py -train_src data/tokenized/${SUFFIX}/src-train.txt \
                              -train_tgt data/tokenized/${SUFFIX}/tgt-train.txt \
                              -valid_src data/tokenized/${SUFFIX}/src-val.txt \
                              -valid_tgt data/tokenized/${SUFFIX}/tgt-val.txt \
                              -save_data data/tokenized/${SUFFIX}/${SUFFIX} \
                              -src_seq_length 1000 -tgt_seq_length 1000 \
                              -src_vocab_size 1000 -tgt_vocab_size 1000 -share_vocab
    ```
    Do not use the `-share_vocab` argument if `prepare_data.py` was run with `--use_special_tokens`.  
    
    Then, run a model:
    ```bash
        python3 train.py -data data/tokenized/${SUFFIX}/${SUFFIX} \
                         -save_model experiments/checkpoints/${SUFFIX}/${SUFFIX}_model \
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
    Do not use `-share_embeddings` argument if `preprocess.py` was run without `-share_vocab`.
 8. Train a product prediction model:  
    Download the tokenized data used in the Molecular Transformer paper from [here](https://ibm.box.com/v/MolecularTransformerData) and save it to the `data/tokenized` directory.  
    For the description of the data, please refer to the README of the original [Molecular Transformer](https://github.com/pschwllr/MolecularTransformer).

    Train a Molecular Transformer on, say, `MIT_separated` data. For this, run the `preprocess.py` script and the `train.py` script  
    as shown above but with `SUFFIX=MIT_separated`, `-share_vocab` and `-share_embeddings`.

 9. Train product prediction models on datasets cleaned by a reagents prediction model:   
    The script `reagent_substitution.py` uses a reagents prediction model to change reagents in data which is the input  
    to a product prediction model. To change the reagents in, say, `MIT_separated data`, run the script as follows:
    ```bash
        python3 reagent_substitution.py --data_dir data/tokenized/MIT_separated \ 
                                        --reagent_model <MODEL_NAME> \ 
                                        --reagent_model_vocab <MODEL_SRC_VOCAB> \
                                        --beam_size 5 --gpu 0
    ```
    `MODEL_NAME` may be, for example, `experiments/checkpoints/simple_tok_with_rep_aug/simple_tok_with_rep_aug_model_step_500000.pt`.
    `MODEL_SRC_VOCAB` then may be `data/vocabs/simple_tok_with_rep_aug_src_vocab.json`.
    
    
## Citation

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

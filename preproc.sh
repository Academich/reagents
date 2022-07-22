dataset=MIT_mixed
python preprocess.py -train_src data/tokenized/${dataset}/src-train.txt \
                     -train_tgt data/tokenized/${dataset}/tgt-train.txt \
                     -valid_src data/tokenized/${dataset}/src-val.txt \
                     -valid_tgt data/tokenized/${dataset}/tgt-val.txt \
                     -save_data data//tokenized/${dataset}/${dataset} \
                     -src_seq_length 1000 -tgt_seq_length 1000 \
                     -src_vocab_size 1000 -tgt_vocab_size 1000 -share_vocab
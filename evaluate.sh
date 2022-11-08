dataset=MIT_mixed
model=${dataset}_model_step_20000.pt

python translate.py -model experiments/checkpoints/${dataset}/${model} \
                    -src data/${dataset}/src-test.txt \
                    -output experiments/results/predictions_${model}_on_${dataset}_test-perm.txt \
                    -batch_size 64 -replace_unk -max_length 200 -fast -beam_size 5 -n_best 5 -attn_debug

python score_predictions.py -targets data/${dataset}/tgt-test.txt \
                    -predictions experiments/results/predictions_${model}_on_${dataset}_test-perm.txt
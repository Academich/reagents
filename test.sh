python translate.py -model experiments/trained_models/reag_simple_tok_with_rep_model_step_500000.pt \
                    -src data/test/src-test-reagents.txt \
                    -output experiments/results/predictions_on_test_reagents.txt \
                    -batch_size 64 -replace_unk -max_length 200 -fast -beam_size 5 -n_best 5
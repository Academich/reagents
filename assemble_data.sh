RAW_DATA_FOLDER=data/raw
MERGED_DATA=uspto_stereo.csv

## === After downloading Schwaller's data from IBM cloud, merging all three files into one
#sed -i '1,2d' ${RAW_DATA_FOLDER}/US_patents_1976-Sep2016_1product_reactions_train.csv
#sed -i '1,3d' ${RAW_DATA_FOLDER}/US_patents_1976-Sep2016_1product_reactions_valid.csv
#sed -i '1,3d' ${RAW_DATA_FOLDER}/US_patents_1976-Sep2016_1product_reactions_test.csv
#cat ${RAW_DATA_FOLDER}/US_patents_1976-Sep2016_1product_reactions_train.csv ${RAW_DATA_FOLDER}/US_patents_1976-Sep2016_1product_reactions_valid.csv ${RAW_DATA_FOLDER}/US_patents_1976-Sep2016_1product_reactions_test.csv >> ${RAW_DATA_FOLDER}/${MERGED_DATA}
#
## === Splitting train and test
#python3 train_val_split.py --val_size 5000 --path_data ${RAW_DATA_FOLDER}/${MERGED_DATA} --save_folder ${RAW_DATA_FOLDER}

# === Preprocessing data
TRAIN_DATA=uspto_train.csv
STANDARD_SOLVENTS=data/standard/solvents.csv
VAL_DATA=uspto_val.csv
MIN_REAG_OCCURS=20
# 1. Default tokenization, repeated molecules allowed in reactions
SUFFIX=simple_tok_with_rep_aug

python3 prepare_data.py --train --filepath ${RAW_DATA_FOLDER}/${TRAIN_DATA} --output_suffix ${SUFFIX} \
                        --standard_solvents_path ${STANDARD_SOLVENTS} \
                        --min_reagent_occurances ${MIN_REAG_OCCURS} --use_augmentations

python3 prepare_data.py --filepath ${RAW_DATA_FOLDER}/${VAL_DATA} --output_suffix ${SUFFIX} \
                        --standard_solvents_path ${STANDARD_SOLVENTS} \
                        --min_reagent_occurances ${MIN_REAG_OCCURS}

# 2. Default tokenization, all molecules in a reaction are unique
SUFFIX=simple_tok_no_rep_aug

python3 prepare_data.py --train --filepath ${RAW_DATA_FOLDER}/${TRAIN_DATA} --output_suffix ${SUFFIX} \
                        --standard_solvents_path ${STANDARD_SOLVENTS} \
                        --min_reagent_occurances ${MIN_REAG_OCCURS} --use_augmentations --keep_only_unique_molecules

python3 prepare_data.py --filepath ${RAW_DATA_FOLDER}/${VAL_DATA} --output_suffix ${SUFFIX} \
                        --standard_solvents_path ${STANDARD_SOLVENTS} \
                        --min_reagent_occurances ${MIN_REAG_OCCURS} --keep_only_unique_molecules

# 3. Special tokenization, repeated molecules allowed in reactions
SUFFIX=special_tok_with_rep_aug

python3 prepare_data.py --train --filepath ${RAW_DATA_FOLDER}/${TRAIN_DATA} --output_suffix ${SUFFIX} \
                        --standard_solvents_path ${STANDARD_SOLVENTS} \
                        --min_reagent_occurances ${MIN_REAG_OCCURS} --use_augmentations --use_special_tokens

python3 prepare_data.py --filepath ${RAW_DATA_FOLDER}/${VAL_DATA} --output_suffix ${SUFFIX} \
                        --standard_solvents_path ${STANDARD_SOLVENTS} \
                        --min_reagent_occurances ${MIN_REAG_OCCURS} --use_special_tokens

# 4. Special tokenization, all molecules in a reaction are unique
SUFFIX=special_tok_no_rep_aug

python3 prepare_data.py --train --filepath ${RAW_DATA_FOLDER}/${TRAIN_DATA} --output_suffix ${SUFFIX} \
                        --standard_solvents_path ${STANDARD_SOLVENTS} \
                        --min_reagent_occurances ${MIN_REAG_OCCURS}  \
                        --use_augmentations --use_special_tokens --keep_only_unique_molecules

python3 prepare_data.py --filepath ${RAW_DATA_FOLDER}/${VAL_DATA} --output_suffix ${SUFFIX} \
                        --standard_solvents_path ${STANDARD_SOLVENTS} \
                        --min_reagent_occurances ${MIN_REAG_OCCURS} --use_special_tokens --keep_only_unique_molecules
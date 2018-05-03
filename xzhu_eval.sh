#!/bin/bash
root_dir=/Users/zhuxz/Documents/studio/wsd-dynamic-sense-vector
version="emnlp"
name="google1b-for-lstm-wsd"

raw_data_path=$root_dir/data/google1b.txt

preprocessed_data_path=$root_dir/preprocessed-data/$version/$name
mkdir -p $root_dir/preprocessed-data/$version
vocab_path=$preprocessed_data_path.index.pkl

output_path=$root_dir/output/$version/$name
result_dir=$output_path/results
mkdir -p $result_dir

# Get Google1B data
echo "Downloading Google1B..."
wget http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz
tar xvf 1-billion-word-language-modeling-benchmark-r13output.tar.gz
mv 1-billion-word-language-modeling-benchmark-r13output google1b
mv googel1b $root_dir/data/
cat $root_dir/data/google1b/training-monolingual.tokenized.shuffled/* >> $raw_data_path
# After this step, a single file where each line is a tokenized sentence should be found at 
#   data/google1b.txt  (raw data)

# Preprocess data
# This will take a file each line is a sentence, and generate word2idx, train.npz, dev.npz etc.
# These files will be put under "preprocessed-data/version/google1b-for-lstm-wsd/*"
# The input and output path is hardcoded right now, need to change
# Also, the vocab size need to be changed here.
echo "Preprocessing data..."
python3 $root_dir/prepare-lstm-wsd.py -i $raw_data_path -o $preprocessed_data_path
# After this step, the vocab, train&dev test sets should be found at
#   preprocessed-data/emnlp/google1b-for-lstm-wsd.index.pkl (vocab)
#   preprocessed-data/emnlp/google1b-for-lstm-wsd.train.npz (train set)
#   preprocessed-data/emnlp/google1b-for-lstm-wsd.dev.npz   (development set)

# Train LSTM Language Model
echo "Training LSTM language model..."
python3 $root_dir/train-lstm-wsd.py --model google \
        --data_path $preprocessed_data_path \
        --dev_path $preprocessed_data_path.dev.npz \
        --vocab_path $vocab_path \
        --save_path $output_path && \
# Then there should be model generated
model_path=$output_path-best-model
# After this step, a trained model should be found at
#.   output/emnlp/google1b-for-lstm-wsd-best-model.index (trained lstm language model)

# Evaluation
mfs_fallback=True
batch_size=400
emb_setting=synset
eval_setting=synset
use_case_strategy=False
use_number_strategy=False
out_dir="$result_dir/synset-se2-framework-semcor"
# For this step, we need processed WSD annotated data under evaluate/higher_level_annotations/
# This could be obtained by checkout master branch, and follow evaluate/README.md
base=$root_dir/evaluate/higher_level_annotations/se2-aw-framework-synset-30_semcor
use_lp_strategy=False

sense_annotations_path="$base.txt"
wsd_df_in="$base.bin"
path_case_freq="$base.case_freq"
path_plural_freq="$base.plural_freq"
path_lp="$base.lp.out"
ehco "Evaluating..."
$root_dir/evaluate/one_full_experiment_v2.sh \
    $out_dir $batch_size $emb_setting \
    $eval_setting $model_path $vocab_path \
    $sense_annotations_path $wsd_df_in \
    $mfs_fallback $path_case_freq $use_case_strategy \
    $path_plural_freq $use_number_strategy \
    $path_lp $use_lp_strategy
# After this step, the result should be found under
#   output/emnlp/google1b-for-lstm-wsd/result/synset-se2-framework-semcor
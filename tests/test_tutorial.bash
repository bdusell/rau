set -euo pipefail

temp_dir=$(mktemp -d)
trap "rm -r -- $temp_dir" EXIT

lm_data=$temp_dir/lm/data
mkdir -p $lm_data
curl -s https://raw.githubusercontent.com/tommccoy1/rnn-hierarchical-biases/master/data/question.train | sed 's/[a-z]\+\t.*//' > $lm_data/main.tok
mkdir $lm_data/datasets
mkdir $lm_data/datasets/validation
curl -s https://raw.githubusercontent.com/tommccoy1/rnn-hierarchical-biases/master/data/question.dev | sed 's/[a-z]\+\t.*//' > $lm_data/datasets/validation/main.tok
mkdir $lm_data/datasets/test
curl -s https://raw.githubusercontent.com/tommccoy1/rnn-hierarchical-biases/master/data/question.test | sed 's/[a-z]\+\t.*//' > $lm_data/datasets/test/main.tok

rau lm prepare \
  --training-data $lm_data \
  --more-data validation \
  --more-data test \
  --never-allow-unk

for architecture in \
  transformer \
  rnn \
  lstm \
  superposition-stack-transformer \
  nondeterministic-stack-transformer \
  ; do

  case $architecture in
    transformer)
      model_args=( \
        --architecture $architecture \
        --num-layers 6 \
        --d-model 64 \
        --num-heads 8 \
        --feedforward-size 256 \
        --dropout 0.1 \
      )
      ;;
    rnn|lstm)
      model_args=( \
        --architecture $architecture \
        --num-layers 6 \
        --hidden-units 256 \
        --dropout 0.1 \
      )
      ;;
    superposition-stack-transformer)
      model_args=( \
        --architecture stack-transformer \
        --d-model 64 \
        --num-heads 8 \
        --feedforward-size 256 \
        --dropout 0.1 \
        --stack-transformer-layers 2.superposition-10.2 \
      )
      ;;
    nondeterministic-stack-transformer)
      model_args=( \
        --architecture stack-transformer \
        --d-model 64 \
        --num-heads 8 \
        --feedforward-size 256 \
        --dropout 0.1 \
        --stack-transformer-layers 2.nondeterministic-3-3-5.2
      )
      ;;
    *) exit 1 ;;
  esac
  model=$temp_dir/lm/models/$architecture

  rau lm train \
    --training-data $lm_data \
    "${model_args[@]}" \
    --init-scale 0.1 \
    --max-epochs 100 \
    --max-tokens-per-batch 2048 \
    --optimizer Adam \
    --initial-learning-rate 0.01 \
    --gradient-clipping-threshold 5 \
    --early-stopping-patience 2 \
    --learning-rate-patience 1 \
    --learning-rate-decay-factor 0.5 \
    --examples-per-checkpoint 50000 \
    --output $model

  rau lm evaluate \
    --load-model $model \
    --training-data $lm_data \
    --input test \
    --batching-max-tokens 2048

  rau lm generate \
    --load-model $model \
    --training-data $lm_data \
    --num-samples 10
done

ss_data=$temp_dir/ss/data
mkdir -p $ss_data
curl -s https://raw.githubusercontent.com/tommccoy1/rnn-hierarchical-biases/master/data/question.train > $ss_data/train.tsv
cut -f 1 < $ss_data/train.tsv > $ss_data/source.tok
cut -f 2 < $ss_data/train.tsv > $ss_data/target.tok
mkdir $ss_data/datasets
mkdir $ss_data/datasets/validation
curl -s https://raw.githubusercontent.com/tommccoy1/rnn-hierarchical-biases/master/data/question.dev > $ss_data/validation.tsv
cut -f 1 < $ss_data/validation.tsv > $ss_data/datasets/validation/source.tok
cut -f 2 < $ss_data/validation.tsv > $ss_data/datasets/validation/target.tok
mkdir $ss_data/datasets/test
curl -s https://raw.githubusercontent.com/tommccoy1/rnn-hierarchical-biases/master/data/question.test > $ss_data/test-full.tsv
head -100 $ss_data/test-full.tsv > $ss_data/test.tsv
rm $ss_data/test-full.tsv
cut -f 1 < $ss_data/test.tsv > $ss_data/datasets/test/source.tok
cut -f 2 < $ss_data/test.tsv > $ss_data/datasets/test/target.tok
rm $ss_data/{train,validation,test}.tsv

rau ss prepare \
  --training-data $ss_data \
  --vocabulary-types shared \
  --more-data validation \
  --more-source-data test \
  --never-allow-unk

architecture=transformer
model=$temp_dir/ss/models/$architecture

rau ss train \
  --training-data $ss_data \
  --vocabulary-type shared \
  --num-encoder-layers 6 \
  --num-decoder-layers 6 \
  --d-model 64 \
  --num-heads 8 \
  --feedforward-size 256 \
  --dropout 0.1 \
  --init-scale 0.1 \
  --max-epochs 100 \
  --max-tokens-per-batch 2048 \
  --optimizer Adam \
  --initial-learning-rate 0.01 \
  --label-smoothing-factor 0.1 \
  --gradient-clipping-threshold 5 \
  --early-stopping-patience 2 \
  --learning-rate-patience 1 \
  --learning-rate-decay-factor 0.5 \
  --examples-per-checkpoint 50000 \
  --output $model

rau ss translate \
  --load-model $model \
  --input $ss_data/datasets/test/source.shared.prepared \
  --beam-size 4 \
  --max-target-length 50 \
  --batching-max-tokens 256 \
  --shared-vocabulary-file $ss_data/shared.vocab

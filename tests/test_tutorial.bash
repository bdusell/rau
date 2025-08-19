set -euo pipefail

mode=${1-small}
case $mode in
  big)
    num_layers=6
    d_model=64
    num_heads=8
    feedforward_size=256
    hidden_units=256
    max_epochs=100
    max_length=50
    ;;
  small)
    num_layers=2
    d_model=32
    num_heads=4
    feedforward_size=64
    hidden_units=32
    max_epochs=1
    max_length=10
    ;;
  *) exit 1 ;;
esac
device_args=()

temp_dir=$(mktemp -d)
trap "rm -r -- $temp_dir" EXIT

lm_data=$temp_dir/lm/data
url=https://raw.githubusercontent.com/tommccoy1/rnn-hierarchical-biases/master/data
mkdir -p $lm_data
curl -s $url/question.train > $lm_data/main-full.tok
head -1000 $lm_data/main-full.tok > $lm_data/main.tok
rm $lm_data/main-full.tok
mkdir $lm_data/datasets
mkdir $lm_data/datasets/validation
curl -s $url/question.dev > $lm_data/datasets/validation/main.tok
mkdir $lm_data/datasets/test
curl -s $url/question.test > $lm_data/datasets/test/main.tok
mkdir $lm_data/datasets/test-source
head -10 $lm_data/datasets/test/main.tok | cut -f 1 > $lm_data/datasets/test-source/main.tok
mkdir $lm_data/datasets/test-target
head -10 $lm_data/datasets/test/main.tok | cut -f 2 > $lm_data/datasets/test-target/main.tok
mkdir $lm_data/datasets/generalization
curl -s $url/question.gen > $lm_data/datasets/generalization/main.tok
mkdir $lm_data/datasets/generalization-source
head -10 $lm_data/datasets/generalization/main.tok | cut -f 1 > $lm_data/datasets/generalization-source/main.tok
mkdir $lm_data/datasets/generalization-target
head -10 $lm_data/datasets/generalization/main.tok | cut -f 2 > $lm_data/datasets/generalization-target/main.tok

rau lm prepare \
  --training-data $lm_data \
  --more-data validation \
  --more-data test \
  --more-data test-source \
  --more-data test-target \
  --more-data generalization \
  --more-data generalization-source \
  --more-data generalization-target \
  --never-allow-unk

for architecture in \
  transformer \
  rnn \
  lstm \
  superposition-stack-transformer \
  nondeterministic-stack-transformer \
  superposition-stack-lstm \
  vector-nondeterministic-stack-lstm \
  ; do

  case $architecture in
    transformer)
      model_args=( \
        --architecture $architecture \
        --num-layers $num_layers \
        --d-model $d_model \
        --num-heads $num_heads \
        --feedforward-size $feedforward_size \
        --dropout 0.1 \
      )
      ;;
    rnn|lstm)
      model_args=( \
        --architecture $architecture \
        --num-layers $num_layers \
        --hidden-units $hidden_units \
        --dropout 0.1 \
      )
      ;;
    superposition-stack-transformer)
      model_args=( \
        --architecture stack-transformer \
        --d-model $d_model \
        --num-heads $num_heads \
        --feedforward-size $feedforward_size \
        --dropout 0.1 \
        --stack-transformer-layers 1.superposition-10.1 \
      )
      ;;
    nondeterministic-stack-transformer)
      model_args=( \
        --architecture stack-transformer \
        --d-model $d_model \
        --num-heads $num_heads \
        --feedforward-size $feedforward_size \
        --dropout 0.1 \
        --stack-transformer-layers 1.nondeterministic-2-3-2.1
      )
      ;;
    superposition-stack-lstm)
      model_args=( \
        --architecture stack-rnn \
        --num-layers $num_layers \
        --dropout 0.1 \
        --hidden-units $hidden_units \
        --stack-rnn-controller lstm \
        --stack-rnn-stack superposition-10 \
      )
      ;;
    vector-nondeterministic-stack-lstm)
      model_args=( \
        --architecture stack-rnn \
        --num-layers $num_layers \
        --dropout 0.1 \
        --hidden-units $hidden_units \
        --stack-rnn-controller lstm \
        --stack-rnn-stack vector-nondeterministic-2-3-2 \
      )
      ;;
    *) exit 1 ;;
  esac
  model=$temp_dir/lm/models/$architecture

  rau lm train \
    --training-data $lm_data \
    "${model_args[@]}" \
    --init-scale 0.1 \
    --max-epochs $max_epochs \
    --max-tokens-per-batch 2048 \
    --optimizer Adam \
    --initial-learning-rate 0.01 \
    --gradient-clipping-threshold 5 \
    --early-stopping-patience 2 \
    --learning-rate-patience 1 \
    --learning-rate-decay-factor 0.5 \
    --examples-per-checkpoint 500 \
    --output $model \
    "${device_args[@]}"

  rau lm evaluate \
    --load-model $model \
    --training-data $lm_data \
    --input test \
    --prompt-and-input test-{source,target} \
    --input generalization \
    --prompt-and-input generalization-{source,target} \
    --output $model/eval/cross-entropy \
    "${device_args[@]}"
  for d in test test-target generalization generalization-target; do
    echo $d
    cat $model/eval/cross-entropy/$d.txt
    echo
  done

  for granularity in position vocabulary logits; do
    rau lm evaluate \
      --load-model $model \
      --training-data $lm_data \
      --input test \
      --prompt-and-input test-{source,target} \
      --input generalization \
      --prompt-and-input generalization-{source,target} \
      --output $model/eval/$granularity \
      --granularity $granularity \
      "${device_args[@]}"
  done

  echo 'random'
  rau lm generate \
    --load-model $model \
    --training-data $lm_data \
    --num-samples 10 \
    --max-length $max_length \
    "${device_args[@]}"
  rau lm generate \
    --load-model $model \
    --training-data $lm_data \
    --prompt-datasets {test,generalization}-source \
    --num-samples 10 \
    --max-length $max_length \
    --output $model/eval/random \
    "${device_args[@]}"

  echo 'greedy'
  rau lm generate \
    --load-model $model \
    --training-data $lm_data \
    --mode greedy \
    --max-length $max_length \
    "${device_args[@]}"
  rau lm generate \
    --load-model $model \
    --training-data $lm_data \
    --prompt-datasets {test,generalization}-source \
    --mode greedy \
    --max-length $max_length \
    --output $model/eval/greedy \
    "${device_args[@]}"

  echo 'beam-search'
  rau lm generate \
    --load-model $model \
    --training-data $lm_data \
    --mode beam-search \
    --beam-size 4 \
    --max-length $max_length \
    "${device_args[@]}"
  rau lm generate \
    --load-model $model \
    --training-data $lm_data \
    --prompt-datasets {test,generalization}-source \
    --mode beam-search \
    --beam-size 4 \
    --max-length $max_length \
    --output $model/eval/beam-search \
    "${device_args[@]}"
done

ss_data=$temp_dir/ss/data
url=https://raw.githubusercontent.com/tommccoy1/rnn-hierarchical-biases/master/data
mkdir -p $ss_data
curl -s $url/question.train > $ss_data/train-full.tsv
head -1000 $ss_data/train-full.tsv > $ss_data/train.tsv
cut -f 1 < $ss_data/train.tsv > $ss_data/source.tok
cut -f 2 < $ss_data/train.tsv > $ss_data/target.tok
mkdir $ss_data/datasets
mkdir $ss_data/datasets/validation
curl -s $url/question.dev > $ss_data/validation.tsv
cut -f 1 < $ss_data/validation.tsv > $ss_data/datasets/validation/source.tok
cut -f 2 < $ss_data/validation.tsv > $ss_data/datasets/validation/target.tok
mkdir $ss_data/datasets/test
curl -s $url/question.test > $ss_data/test-full.tsv
head -100 $ss_data/test-full.tsv > $ss_data/test.tsv
cut -f 1 < $ss_data/test.tsv > $ss_data/datasets/test/source.tok
cut -f 2 < $ss_data/test.tsv > $ss_data/datasets/test/target.tok
rm $ss_data/{train,test}-full.tsv $ss_data/{train,validation,test}.tsv

rau ss prepare \
  --training-data $ss_data \
  --vocabulary-types shared \
  --more-data validation \
  --more-source-data test \
  --never-allow-unk

for architecture in \
  transformer \
  superposition-stack-transformer \
  nondeterministic-stack-transformer \
  ; do

  case $architecture in
    transformer)
      model_args=( \
        --architecture transformer \
        --num-encoder-layers $num_layers \
        --num-decoder-layers $num_layers \
      )
      ;;
    superposition-stack-transformer)
      model_args=( \
        --architecture stack-transformer \
        --stack-transformer-encoder-layers 1.superposition-10.1 \
        --stack-transformer-decoder-layers 1.superposition-10.1 \
      )
      ;;
    nondeterministic-stack-transformer)
      model_args=( \
        --architecture stack-transformer \
        --stack-transformer-encoder-layers 1.nondeterministic-2-3-2.1 \
        --stack-transformer-decoder-layers 1.nondeterministic-2-3-2.1 \
      )
      ;;
    *) exit 1 ;;
  esac
  model=$temp_dir/ss/models/$architecture

  rau ss train \
    --training-data $ss_data \
    --vocabulary-type shared \
    "${model_args[@]}" \
    --d-model $d_model \
    --num-heads $num_heads \
    --feedforward-size $feedforward_size \
    --dropout 0.1 \
    --init-scale 0.1 \
    --max-epochs $max_epochs \
    --max-tokens-per-batch 2048 \
    --optimizer Adam \
    --initial-learning-rate 0.01 \
    --label-smoothing-factor 0.1 \
    --gradient-clipping-threshold 5 \
    --early-stopping-patience 2 \
    --learning-rate-patience 1 \
    --learning-rate-decay-factor 0.5 \
    --examples-per-checkpoint 500 \
    --output $model \
    "${device_args[@]}"

  rau ss translate \
    --load-model $model \
    --input $ss_data/datasets/test/source.shared.prepared \
    --beam-size 4 \
    --max-target-length 50 \
    --batching-max-tokens 256 \
    --shared-vocabulary-file $ss_data/shared.vocab \
    "${device_args[@]}"
done

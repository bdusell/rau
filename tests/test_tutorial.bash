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
    d_model=8
    num_heads=2
    feedforward_size=8
    hidden_units=8
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
  {superposition,nondeterministic}-stack-transformer \
  {stratification,superposition,nondeterministic,vector-nondeterministic}-stack-{rnn,lstm}{,-r} \
  ; do
  if [[ $architecture = transformer ]]; then
    model_args=( \
      --architecture $architecture \
      --num-layers $num_layers \
      --d-model $d_model \
      --num-heads $num_heads \
      --feedforward-size $feedforward_size \
      --dropout 0.1 \
    )
  elif [[ $architecture =~ ^(rnn|lstm)$ ]]; then
    model_args=( \
      --architecture $architecture \
      --num-layers $num_layers \
      --hidden-units $hidden_units \
      --dropout 0.1 \
    )
  elif [[ $architecture =~ ^(.+)-stack-transformer$ ]]; then
    stack=${BASH_REMATCH[1]}
    if [[ $stack = superposition ]]; then
      stack_transformer_layers=1.superposition-10.1
    elif [[ $stack = nondeterministic ]]; then
      stack_transformer_layers=1.nondeterministic-2-2-2.1
    fi
    model_args=( \
      --architecture stack-transformer \
      --d-model $d_model \
      --num-heads $num_heads \
      --feedforward-size $feedforward_size \
      --dropout 0.1 \
      --stack-transformer-layers $stack_transformer_layers \
    )
  elif [[ $architecture =~ ^(.+)-stack-(rnn|lstm)(-r)?$ ]]; then
    stack=${BASH_REMATCH[1]}
    stack_rnn_controller=${BASH_REMATCH[2]}
    stack_reading=${BASH_REMATCH[3]}
    case $stack in
      stratification) stack_rnn_stack=stratification-10 ;;
      superposition) stack_rnn_stack=superposition-10 ;;
      nondeterministic) stack_rnn_stack=nondeterministic-2-2 ;;
      vector-nondeterministic) stack_rnn_stack=vector-nondeterministic-2-2-2 ;;
      *) exit 1 ;;
    esac
    model_args=( \
      --architecture stack-rnn \
      --num-layers $num_layers \
      --dropout 0.1 \
      --hidden-units $hidden_units \
      --stack-rnn-controller $stack_rnn_controller \
      --stack-rnn-stack $stack_rnn_stack \
    )
    if [[ $stack_reading ]]; then
      model_args+=(--stack-rnn-connect-reading-to-output)
    fi
  else
    echo "unknown architecture: $architecture" >&2
    exit 1
  fi
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

  # For the transformer only, test the --every-n-examples option.
  if [[ $architecture = transformer ]]; then
    rau lm train \
      --training-data $lm_data \
      --architecture $architecture \
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
      --examples-per-checkpoint 50000 \
      --output $temp_dir/lm/models/$architecture-every-n-examples \
      --every-n-examples 10000 'print("every 10000:", state.every_n_examples_no[index])' \
      --every-n-examples 20000 'print("every 20000:", state.every_n_examples_no[index])' \
      "${device_args[@]}"
  fi

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
        --stack-transformer-encoder-layers 1.nondeterministic-2-2-2.1 \
        --stack-transformer-decoder-layers 1.nondeterministic-2-2-2.1 \
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

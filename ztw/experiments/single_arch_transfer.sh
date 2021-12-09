#!/bin/bash
set -xe

declare -A num_heads

num_heads=( ["resnet56"]=27 ["vgg16bn"]=14 ["wideresnet32_4"]=15 ["mobilenet"]=13 ["tv_resnet"]=5 )

if [[ -n $3 ]]; then
  seed=$3
else
  seed=1666
fi
echo "seed: $seed"

if [ -z "$1" ] | [ -z "$2" ]; then
  echo "No arch or dataset passed!"
  exit 1
fi

arch=$1
dataset=$2
heads="${num_heads[$arch]}"
echo "arch: $arch dataset: $dataset heads: $heads"

base_net="${dataset}_${arch}_cnn"
cmd_base="python train_networks.py -a $arch -s $seed --heads all --skip_train_logits --save_test_logits"

#$cmd_base -t cnn --tag "${seed}_${base_net}" -d $dataset

for transfer_dataset in cifar10 cifar100 ; do
#  $cmd_base -t sdn_ic --head_arch conv sdn_pool -d $transfer_dataset \
#    --override_cnn_to_tune $base_net \
#    --tag "${seed}_${dataset}_${arch}_sdn_${dataset}_transfer_to_${transfer_dataset}" \
#    --suffix "transfer_from_${dataset}_new_ft_test"

#  $cmd_base -t sdn_ic --head_arch conv sdn_pool -d $transfer_dataset \
#    --override_cnn_to_tune $base_net --stacking --detach_prev \
#    --tag "${seed}_${dataset}_${arch}_sdn-ic-stacking_${dataset}_transfer_to_${transfer_dataset}" \
#    --suffix "transfer_from_${dataset}_sdn-ic-stacking"

  for head_id in $(seq 0 $heads); do
    $cmd_base \
      -d $transfer_dataset --override_cnn_to_tune $base_net \
      --head_arch conv sdn_pool \
      --stacking --detach_prev \
      --head_ids $head_id --alpha 0. \
      --tag "${seed}_${dataset}_${arch}_sdn-ic-ensb-${head_id}_${dataset}_transfer_to_${transfer_dataset}" \
      --heads all #&
#    if ! (( (head_id + 1) % 2)); then
#      wait
#    fi
  done
done

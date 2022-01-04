#!/usr/bin/env bash
set -xe

#arch="vgg16bn"
arch="tv_resnet"

cmd_base="python train_networks.py -a $arch -s $seed --skip_train_logits --save_test_logits -t cnn"

if [[ $arch == "tv_resnet" ]]; then
  cmd_base="$cmd_base --heads quarter --head_arch conv_less_ch sdn_pool -p 4 --lr_scaler 2"
  ztw_arg="--detach_norm layernorm"
else
  cmd_base="$cmd_base --heads all"
  ztw_arg=""
fi


for FROM_D in cifar10 cifar100;
do
  CNN_TO_TUNE=${FROM_D}_${arch}_cnn
  for TO_D in oct2017 cifar10 cifar100 ;
  do
    for UPTO in 2 4;
    do

      SUF=${FROM_D}_to_${TO_D}_upto_${UPTO}_rest_copied
      $CMD -d $TO_D --override_cnn_to_tune $CNN_TO_TUNE --freeze_cnn_up_to $UPTO --suffix $SUF --tag $SUF
	
      SUF=${FROM_D}_to_${TO_D}_upto_${UPTO}_rest_random
      $CMD -d $TO_D --override_cnn_to_tune $CNN_TO_TUNE --freeze_cnn_up_to $UPTO --override_cnn_up_to $UPTO --suffix $SUF --tag $SUF

      SUF=${FROM_D}_to_${TO_D}_upto_${UPTO}_rest_copied-nofreeze
      $CMD -d $TO_D --override_cnn_to_tune $CNN_TO_TUNE --suffix $SUF --tag $SUF

      SUF=${FROM_D}_to_${TO_D}_upto_${UPTO}_rest_random-nofreeze
      $CMD -d $TO_D --override_cnn_to_tune $CNN_TO_TUNE --override_cnn_up_to $UPTO --suffix $SUF --tag $SUF

    done
  done
done


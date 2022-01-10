#!/usr/bin/env bash
set -xe

seed=200

#ARCH="vgg16bn"
ARCH="tv_resnet"

CMD="python train_networks.py -a $ARCH -s $seed --skip_train_logits --save_test_logits -t cnn"

if [[ $ARCH == "tv_resnet" ]]; then
  CMD="$CMD --heads quarter --head_arch conv_less_ch sdn_pool -p 4 --lr_scaler 2"
  ztw_arg="--detach_norm layernorm"
else
  CMD="$CMD --heads all"
  ztw_arg=""
fi



for FROM_D in cifar10 cifar100;
do
  CNN_TO_TUNE=${FROM_D}_${ARCH}_cnn
  for TO_D in hymenoptera; #oct2017 cifar10 cifar100 ;
  do
    for UPTO in 1 2 4;
    do
      SUF_BASE=${FROM_D}_to_${TO_D}_${ARCH}_upto_${UPTO}

      SUF="${SUF_BASE}_rest_copied"
      $CMD -d $TO_D --override_cnn_to_tune $CNN_TO_TUNE --freeze_cnn_up_to $UPTO --suffix $SUF --tag $SUF
	
      SUF="${SUF_BASE}_rest_random"
      $CMD -d $TO_D --override_cnn_to_tune $CNN_TO_TUNE --freeze_cnn_up_to $UPTO --override_cnn_up_to $UPTO --suffix $SUF --tag $SUF

      SUF="${SUF_BASE}_rest_copied-nofreeze"
      $CMD -d $TO_D --override_cnn_to_tune $CNN_TO_TUNE --suffix $SUF --tag $SUF

      SUF="${SUF_BASE}_rest_random-nofreeze"
      $CMD -d $TO_D --override_cnn_to_tune $CNN_TO_TUNE --override_cnn_up_to $UPTO --suffix $SUF --tag $SUF

    done
  done
done


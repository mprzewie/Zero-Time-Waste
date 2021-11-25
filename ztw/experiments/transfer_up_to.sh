#!/usr/bin/env bash
set -xe

CMD="python train_networks.py -a vgg16bn -s 200 --heads all --skip_train_logits --save_test_logits -t cnn --relearn_final_layer
"

for FROM_D in cifar10 cifar100;
do
  for TO_D in cifar10 cifar100;
  do
    for UPTO in 2 4 6 8 10 12;
    do
      SUF=${FROM_D}_to_${TO_D}_upto_${UPTO}
      $CMD -d $TO_D --override_cnn_to_tune ${FROM_D}_vgg16bn_cnn --freeze_cnn_up_to $UPTO --suffix $SUF --tag $SUF
    done
  done
done


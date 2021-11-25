#!/usr/bin/env bash


CMD="python train_networks.py -a vgg16bn -s 200 --heads all --skip_train_logits --save_test_logits -t cnn --relearn_final_layer
"

for FROM_D in cifar10 cifar100;
do
  for TO_D in cifar10 cifar100;
  do
    $CMD -d TO_D --override_cnn_to_tune ${FROM_D}_vgg16bn_cnn --suffix ${FROM_D}_to_${TO_D} --tag trasfer_tryout_${FROM_D}_to_${TO_D}
  done

done


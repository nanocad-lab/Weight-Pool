#!/bin/bash

run_default()
{
    #Set the values
    network=${1:all}
    epochs=${2: }

    case $network in
      all)
        echo "all network will be executed"
        python3 resnet_10_cifar_prec_sweep_lut.py  
        python3 resnet_14_cifar_prec_sweep_lut.py  
        python3 resnet_mlperf_cifar_prec_sweep_lut.py  
        python3 tinyconv_qd_prec_sweep_lut.py  
        python3 mobilenetv2_qd_prec_sweep_lut.py  
        ;;

      resnet_10)
        echo "resent-10 will be executed"
        python3 resnet_10_cifar_prec_sweep_lut.py  
        ;;

      resnet_14)
        echo "resent-14 will be executed"
        python3 resnet_14_cifar_prec_sweep_lut.py  
        ;;

      resnet_mlperf)
        echo "resent-mlperf will be executed"
        python3 resnet_mlperf_cifar_prec_sweep_lut.py  
        ;;

      mobilenet_v2)
        echo "mobilenet-v2 will be executed"
        python3 mobilenetv2_qd_prec_sweep_lut.py  
        ;;

      tinyconv)
        echo "tinyconv will be executed"
        python3 tinyconv_qd_prec_sweep_lut.py  
        ;;
      *)
        echo "unknown network, program will quit"
        ;;
    esac
}

#need to enter the directory
cd fw_training
if [[ $# == 0 ]]; then
    echo "No options given, default values are used"
    run_default
elif [[ $# == 1 ]]; then
    echo "Number of epoch is not provided, default value will be used"
    run_default $1
else
    echo "too many arguments!"
fi
#return to the root directory
cd ..

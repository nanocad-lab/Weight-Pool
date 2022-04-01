#!/bin/bash
run_test()
{
    #Set the values
    network=${1:all}
    epochs=${2: }

    case $network in
      all)
        echo "all network will be executed"
        python3 cifar_resnet_10.py --epochs $epochs
        python3 cifar_resnet_14.py --epochs $epochs
        python3 cifar_resnet_mlperf.py --epochs $epochs
        python3 qd_mobilenet_v2.py --epochs $epochs
        python3 qd_tinyconv_bn.py --epochs $epochs
        ;;

      resnet_10)
        echo "resent-10 will be executed"
        python3 cifar_resnet_10.py --epochs $epochs
        ;;

      resnet_14)
        echo "resent-14 will be executed"
        python3 cifar_resnet_14.py --epochs $epochs
        ;;

      resnet_mlperf)
        echo "resent-mlperf will be executed"
        python3 cifar_resnet_mlperf.py --epochs $epochs
        ;;

      mobilenet_v2)
        echo "mobilenet-v2 will be executed"
        python3 qd_mobilenet_v2.py --epochs $epochs
        ;;

      tinyconv)
        echo "tinyconv will be executed"
        python3 qd_tinyconv_bn.py --epochs $epochs
        ;;
      *)
        echo "unknown network, program will quit"
        ;;
    esac
}

run_default()
{
    #Set the values
    network=${1:all}
    epochs=${2: }

    case $network in
      all)
        echo "all network will be executed"
        python3 cifar_resnet_10.py 
        python3 cifar_resnet_14.py  
        python3 cifar_resnet_mlperf.py  
        python3 qd_mobilenet_v2.py  
        python3 qd_tinyconv_bn.py  
        ;;

      resnet_10)
        echo "resent-10 will be executed"
        python3 cifar_resnet_10.py  
        ;;

      resnet_14)
        echo "resent-14 will be executed"
        python3 cifar_resnet_14.py  
        ;;

      resnet_mlperf)
        echo "resent-mlperf will be executed"
        python3 cifar_resnet_mlperf.py  
        ;;

      mobilenet_v2)
        echo "mobilenet-v2 will be executed"
        python3 qd_mobilenet_v2.py  
        ;;

      tinyconv)
        echo "tinyconv will be executed"
        python3 qd_tinyconv_bn.py  
        ;;
      *)
        echo "unknown network, program will quit"
        ;;
    esac
}

#need to enter the directory
cd original_model_training
if [[ $# == 0 ]]; then
    echo "No options given, default values are used"
    run_default
elif [[ $# == 1 ]]; then
    echo "Number of epoch is not provided, default value will be used"
    run_default $1
else
    echo "Number of epoch is provided, all network will be trained with the specified number of epochs"
    run_test $1 $2
fi
#return to the root directory
cd ..

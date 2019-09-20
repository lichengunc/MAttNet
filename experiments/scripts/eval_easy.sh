
GPU_ID=$1
DATASET=$2
SPLITBY=$3
ID=$4

case ${DATASET} in
    refcoco)
        for SPLIT in val testA testB
        do
            CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/eval_easy.py \
                --dataset ${DATASET} \
                --splitBy ${SPLITBY} \
                --split ${SPLIT} \
                --id ${ID}
        done
    ;;
    refcoco+)
        for SPLIT in val testA testB
        do
            CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/eval_easy.py \
                --dataset ${DATASET} \
                --splitBy ${SPLITBY} \
                --split ${SPLIT} \
                --id ${ID}
        done
    ;;
    refcocog)
        for SPLIT in val test
        do
            CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/eval_easy.py \
                --dataset ${DATASET} \
                --splitBy ${SPLITBY} \
                --split ${SPLIT} \
                --id ${ID}
        done
    ;;
esac
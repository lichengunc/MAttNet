

GPU_ID=$1
DATASET=$2
SPLITBY=$3

# IMDB="coco_minus_refer"
# ITERS=1150000
# TAG="notime"
# NET="res101"
ID="mrcn_cmr_with_st"

case ${DATASET} in
    refcoco)
        for SPLIT in val testA testB
        do
            python ./tools/eval_masks.py \
                --dataset ${DATASET} \
                --splitBy ${SPLITBY} \
                --split ${SPLIT} \
                --id ${ID}
        done
    ;;
    refcoco+)
        for SPLIT in val testA testB
        do
            python ./tools/eval_masks.py \
                --dataset ${DATASET} \
                --splitBy ${SPLITBY} \
                --split ${SPLIT} \
                --id ${ID}
        done
    ;;
    refcocog)
        for SPLIT in val test
        do
            python ./tools/eval_masks.py \
                --dataset ${DATASET} \
                --splitBy ${SPLITBY} \
                --split ${SPLIT} \
                --id ${ID}
        done
    ;;
esac
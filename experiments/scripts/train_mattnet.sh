

GPU_ID=$1
DATASET=$2
SPLITBY=$3
ID=$4

CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/train.py \
    --dataset ${DATASET} \
    --splitBy ${SPLITBY} \
    --max_iters 30000 \
    --with_st 1 \
    --id ${ID}

#!/bin/bash
for i in `seq 0 6`;
do
    OMP_NUM_THREADS=22 KMP_AFFINITY=scatter,granularity=fine th eval.lua -model model_id1-501-1448236541.t7_cpu.t7 -image_folder ~/coco_images/ -num_images $(( 2 ** ($i + 3) )) -batch_size $(( 2 ** $i )) -gpuid -1 -dump_images 0 > cpu_perf_data_$(( 2 ** $i)).txt
    cat cpu_perf_data_$(( 2 ** $i )).txt | grep '\(forward\|LSTM\)' | tail -n 14 | awk -v BATCH=$(( 2 ** $i )) -F ' |s' '{s+=$3} END {print BATCH, s/7/BATCH/1000}'
done

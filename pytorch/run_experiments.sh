#!/bin/bash

# 実験設定
DATASETS=("jurkat7" "jurkat4" "sysmex7" "sysmex4" "phenocam_monthly")
LOSSES=("ce" "msevl")
SEEDS=(0 1 2 3 4)
EPOCHS=100

# 実験数の計算
TOTAL=$((${#DATASETS[@]} * ${#LOSSES[@]} * ${#SEEDS[@]}))
COUNT=0

echo "=============================================="
echo "実験開始: ${#DATASETS[@]}データセット × ${#LOSSES[@]}損失関数 × ${#SEEDS[@]}シード = ${TOTAL}回"
echo "=============================================="

for dataset in "${DATASETS[@]}"; do
    for loss in "${LOSSES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            COUNT=$((COUNT + 1))
            echo ""
            echo "[${COUNT}/${TOTAL}] Dataset: ${dataset}, Loss: ${loss}, Seed: ${seed}"
            echo "----------------------------------------------"

            python pytorch/train.py \
                --dataset "$dataset" \
                --loss "$loss" \
                --epochs "$EPOCHS" \
                --seed "$seed"

            # エラーチェック
            if [ $? -ne 0 ]; then
                echo "ERROR: 実験が失敗しました (${dataset}, ${loss}, seed=${seed})"
                exit 1
            fi
        done
    done
done

echo ""
echo "=============================================="
echo "全${TOTAL}回の実験が完了しました"
echo "=============================================="

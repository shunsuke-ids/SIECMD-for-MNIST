#!/bin/bash

# 実験設定
DATASETS=("jurkat7" "jurkat4" "sysmex7" "sysmex4" "phenocam_monthly")
LOSSES=("ce" "msevl")
SEEDS=(0 1 2 3 4)
EPOCHS=100

# 結果ファイル
RESULT_FILE="experiment_results.csv"
echo "dataset,loss,seed,best_acc,best_cmae,final_acc,final_cmae" > "$RESULT_FILE"

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

            # 実験実行、出力をキャプチャ
            OUTPUT=$(python pytorch/train.py \
                --dataset "$dataset" \
                --loss "$loss" \
                --epochs "$EPOCHS" \
                --seed "$seed" 2>&1)

            # エラーチェック
            if [ $? -ne 0 ]; then
                echo "ERROR: 実験が失敗しました (${dataset}, ${loss}, seed=${seed})"
                echo "$OUTPUT"
                exit 1
            fi

            echo "$OUTPUT"

            # 結果をパース
            BEST_LINE=$(echo "$OUTPUT" | grep "RESULT_BEST:")
            FINAL_LINE=$(echo "$OUTPUT" | grep "RESULT_FINAL:")

            BEST_ACC=$(echo "$BEST_LINE" | sed 's/.*acc=\([0-9.]*\).*/\1/')
            BEST_CMAE=$(echo "$BEST_LINE" | sed 's/.*cmae=\([0-9.]*\).*/\1/')
            FINAL_ACC=$(echo "$FINAL_LINE" | sed 's/.*acc=\([0-9.]*\).*/\1/')
            FINAL_CMAE=$(echo "$FINAL_LINE" | sed 's/.*cmae=\([0-9.]*\).*/\1/')

            # CSVに追記
            echo "${dataset},${loss},${seed},${BEST_ACC},${BEST_CMAE},${FINAL_ACC},${FINAL_CMAE}" >> "$RESULT_FILE"
        done
    done
done

echo ""
echo "=============================================="
echo "全${TOTAL}回の実験が完了しました"
echo "=============================================="
echo ""
echo "結果集計 (mean ± std)"
echo "=============================================="

# 集計（awkで計算）
for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "[$dataset]"
    for loss in "${LOSSES[@]}"; do
        # Best Model
        STATS=$(awk -F',' -v ds="$dataset" -v ls="$loss" '
            $1==ds && $2==ls {
                acc[NR]=$4; cmae[NR]=$5; n++
            }
            END {
                if(n==0) exit
                # mean
                sum_acc=0; sum_cmae=0
                for(i in acc) { sum_acc+=acc[i]; sum_cmae+=cmae[i] }
                mean_acc=sum_acc/n; mean_cmae=sum_cmae/n
                # std
                ss_acc=0; ss_cmae=0
                for(i in acc) { ss_acc+=(acc[i]-mean_acc)^2; ss_cmae+=(cmae[i]-mean_cmae)^2 }
                std_acc=sqrt(ss_acc/n); std_cmae=sqrt(ss_cmae/n)
                printf "%.4f %.4f %.4f %.4f", mean_acc, std_acc, mean_cmae, std_cmae
            }
        ' "$RESULT_FILE")

        BEST_ACC_MEAN=$(echo $STATS | cut -d' ' -f1)
        BEST_ACC_STD=$(echo $STATS | cut -d' ' -f2)
        BEST_CMAE_MEAN=$(echo $STATS | cut -d' ' -f3)
        BEST_CMAE_STD=$(echo $STATS | cut -d' ' -f4)

        # Final Model
        STATS_FINAL=$(awk -F',' -v ds="$dataset" -v ls="$loss" '
            $1==ds && $2==ls {
                acc[NR]=$6; cmae[NR]=$7; n++
            }
            END {
                if(n==0) exit
                sum_acc=0; sum_cmae=0
                for(i in acc) { sum_acc+=acc[i]; sum_cmae+=cmae[i] }
                mean_acc=sum_acc/n; mean_cmae=sum_cmae/n
                ss_acc=0; ss_cmae=0
                for(i in acc) { ss_acc+=(acc[i]-mean_acc)^2; ss_cmae+=(cmae[i]-mean_cmae)^2 }
                std_acc=sqrt(ss_acc/n); std_cmae=sqrt(ss_cmae/n)
                printf "%.4f %.4f %.4f %.4f", mean_acc, std_acc, mean_cmae, std_cmae
            }
        ' "$RESULT_FILE")

        FINAL_ACC_MEAN=$(echo $STATS_FINAL | cut -d' ' -f1)
        FINAL_ACC_STD=$(echo $STATS_FINAL | cut -d' ' -f2)
        FINAL_CMAE_MEAN=$(echo $STATS_FINAL | cut -d' ' -f3)
        FINAL_CMAE_STD=$(echo $STATS_FINAL | cut -d' ' -f4)

        printf "  %-6s [Best]  Acc=%s±%s, cMAE=%s±%s\n" "${loss^^}" "$BEST_ACC_MEAN" "$BEST_ACC_STD" "$BEST_CMAE_MEAN" "$BEST_CMAE_STD"
        printf "         [Final] Acc=%s±%s, cMAE=%s±%s\n" "$FINAL_ACC_MEAN" "$FINAL_ACC_STD" "$FINAL_CMAE_MEAN" "$FINAL_CMAE_STD"
    done
done

echo ""
echo "詳細結果: $RESULT_FILE"

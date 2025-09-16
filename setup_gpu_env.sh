#!/bin/bash

# GPU対応機械学習環境セットアップ用スクリプト
# 使用方法: source setup_gpu_env.sh

echo "=== GPU対応機械学習環境をセットアップ中 ==="

# conda環境をアクティベート
if command -v conda &> /dev/null; then
    echo "conda環境 'mnist-ml' をアクティベート中..."
    conda activate mnist-ml
    
    # GPU動作確認
    echo "GPU動作確認中..."
    python3 -c "
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print(f'✓ GPU {len(gpus)}台が利用可能です')
    for i, gpu in enumerate(gpus):
        print(f'  GPU {i}: {gpu.name}')
else:
    print('⚠ GPUが見つかりません。CPUで実行されます。')
print(f'TensorFlow version: {tf.__version__}')
print(f'CUDA support: {tf.test.is_built_with_cuda()}')
"
    
    echo ""
    echo "=== セットアップ完了 ==="
    echo "使用方法:"
    echo "  cd /home/shunsuke/lab/mnistSIECMD"
    echo "  python3 src/regression/fine_tuning.py [引数]"
    echo ""
    echo "GPU使用率の監視:"
    echo "  watch -n 1 nvidia-smi"
    echo ""
else
    echo "❌ condaが見つかりません。先にcondaをインストールしてください。"
    exit 1
fi
export PATH=/mnt/lustre/geyuying/anaconda3_new/envs/pytorch_new/bin:/mnt/lustre/share/cuda-9.0/bin:$PATH
export LD_LIBRARY_PATH=/mnt/lustre/share/cuda-9.0/lib64/:$LD_LIBRARY_PATH

if [ ! -d "log" ]; then
  mkdir log
fi

jobname=cal_ssim
num_gpus=1

srun -p sensevideo --job-name=$jobname --gres=gpu:$num_gpus  \
python test.py \

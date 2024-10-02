# 交互式命令，适合debug，因为Max Walltime只有8小时
srun -p interactive -J test --gres=gpu:1 --mem-per-cpu=32G --cpus-per-task 4 --time 01:00:00 --pty bash -i
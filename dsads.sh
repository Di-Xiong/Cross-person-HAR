#! /bin/bash
/usr/bin/python3 train.py --algorithm="ERM" --dataset="dsads" --gpu_id =0 --test_envs=0 
/usr/bin/python3 train.py --algorithm="ERM" --dataset="dsads" --gpu_id =0 --test_envs=1 
/usr/bin/python3 train.py --algorithm="ERM" --dataset="dsads" --gpu_id =0 --test_envs=2 
/usr/bin/python3 train.py --algorithm="ERM" --dataset="dsads" --gpu_id =0 --test_envs=3 

/usr/bin/python3 train.py --algorithm="Our" --dataset="dsads" --gpu_id =0 --test_envs=0 
/usr/bin/python3 train.py --algorithm="Our" --dataset="dsads" --gpu_id =0 --test_envs=1 
/usr/bin/python3 train.py --algorithm="Our" --dataset="dsads" --gpu_id =0 --test_envs=2 
/usr/bin/python3 train.py --algorithm="Our" --dataset="dsads" --gpu_id =0 --test_envs=3 

/usr/bin/python3 train.py --algorithm="Fixed" --dataset="dsads" --gpu_id =0 --test_envs=0 
/usr/bin/python3 train.py --algorithm="Fixed" --dataset="dsads" --gpu_id =0 --test_envs=1 
/usr/bin/python3 train.py --algorithm="Fixed" --dataset="dsads" --gpu_id =0 --test_envs=2 
/usr/bin/python3 train.py --algorithm="Fixed" --dataset="dsads" --gpu_id =0 --test_envs=3 




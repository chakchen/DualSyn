/home/.conda/envs/DualSyn/bin/python /home/DualSyn/train_leave_out.py --leave_type leave_drug --dropping_method DropNode --dropout_rate 0.1 --device_num 1
/home/.conda/envs/DualSyn/bin/python /home/DualSyn/train_leave_out.py --leave_type leave_comb --dropping_method DropNode --dropout_rate 0.1 --device_num 1
/home/.conda/envs/DualSyn/bin/python /home/DualSyn/train_leave_out.py --leave_type leave_cell --dropping_method DropNode --dropout_rate 0.1 --device_num 1

/home/.conda/envs/DualSyn/bin/python /home/DualSyn/train_independent.py --dropping_method DropNode --dropout_rate 0.2 --device_num 1 --lr 0.000005

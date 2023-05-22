dataset='CSC-NDGS'
task='img_dg'
data_dir='/data/sihan.zhu/myfile/dataset/CSC-NDGS/'
max_epoch=100
lr=1e-3
algorithm='SCSN'
net='resnet18_scsn'
#net='resnet50_scsn'
test_envs=0
gpu_ids='1'
output='./record_resnet18/scsn_test/domain111'
python train.py --data_dir $data_dir --gpu_id $gpu_ids --lr $lr --max_epoch $max_epoch --net $net \
--task $task --output $output --test_envs $test_envs --dataset $dataset --algorithm $algorithm
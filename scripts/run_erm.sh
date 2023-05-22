dataset='CSC-NDGS'
task='img_dg'
data_dir='/data/sihan.zhu/myfile/dataset/CSC-NDGS/'
max_epoch=100
lr=1e-3
algorithm='ERM'
net='resnet18'
#net='resnet50'
test_envs=0
gpu_ids='2'
output='./record_resnet50/erm_imp/domain1'
python train.py --data_dir $data_dir --gpu_id $gpu_ids --lr $lr --max_epoch $max_epoch --net $net \
--task $task --output $output --test_envs $test_envs --dataset $dataset --algorithm $algorithm
dataset='CSC-NDGS'
algorithm='SCSN'
#algorithm='ERM'
# ERM SCSN
test_envs=0
gpu_ids='0'
data_dir='/data/sihan.zhu/myfile/dataset/CSC-NDGS/'
max_epoch=200
net='resnet18-scsn'
#net='resnet18'
# resnet18  resnet18-scsn
task='img_dg'
output='./resnet18_SCSN/domain1'
#output='./resnet18_ERM/domain1'
#python cam_erm.py --data_dir $data_dir --gpu_id $gpu_ids --max_epoch $max_epoch --net $net --task $task --output $output \
#--test_envs $test_envs --dataset $dataset --algorithm $algorithm --matpath $matpath
#python cam_scsn.py --data_dir $data_dir --gpu_id $gpu_ids --max_epoch $max_epoch --net $net --task $task --output $output \
#--test_envs $test_envs --dataset $dataset --algorithm $algorithm --matpath $matpath
python infer.py --data_dir $data_dir --gpu_id $gpu_ids --max_epoch $max_epoch --net $net --task $task --output $output \
--test_envs $test_envs --dataset $dataset --algorithm $algorithm --matpath $matpath

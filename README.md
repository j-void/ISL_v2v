ISL - HAND SK

**run -- 
python train_refine.py --name slv07_rz_hand_refine --dataroot ../edn_data/slv07_rz/train/ --checkpoints_dir checkpoints/ --load_pretrain checkpoints/slv07_rz_hand_sgen2_d4/ --netG local --refine --shand_gen --shandGtype global  --ngf 32 --num_D 1 --resize_or_crop none --no_instance --no_flip --tf_log --label_nc 6


python test_refine.py --name slv07_rz_hand_refine_nc --dataroot ../edn_data/surveyvids/sentences_1/09/ --checkpoints_dir checkpoints/ --results_dir ../edn_data/surveyvids/sentences_1/09/ --shand_gen --shandGtype global --netG local --ngf 32 --no_instance --resize_or_crop none --how_many 10000 --label_nc 6


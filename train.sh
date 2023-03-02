#!/usr/bin/env bashclear

#### Synthetic to Real
### 7-class setting warmup
python train.py --name warmup_G2CI --stage warm_up --freeze_bn --gan Vanilla --lr 2.5e-4 --adv 0.001 --no_resume
python train.py --name warmup_G2CM --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset cityscapes mapillary --tgt_rootpath Dataset/Cityscapes Dataset/Mapillary --stage warm_up --freeze_bn --gan Vanilla --lr 2.5e-4 --adv 0.001 --no_resume
python train.py --name warmup_G2MI --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset mapillary idd --tgt_rootpath Dataset/Mapillary Dataset/IDD_Segmentation --stage warm_up --freeze_bn --gan Vanilla --lr 2.5e-4 --adv 0.001 --no_resume
python train.py --name warmup_G2CMI --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset cityscapes mapillary idd --tgt_rootpath Dataset/Cityscapes Dataset/Mapillary Dataset/IDD_Segmentation --stage warm_up --freeze_bn --gan Vanilla --lr 2.5e-4 --adv 0.001 --no_resume

### 19-class setting warmup
python train.py --name warmup_G2CI_19 --n_class 19 --img_size '1024,512' --resize 1024 --rcrop '512,256' --stage warm_up --freeze_bn --gan Vanilla --lr 2.5e-4 --adv 0.001 --no_resume
python train.py --name warmup_G2CM_19 --n_class 19 --img_size '1024,512' --resize 1024 --rcrop '512,256' --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset cityscapes mapillary --tgt_rootpath Dataset/Cityscapes Dataset/Mapillary --stage warm_up --freeze_bn --gan Vanilla --lr 2.5e-4 --adv 0.001 --no_resume
python train.py --name warmup_G2MI_19 --n_class 19 --img_size '1024,512' --resize 1024 --rcrop '512,256' --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset mapillary idd --tgt_rootpath Dataset/Mapillary Dataset/IDD_Segmentation --stage warm_up --freeze_bn --gan Vanilla --lr 2.5e-4 --adv 0.001 --no_resume
python train.py --name warmup_G2CMI_19 --n_class 19 --img_size '1024,512' --resize 1024 --rcrop '512,256' --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset cityscapes mapillary idd --tgt_rootpath Dataset/Cityscapes Dataset/Mapillary Dataset/IDD_Segmentation --stage warm_up --freeze_bn --gan Vanilla --lr 2.5e-4 --adv 0.001 --no_resume


### 7-class setting stage1: self-training
python train.py --name stage1_G2CI --used_save_pseudo --rectify --resume_path ./logs/warmup_G2CI/from_gta5_to_2_on_deeplabv2_best_model.pkl
python train.py --name stage1_G2CM --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset cityscapes mapillary --tgt_rootpath Dataset/Cityscapes Dataset/Mapillary --used_save_pseudo --rectify --resume_path ./logs/warmup_G2CM/from_gta5_to_2_on_deeplabv2_best_model.pkl
python train.py --name stage1_G2MI --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset mapillary idd --tgt_rootpath Dataset/Mapillary Dataset/IDD_Segmentation --used_save_pseudo --rectify --resume_path ./logs/warmup_G2MI/from_gta5_to_2_on_deeplabv2_best_model.pkl
python train.py --name stage1_G2CMI --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset cityscapes mapillary idd --tgt_rootpath Dataset/Cityscapes Dataset/Mapillary Dataset/IDD_Segmentation --used_save_pseudo --rectify --resume_path ./logs/warmup_G2CMI/from_gta5_to_3_on_deeplabv2_best_model.pkl

### 19-class setting stage1: self-training
python train.py --name stage1_G2CI_19 --n_class 19 --img_size '1024,512' --resize 1024 --rcrop '512,256' --used_save_pseudo --rectify --resume_path ./logs/warmup_G2CI_19/from_gta5_to_2_on_deeplabv2_best_model.pkl
python train.py --name stage1_G2CM_19 --n_class 19 --img_size '1024,512' --resize 1024 --rcrop '512,256' --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset cityscapes mapillary --tgt_rootpath Dataset/Cityscapes Dataset/Mapillary --used_save_pseudo --rectify --resume_path ./logs/warmup_G2CM_19/from_gta5_to_2_on_deeplabv2_best_model.pkl
python train.py --name stage1_G2MI_19 --n_class 19 --img_size '1024,512' --resize 1024 --rcrop '512,256' --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset mapillary idd --tgt_rootpath Dataset/Mapillary Dataset/IDD_Segmentation --used_save_pseudo --rectify --resume_path ./logs/warmup_G2MI_19/from_gta5_to_2_on_deeplabv2_best_model.pkl
python train.py --name stage1_G2CMI_19 --n_class 19 --img_size '1024,512' --resize 1024 --rcrop '512,256' --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset cityscapes mapillary idd --tgt_rootpath Dataset/Cityscapes Dataset/Mapillary Dataset/IDD_Segmentation --used_save_pseudo --rectify --resume_path ./logs/warmup_G2CMI_19/from_gta5_to_3_on_deeplabv2_best_model.pkl

### evaluate 7 class
python test.py --bs 1 --stage stage1 --resume_path ./logs/stage1_G2CI/from_gta5_to_2_on_deeplabv2_best_model.pkl
python test.py --bs 1 --stage stage1 --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset cityscapes mapillary --tgt_rootpath Dataset/Cityscapes Dataset/Mapillary --resume_path ./logs/stage1_G2CM/from_gta5_to_2_on_deeplabv2_best_model.pkl
python test.py --bs 1 --stage stage1 --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset mapillary idd --tgt_rootpath Dataset/Mapillary Dataset/IDD_Segmentation --resume_path ./logs/stage1_G2MI/from_gta5_to_2_on_deeplabv2_best_model.pkl
python test.py --bs 1 --stage stage1 --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset cityscapes mapillary idd --tgt_rootpath Dataset/Cityscapes Dataset/Mapillary Dataset/IDD_Segmentation --resume_path ./logs/stage1_G2CMI/from_gta5_to_3_on_deeplabv2_best_model.pkl

### evaluate 19 class
python test.py --bs 1 --n_class 19 --img_size '1024,512' --resize 1024 --rcrop '512,256' --stage stage1 --resume_path ./logs/stage1_G2CI_19/from_gta5_to_2_on_deeplabv2_best_model.pkl
python test.py --bs 1 --n_class 19 --img_size '1024,512' --resize 1024 --rcrop '512,256' --stage stage1 --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset cityscapes mapillary --tgt_rootpath Dataset/Cityscapes Dataset/Mapillary --resume_path ./logs/stage1_G2CM_19/from_gta5_to_2_on_deeplabv2_best_model.pkl
python test.py --bs 1 --n_class 19 --img_size '1024,512' --resize 1024 --rcrop '512,256' --stage stage1 --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset mapillary idd --tgt_rootpath Dataset/Mapillary Dataset/IDD_Segmentation --resume_path ./logs/stage1_G2MI_19/from_gta5_to_2_on_deeplabv2_best_model.pkl
python test.py --bs 1 --n_class 19 --img_size '1024,512' --resize 1024 --rcrop '512,256' --stage stage1 --src_dataset gta5 --src_rootpath Dataset/GTA5 --tgt_dataset cityscapes mapillary idd --tgt_rootpath Dataset/Cityscapes Dataset/Mapillary Dataset/IDD_Segmentation --resume_path ./logs/stage1_G2CMI_19/from_gta5_to_3_on_deeplabv2_best_model.pkl





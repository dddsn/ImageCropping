python eval.py --dataset_path /lfs1/data --weight weights/gan_weights.h5 --log eval_gan.txt
python demo.py --weight weights/gan_weights.h5 --img_path images --save_path result --log True
rlaunch --cpu 32 --gpu 8 --memory 96000 -- python3 train_net.py -d 8
rlaunch --cpu 32 --gpu 8 --memory 120000 -- python3 testing.py -d 8 -s 30 e 36
rlaunch --cpu 32 --memory 96000 -- python3 demo.py

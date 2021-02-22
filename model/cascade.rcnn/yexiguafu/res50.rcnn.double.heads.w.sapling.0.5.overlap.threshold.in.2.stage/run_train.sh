rlaunch --cpu 32 --gpu 8 --memory 96000 -- python3 train_net.py -d 8
rlaunch --cpu 32 --gpu 8 --memory 120000 -- python3 test_net.py -d 0-7 -s 28 -e 32
rlaunch --cpu 32 --memory 96000 -- python3 demo.py

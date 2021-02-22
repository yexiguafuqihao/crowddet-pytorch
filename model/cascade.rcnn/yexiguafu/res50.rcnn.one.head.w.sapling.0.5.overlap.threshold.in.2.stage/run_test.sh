rlaunch --cpu 32 --gpu 8 --memory 96000  -- python3 test_net.py   \
	-d 0-7 -s 6 -e 40
rlaunch --cpu 32 --memory 120000 -- python3 demo.py

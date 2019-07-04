#!/bin/bash
shopt -s expand_aliases
source ~/.bashrc
# py36="/usr/bin/python3.6"

# pre_train T_net 
py36 train.py \
	--train_list="./data/id_20190418/train_names.txt" \
	--fg_path="./data/id_20190418/anotation" \
	--bg_path="./data/id_20190418/fg" \
	--alpha_apth='./data/id_20190418/alpha' \
	--saveDir='./ckpt' \
	--trainData='human_matting_data' \
	--nThreads=4 \
	--train_batch=8 \
	--patch_size=400 \
	--lr=1e-2 \
	--lrDecay=100 \
	--lrdecayType='poly' \
	--nEpochs=100 \
	--save_epoch=1 \
	--print_iter=1000 \
	--train_phase='pre_train_t_net' \
	#--debug=False \
	#--without_gpu=False \
	#--pretrain=False \
	#--continue_train=False \
	# --fgLists =["./data/id_20190415/train_names.txt"] \
	# --bg_list= "./data/id_20190415/train_names.txt"\
	# --dataRatio = [1]\
	#--dataDir='./data' \

# pre_train M_net

# python3 train.py \
# 	--train_list = "./data/id_20190415/train_names.txt" \
# 	--fg_path = "./data/id_20190415/transparent"\
# 	--bg_apth = "./data/id_20190415/bg" \
# 	--alpha_apth = './data/id_20190415/alpha' \
# 	--saveDir='./ckpt' \
# 	--trainData='human_matting_data' \
# 	--continue_train = False\
# 	--pretrain = True\
# 	--without_gpu = False\
# 	--nThreads =4 \
# 	--train_batch=8 \
# 	--patch_size=320 \
# 	--lr=1e-5 \
# 	--lrDecay=100 \
# 	--lrdecayType= 'keep'\
# 	--nEpochs=300 \
# 	--save_epoch= 1\
# 	--print_iter=1000 \
# 	--train_phase='pre_train_m_net' \
# 	--debug=False \
# 	# --fgLists =["./data/id_20190415/train_names.txt"] \
# 	# --bg_list= "./data/id_20190415/train_names.txt"\
# 	# --dataRatio = [1]\
# 	#--dataDir='./data' \


# train end to end
# python3 train.py \
# 	--train_list = "./data/id_20190415/train_names.txt" \
# 	--fg_path = "./data/id_20190415/transparent"\
# 	--bg_apth = "./data/id_20190415/bg" \
# 	--alpha_apth = './data/id_20190415/alpha' \
# 	--saveDir='./ckpt' \
# 	--trainData='human_matting_data' \
# 	--continue_train = False\
# 	--pretrain = True\
# 	--without_gpu = False\
# 	--nThreads =4 \
# 	--train_batch=8 \
# 	--patch_size=800 \
# 	--lr=1e-5 \
# 	--lrDecay=100 \
# 	--lrdecayType= 'keep'\
# 	--nEpochs=300 \
# 	--save_epoch= 1\
# 	--print_iter=1000 \
# 	--train_phase='end_to_end' \
# 	--debug=False \
# 	# --fgLists =["./data/id_20190415/train_names.txt"] \
# 	# --bg_list= "./data/id_20190415/train_names.txt"\
# 	# --dataRatio = [1]\
# 	#--dataDir='./data' \


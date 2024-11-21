
# FMNIST, ACC @ 88.08%, increase training round may yeild better accuracy.
CUDA_VISIBLE_DEVICES=7 python federated.py --aggr avg --data fmnist --num_agents 6000 --bs 10 --local_ep 5 --round 500

# SVHN, ACC @ 91.12%, increase training round may yeild better accuracy.
CUDA_VISIBLE_DEVICES=7 python federated.py --aggr avg --data svhn --num_agents 1000 --bs 50 --local_ep 5 --round 500

# CIFAR-10, ACC @ 87.20%, increase training round may yeild better accuracy.
# For this setting, need 7 hours to train the model.
CUDA_VISIBLE_DEVICES=7 python federated.py --aggr avg --data cifar10 --num_agents 1000 --bs 50 --local_ep 5 --round 1000 --client_lr 0.01 --momentum 0.9

# CIFAR-100, ACC @ 60.80%, increase training round may yeild better accuracy.
# For this setting, need 15 hours to train the model.
CUDA_VISIBLE_DEVICES=7 python federated.py --aggr avg --data cifar100 --num_agents 500 --bs 50 --local_ep 5 --round 1000 --client_lr 0.01 --momentum 0.9



#######

# --aggr, aggregation rule
# --data, dataset
# --num_agents, number of clients
# --bs, batch size
# --local_ep, local training epoch
# --round, global training round
# --client_lr, local learning rate for each client
# --momentum, local momentum for optimizor

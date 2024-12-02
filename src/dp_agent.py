import time

import torch
import utils
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader


class DPAgent():
    def __init__(self, id, args, num_selected_clients=None, train_dataset=None, data_idxs=None):
        self.id = id
        self.args = args
        self.error = 0
        self.hessian_metrix = []
        # update the args setup to add these two arguments
        # self.noise_multiplier = args.noise_multiplier
        # self.l2_norm_clip = args.l2_norm_clip 
        self.num_selected_clients = num_selected_clients # this is the number of selected clients within this group

        self.train_dataset = utils.DatasetSplit(train_dataset, data_idxs)
        # print(len(self.train_dataset))

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.bs, shuffle=True, \
                                       num_workers=args.num_workers, pin_memory=False, drop_last=True)
        # size of local dataset
        self.n_data = len(self.train_dataset)

    def local_train(self, global_model, criterion, round=None):
        """ Do a local training over the received global model, return the update """
        initial_global_model_params = parameters_to_vector(
            [global_model.state_dict()[name] for name in global_model.state_dict()]).detach()

        global_model.train()
        current_lr = self.args.client_lr
        # optimizer = torch.optim.SGD(global_model.parameters(), lr=current_lr * (self.args.lr_decay) ** round,
        #                             weight_decay=self.args.wd, momentum=self.args.momentum)
        optimizer = torch.optim.SGD(global_model.parameters(), lr=current_lr,
                                    weight_decay=self.args.wd, momentum=self.args.momentum)

        for local_epoch in range(self.args.local_ep):
            start = time.time()
            for i, (inputs, labels) in enumerate(self.train_loader):
                
                optimizer.zero_grad()
                inputs, labels = inputs.to(device=self.args.device, non_blocking=True), \
                                 labels.to(device=self.args.device, non_blocking=True)
                outputs = global_model(inputs)
                minibatch_loss = criterion(outputs, labels)
                minibatch_loss.backward()
                optimizer.step()

            end = time.time()
            train_time = end - start
            print("local epoch %d \t client: %d \t loss: %.8f \t time: %.2f" % (local_epoch, self.id,
                                                                     minibatch_loss, train_time))

        with torch.no_grad():
            after_train = parameters_to_vector(
                [global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
            update = after_train - initial_global_model_params

            # Norm clipping to bound sensitivity
            l2_norm = torch.norm(update, p=2)
            if l2_norm > self.args.l2_norm_clip:
                update = update * (self.args.l2_norm_clip / l2_norm)

            # Add Gaussian noise for DP
            noise = torch.normal(
                mean=0,
                std=self.args.noise_multiplier * self.args.l2_norm_clip/torch.sqrt(self.num_selected_clients),
                size=update.size(),
                device=self.args.device
            )
            self.update = update + noise

            return self.update
    

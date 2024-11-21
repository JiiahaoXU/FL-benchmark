import copy

import torch
from torch.nn.utils import parameters_to_vector
import numpy as np
from copy import deepcopy
import logging
from utils import vector_to_model, vector_to_name_param
from torchvision import datasets, transforms


class Aggregation():
    def __init__(self, agent_data_sizes, args, n_params):
        self.agent_data_sizes = agent_data_sizes
        self.args = args
        self.server_lr = args.server_lr
        self.n_params = n_params
        
         
    def aggregate_updates(self, global_model, agent_updates_dict):

        
        lr_vector = torch.Tensor([self.server_lr]*self.n_params).to(self.args.device)
        aggregated_updates = 0
        cur_global_params = parameters_to_vector(
            [global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
        if self.args.aggr=='avg':    
            aggregated_updates = self.agg_avg(agent_updates_dict)

        updates_dict = vector_to_name_param(aggregated_updates, copy.deepcopy(global_model.state_dict()))

        cur_global_params = parameters_to_vector([ global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
        new_global_params =  (cur_global_params + lr_vector*aggregated_updates).float()
        vector_to_model(new_global_params, global_model)
        return updates_dict

    def agg_avg(self, agent_updates_dict):
        """ classic fed avg """

        sm_updates, total_data = 0, 0
        for _id, update in agent_updates_dict.items():
            n_agent_data = self.agent_data_sizes[_id]
            sm_updates +=  n_agent_data * update
            total_data += n_agent_data
        return  sm_updates / total_data

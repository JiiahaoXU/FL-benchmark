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
        if self.agrs.aggr=='hier_avg':
            aggregated_updates = self.hier_avg(agent_updates_dict)
        if self.agrs.aggr=='agg_group_avg':
            aggregated_updates = self.agg_group_avg(agent_updates_dict)
            

        updates_dict = vector_to_name_param(aggregated_updates, copy.deepcopy(global_model.state_dict()))

        cur_global_params = parameters_to_vector([ global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
        new_global_params =  (cur_global_params + lr_vector*aggregated_updates).float()
        vector_to_model(new_global_params, global_model)
        return updates_dict

    def agg_avg(self, agent_updates_dict):
        """ classic fed avg: Weighted average based on data size."""

        sm_updates, total_data = 0, 0
        for _id, update in agent_updates_dict.items():
            n_agent_data = self.agent_data_sizes[_id]
            sm_updates +=  n_agent_data * update
            total_data += n_agent_data
        return  sm_updates / total_data
    
    def agg_group_avg(self, group_updates_dict):
        """
        Group Mean Average: Average updates within each group, then compute the average over groups. (this aggregation treats groups evenly)
        
        
        Parameters:
            group_updates_dict (dict): A dictionary where each key is a group ID, 
                                    and the value is another dictionary of agent updates in that group.
        
        Returns:
            float: The group mean average.
        """
        group_means = []
        for group_id, agent_updates in group_updates_dict.items():
            # Sum the updates within the group and divide by the number of agents in the group
            group_mean = 0
            num_client_per_group = len(agent_updates)

            for _id, update in agent_updates.items():
                group_mean += 1/num_client_per_group * update  # average for the group
                
            group_means.append(group_mean)  # Average for the group

        # Final average over group-level averages
        return sum(group_means) / len(group_means)


    def hier_avg(group_data):
        """
        Sum the updates for each group, then do a weighted sum over groups based on group size. (this aggregation shifts towards groups with more clients)
        
        Parameters:
            group_data (dict): A dictionary where each key is a group ID, 
                            and the value is a dictionary of agent updates within that group.
                            Example:
                            {
                                'group_1': {'agent_1': update_1, 'agent_2': update_2, ...},
                                'group_2': {'agent_3': update_3, 'agent_4': update_4, ...},
                                ...
                            }
        
        Returns:
            float: The weighted average of all updates, weighted by the group sizes.
        """
        total_weighted_sum = 0
        total_clients = 0
        
        for group_id, agent_updates in group_data.items():
            group_sum = 0
            num_clients_per_group = len(agent_updates)
            
            for agent_id, update in agent_updates.items():
                group_sum +=  update        # sum of updates for this group
            
            # Accumulate totals for global weighted average
            total_weighted_sum += num_clients_per_group * group_sum
            total_clients += num_clients_per_group

        return total_weighted_sum / total_clients if total_clients > 0 else 0


from opacus.accountants.rdp import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier

def setup_noise_multiplier(args, in_group_sampling_rate):
    epsilon = args.target_epsilon # Desired epsilon
    delta = args.target_epsilon  # Desired delta
    steps = args.rounds  # Number of training rounds
    noise_multiplier = get_noise_multiplier(target_epsilon=epsilon,
                                            target_delta=delta, sample_rate=in_group_sampling_rate,
                                            steps=steps)
    # args.noise_multiplier =  noise_multiplier
    # if noise_multiplier != args.noise_multiplier:
    #     raise ValueError(f"Mismatch in noise multiplier: calculated {noise_multiplier}, but expected {args.noise_multiplier}.")
    
    return noise_multiplier

# Example usage
if __name__ == "__main__":
    print('-'*20 + 'find the optimal noise multiplier' +'-'*20)
    target_epsilon = 1.0  # Desired epsilon
    delta = 1e-5  # Desired delta
    sampling_rate = 100/6000 # client sampling rate per group, may vary across groups
    steps = 200  # Number of training rounds
    noise_multiplier = get_noise_multiplier(target_epsilon=target_epsilon,
                                            target_delta=delta, sample_rate=sampling_rate,
                                            steps=steps)
    print('the optimal noise multiplier for ({}, {})-DP is: {}'.format(target_epsilon, delta, noise_multiplier))

    print('-'*20 + 'find the minimial epsilon' +'-'*20)
    accountant = RDPAccountant()
    # Simulate training steps to compute epsilon
    for _ in range(steps):
        accountant.step(noise_multiplier=noise_multiplier, sample_rate=sampling_rate)
        computed_epsilon = accountant.get_epsilon(delta)
    print('computed epsilon is {} if noise multiplier is {}'.format(computed_epsilon, noise_multiplier))



import torch

def empirical_logistic_bellman(pred, label, eta, values, discount, clip=None):
    z = (label - pred) / eta
    if clip is not None:
        z = torch.clamp(z, -clip, clip)

    loss = torch.logsumexp(z, 0)
    return  eta * loss + torch.mean((1 - discount) * values, 0)

def S(pred, label, sampler, values, eta, discount):
    bellman = label - pred
    return torch.sum(sampler.probs().detach() * (bellman - eta * torch.log((sampler.n * sampler.probs().detach())))) +  (1-discount) * values.mean()
    
def nll_loss(observations, next_observations, rewards, actions, agent, args):
    newqvalue, newvalue = agent.get_value(observations)
    new_q_a_value = newqvalue.gather(1, actions.long().unsqueeze(-1)).squeeze(-1)
    weights = torch.clamp((new_q_a_value) / args.alpha, -50, 50)
    _, newlogprob, _, _ = agent.get_action(observations)
    nll = -torch.mean(torch.exp(weights.detach()) * newlogprob)
    return nll

import torch

def optimize_loss(buffer, loss_fn, optimizer: torch.optim.Optimizer, args, agent, optimizer_steps=300, sampler=None):
    (   observations,
        actions,
        rewards,
        next_observations,
        next_done,
    ) = buffer.get_all()

    def closure():
        newqvalue, newvalue = agent.get_value(observations)
        new_q_a_value = newqvalue.gather(1, actions.long().unsqueeze(1)).view(-1)
        with torch.no_grad(): target = rewards + args.gamma * agent.get_value(next_observations)[1] * (1 - next_done)
        optimizer.zero_grad()
        if loss_fn == empirical_logistic_bellman:
            loss = loss_fn(target, new_q_a_value, args.eta, newvalue, args.gamma)
        elif loss_fn == S:
            loss = loss_fn(target, new_q_a_value, sampler, newvalue, args.eta, args.gamma)
        else:
            loss = loss_fn(observations, next_observations, rewards, actions, agent, args)
        loss.backward()
        return loss

    for i in range(optimizer_steps):
        optimizer.step(closure)

def empirical_logistic_bellman(pred, label, eta, values, discount):
    z = (label - pred) / eta
    z = torch.clamp(z, -50, 50)
    return eta * torch.log(torch.exp(z).mean()) + torch.mean((1 - discount) * values, 0)

def S(pred, label, sampler, values, eta, discount):
    bellman = label - pred
    return torch.sum(sampler.probs().detach() * (bellman - eta * torch.log((sampler.n * sampler.probs().detach()))) +  (1-discount) * values)
    
def nll_loss(observations, next_observations, rewards, actions, agent, args):
    newqvalue, newvalue = agent.get_value(observations)
    new_q_a_value = newqvalue.gather(1, actions.long().unsqueeze(-1)).squeeze(-1)
    weights = torch.clamp((new_q_a_value) / args.alpha, -50, 50)
    _, newlogprob, _, _ = agent.get_action(observations)
    nll = -torch.mean(torch.exp(weights.detach()) * newlogprob)
    return nll
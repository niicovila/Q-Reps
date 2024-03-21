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
        if loss_fn == empirical_logistic_bellman or loss_fn == log_gumbel:
            loss = loss_fn(target, new_q_a_value, args.eta, newvalue, args.gamma)
        elif loss_fn == S:
            loss = loss_fn(target, new_q_a_value, sampler, newvalue, args.eta, args.gamma)
        else:
            loss = loss_fn(observations, next_observations, rewards, actions, agent, args)
        loss.backward()
        return loss

    for i in range(optimizer_steps):
        optimizer.step(closure)

def empirical_logistic_bellman(pred, label, probs, eta, values, discount):
    z = torch.mean(((label - pred) / eta) * torch.log(probs), dim=1)
    a = torch.max(z)
    t = a + torch.log(torch.sum(torch.exp(z-a)))
    return  eta * t + torch.mean((1 - discount) * values, 0)

def empirical_logistic_bellman_trick(pred, label, eta, values, discount):
    z = (label - pred) / eta
    a = torch.max(z)
    return  eta * (a + torch.log(torch.sum(torch.exp(z-a)-1))) + torch.mean((1 - discount) * values, 0)


def log_gumbel(pred, label, eta, values, discount):
    z = (label - pred) / eta
    z = torch.clamp(z, -50, 50)
    return  torch.log(torch.exp(z).mean() - (z + 1).mean())

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

def sac_loss(observations, next_observations, rewards, actions, agent, args):
    newqvalue, _ = agent.get_value(observations)
    _, _, newlogprobs, action_probs = agent.get_action(observations)
    actor_loss = torch.mean(action_probs * (args.alpha * newlogprobs - newqvalue))
    return actor_loss
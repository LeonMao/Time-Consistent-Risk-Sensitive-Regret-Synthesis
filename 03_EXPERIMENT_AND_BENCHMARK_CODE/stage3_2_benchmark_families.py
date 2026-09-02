
from __future__ import annotations
import random
from stage0_solver import DFA, PKWTS
from factored_belief import MixtureProductPrior, ProductComponent

def eventually_goal_dfa():
    def tr(q, label):
        if q == "qF": return "qF"
        return "qF" if "goal" in label else "qN"
    return DFA(("qN","qF"), "qN", frozenset({"qF"}), tr)

def distributed_layered_pkwts(layers, width, seed):
    if layers < 3 or width < 2:
        raise ValueError
    rng = random.Random(seed)
    layer_nodes = {l: tuple(f"v{l}_{j}" for j in range(width)) for l in range(1,layers+1)}
    states = ("s",) + tuple(x for l in range(1,layers+1) for x in layer_nodes[l]) + ("g",)
    labels = {x:frozenset() for x in states}; labels["g"]=frozenset({"goal"})
    patterns={}; weights={}
    patterns["s"]=(frozenset(layer_nodes[1]),)
    for j,y in enumerate(layer_nodes[1]):
        weights[("s",y)] = 1.0 + 0.15*j + rng.uniform(0,0.15)
    for l in range(1,layers+1):
        for j,x in enumerate(layer_nodes[l]):
            if l == layers:
                patterns[x]=(frozenset({"g"}),)
                weights[(x,"g")] = 2.0 + rng.uniform(0,0.8)
                continue
            nxt = layer_nodes[l+1]
            primary = nxt[(j+rng.randrange(width))%width]
            closed={primary}
            if rng.random()<0.55:
                closed.add(nxt[(j+1+rng.randrange(width))%width])
            if l+2 <= layers:
                skip=layer_nodes[l+2][rng.randrange(width)]
            else:
                skip="g"
            open_pat=set(closed); open_pat.add(skip)
            patterns[x]=(frozenset(closed),frozenset(open_pat))
            for y in sorted(closed):
                weights[(x,y)] = 2.2 + rng.uniform(0,1.8)
            if (x,skip) not in weights:
                weights[(x,skip)] = 1.0 + rng.uniform(0,1.4)
    patterns["g"]=(frozenset(),)
    return PKWTS(states=states,x0="s",patterns=patterns,weights=weights,labels=labels)

def independent_prior(T,p_open,seed=0,jitter=.05):
    rng=random.Random(seed); marg={}
    for x in T.states:
        if len(T.patterns[x])>1:
            p=min(.92,max(.08,p_open+rng.uniform(-jitter,jitter)))
            marg[x]=(1-p,p)
    return MixtureProductPrior(T,[ProductComponent(1.0,marg)])

def two_mode_correlated_prior(T,seed=0):
    rng=random.Random(seed); good={}; poor={}
    for x in T.states:
        if len(T.patterns[x])>1:
            pg=min(.94,max(.65,.82+rng.uniform(-.06,.06)))
            pp=min(.30,max(.05,.14+rng.uniform(-.05,.05)))
            good[x]=(1-pg,pg); poor[x]=(1-pp,pp)
    return MixtureProductPrior(T,[ProductComponent(.42,good),ProductComponent(.58,poor)])

def unknown_count(T):
    return sum(1 for x in T.states if len(T.patterns[x])>1)

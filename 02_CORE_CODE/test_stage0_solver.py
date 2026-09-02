
from math import isclose
from stage0_solver import DFA, PKWTS, Stage0RegretSolver

def eventually_goal_dfa():
    def tr(q,label):
        if q=="qF": return "qF"
        return "qF" if "goal" in label else "qN"
    return DFA(("qN","qF"),"qN",frozenset({"qF"}),tr)

def extinguisher_then_fire_dfa():
    def tr(q,label):
        if q in ("qF","qDead"): return q
        e="ext" in label; f="fire" in label
        if q=="q0":
            if f and not e: return "qDead"
            if e: return "q1"
            return "q0"
        if q=="q1": return "qF" if f else "q1"
        raise ValueError(q)
    return DFA(("q0","q1","qF","qDead"),"q0",frozenset({"qF"}),tr)

def shortcut_problem():
    states=("0","1","2","3","4","5")
    patterns={
        "0":(frozenset({"1","2"}),),
        "1":(frozenset({"3"}),),
        "2":(frozenset({"0","5"}),frozenset({"0"})), # 0=open, 1=closed
        "3":(frozenset({"4"}),),
        "4":(frozenset({"5"}),),
        "5":(frozenset(),),
    }
    weights={
        ("0","1"):2,("0","2"):1,("1","3"):2,
        ("2","0"):1,("2","5"):2,("3","4"):3,("4","5"):3,
    }
    labels={x:frozenset() for x in states}; labels["5"]=frozenset({"goal"})
    return PKWTS(states,"0",patterns,weights,labels)

def deterministic_problem():
    states=("s","a","g")
    patterns={"s":(frozenset({"a","g"}),),"a":(frozenset({"g"}),),"g":(frozenset(),)}
    weights={("s","a"):1,("s","g"):5,("a","g"):1}
    labels={"s":frozenset(),"a":frozenset(),"g":frozenset({"goal"})}
    return PKWTS(states,"s",patterns,weights,labels)

def temporal_logic_problem():
    states=("s","e","f")
    patterns={"s":(frozenset({"e","f"}),),"e":(frozenset({"f"}),),"f":(frozenset(),)}
    weights={("s","e"):2,("s","f"):1,("e","f"):2}
    labels={"s":frozenset(),"e":frozenset({"ext"}),"f":frozenset({"fire"})}
    return PKWTS(states,"s",patterns,weights,labels)

def run_shortcut():
    solver=Stage0RegretSolver(shortcut_problem(),eventually_goal_dfa())
    result=solver.solve()
    ow=next(w for w in solver.worlds if w[2]==0)
    cw=next(w for w in solver.worlds if w[2]==1)
    assert isclose(result.oracle_costs[ow],3.0)
    assert isclose(result.oracle_costs[cw],10.0)
    assert isclose(result.value,2.0)
    assert result.policy[solver.start].target=="2"
    wr,costs,regrets=solver.policy_worst_regret(result.policy)
    assert isclose(wr,2.0)
    assert isclose(costs[ow],3.0) and isclose(costs[cw],12.0)
    assert isclose(regrets[ow],0.0) and isclose(regrets[cw],2.0)
    penalties=sorted(round(result.dist[f]-solver.best_response(f.K),10) for f in solver.accepting_nodes)
    assert penalties==[0.0,2.0,7.0]
    return {
        "game_nodes":len(solver.adj),
        "agent_nodes":len(solver.agent_nodes),
        "env_nodes":len(solver.env_nodes),
        "E_SP_edges":len(result.shortest_path_edges),
        "oracle_open":result.oracle_costs[ow],
        "oracle_closed":result.oracle_costs[cw],
        "optimal_regret":result.value,
        "initial_action":result.policy[solver.start].target,
        "cost_open":costs[ow],
        "cost_closed":costs[cw],
        "regret_open":regrets[ow],
        "regret_closed":regrets[cw],
        "terminal_penalties":penalties,
    }

def run_deterministic():
    solver=Stage0RegretSolver(deterministic_problem(),eventually_goal_dfa())
    result=solver.solve()
    wr,costs,regrets=solver.policy_worst_regret(result.policy)
    assert isclose(result.value,0.0)
    assert result.policy[solver.start].target=="a"
    assert isclose(wr,0.0)
    assert list(costs.values())==[2.0]
    return {"optimal_regret":result.value,"initial_action":"a","cost":2.0}

def run_temporal():
    solver=Stage0RegretSolver(temporal_logic_problem(),extinguisher_then_fire_dfa())
    result=solver.solve()
    assert list(result.oracle_costs.values())==[4.0]
    assert isclose(result.value,0.0)
    assert result.policy[solver.start].target=="e"
    wr,costs,_=solver.policy_worst_regret(result.policy)
    assert isclose(wr,0.0)
    assert list(costs.values())==[4.0]
    return {"optimal_regret":result.value,"initial_action":"e","valid_task_cost":4.0}

if __name__=="__main__":
    s1=run_shortcut(); s2=run_deterministic(); s3=run_temporal()
    print("ALL TESTS PASSED")
    print("\n[Shortcut]")
    for k,v in s1.items(): print(f"{k}: {v}")
    print("\n[Deterministic sanity]")
    for k,v in s2.items(): print(f"{k}: {v}")
    print("\n[Temporal-logic sanity]")
    for k,v in s3.items(): print(f"{k}: {v}")

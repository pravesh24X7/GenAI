Starting with LangGraphs:
    framework that let's you build multi step, stateful and event driven workflows using LLMs
    ideal for designing single and multi-agent applications.

Core concepts of LangGraphs:
    LangGraphs tries to draw graph of given workflow where each node represent the task and edges represents the routing.
    tasks can be executed parallel, branching, loops, memory recorded.

    each step in workflow perform different task.

LangGraphs terminologies:
    graph, nodes, edges.    (we talked about these earlier)

    state: means data, which is shared among all the nodes in the graph. It is mutable in nature.
    reducers: how updates from nodes are applied to state. Each key in state can have it's own reducers.

Execution method in LangGraphs:
    step 1:     define the graph. (nodes & edges)
    step 2:     compilation: run `.compile` method on the state graph. It checks graph structure and prepare it for execution.
    step 3:     invocation: run graph with `.invoke(initial_state)`, LangGraphs sends initial_state as message to the entry node.
    step 4: super step begins: execution proceeds in rounds.
    step 5: message passing and node activation: message are passed to downstream via nodes and edges. Node that receives messages become active  for next round.
    step 6:     Halting condition:  execution stops when
                    no nodes are active.
                    no messages are in transit.
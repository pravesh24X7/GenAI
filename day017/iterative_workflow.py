import operator
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal, Annotated
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv


load_dotenv()


class TweetState(TypedDict):
    
    topic: str
    tweet: str
    evaluation: Literal["approved", "needs_improvement"]
    feedback: str
    iteration: int
    max_iteration: int

    tweet_history: Annotated[list[str], operator.add]
    feedback_history: Annotated[list[str], operator.add]


class EvalStructure(BaseModel):

    evaluation: Literal["approved", "needs_improvement"] = Field(description="Classify the given tweet into approved or need_improvement category")
    feedback: str = Field(description="Textual feedback for the given tweet.")


# ideally we should use model which is good at generation like Gpt-4 or some other models. But here for learning purpose we're using the same model for every task.
generator_llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct",
                         temperature=1.0,
                         model_kwargs={})


evaluator_llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct",
                         temperature=0.5,
                         model_kwargs={}).with_structured_output(EvalStructure)


optimizer_llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.7,
                         model_kwargs={})


def generation(state: TweetState) -> dict:
    messages = [
        SystemMessage(content="You're a funny and clever X (Twitter) / Instagram influencer. "),
        HumanMessage(content=f"""
        Write a short, original, and hilarious tweet on topic: {state['topic']}.

        Rules:
        1. Do not use question-answer format.
        2. Max. 280 characters.
        3. Use observational humor, irony, sarcasm, or cultural references.
        4. Think in meme logic, punchlines, or relatable takes.
        5. Use simple, day to day english.
        6. This is version {(int(state['iteration']) + 1)} 
""")
    ]
    tweet = generator_llm.invoke(messages).content

    return {'tweet': tweet, 'tweet_history': [tweet]}


def evaluation(state: TweetState) -> dict:

    evaluation_messages = [
        SystemMessage(content="You're a sharp and insightful social media content critic."),
        HumanMessage(content=f"""
        Evaluate the following tweet based on humor, originality, and engagement potential:

        Tweet:
        {state['tweet']}

        Criteria:
        1. Is it funny? Explain briefly.
        2. Is it original or does it feel overused?
        3. Is it relatable or culturally relevant?
        4. Does it follow the given rules (no Q&A, under 280 chars, simple English)?
        5. Give a score out of 10.

        Keep feedback concise but meaningful.
    """)
    ]   
    response = evaluator_llm.invoke(evaluation_messages)

    return {'evaluation': response.evaluation, 'feedback': response.feedback, 'feedback_history': [response.feedback]}


def optimization(state: TweetState) -> dict:
    
    optimization_messages = [
    SystemMessage(content="You're a witty and highly creative social media expert who improves content."),
    HumanMessage(content=f"""
    Improve the following tweet based on the evaluation feedback.

    Original Tweet:
    {state['tweet']}

    Feedback:
    {state['feedback']}

    Instructions:
    1. Keep it under 280 characters.
    2. Make it funnier, sharper, and more engaging.
    3. Preserve the core idea but enhance the punchline.
    4. Use simple, everyday English.
    5. Avoid question-answer format.

    This is optimized version {(int(state['iteration']) + 1)}.
""")
]
    new_tweet = optimizer_llm.invoke(optimization_messages).content

    return {'tweet': new_tweet, 'iteration': (int(state['iteration']) + 1), 'tweet_history': [new_tweet]}


def perform_evaluation(state: TweetState) -> Literal["approved", "needs_improvement"]:
    if state['evaluation'] == 'approved' or state['iteration'] >= state['max_iteration']:
        return "approved"
    else:
        return "needs_improvement"


graph = StateGraph(TweetState)

graph.add_node('generate', generation)
graph.add_node('evaluate', evaluation)
graph.add_node('optimize', optimization)

graph.add_edge(START, 'generate')
graph.add_edge('generate', 'evaluate')
graph.add_conditional_edges('evaluate', perform_evaluation, {'approved': END, 'needs_improvement': 'optimize'})
graph.add_edge('optimize', 'evaluate')

workflow = graph.compile()

initial_state = {
    'topic': "Munna bhaiya",
    'max_iteration': 5,
    'iteration': 0
}

final_state = workflow.invoke(initial_state)
print(f"Final Tweet: {final_state['tweet']}\n\nIteration runs: {final_state['iteration']}")

print(f"\n\n\n\nAll Intermediate tweets are:")
for tweet in final_state['tweet_history']:
    print(tweet)

img = workflow.get_graph().draw_mermaid_png()
with open("./img001.png", "wb") as f:
    f.write(img)
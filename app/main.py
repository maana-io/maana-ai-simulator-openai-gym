# --- External imports
from asgi_lifespan import Lifespan, LifespanMiddleware
import json
import time
import threading
import numpy as np
# GraphQL
from ariadne import ObjectType, QueryType, MutationType, gql, make_executable_schema
from ariadne.asgi import GraphQL
from graphqlclient import GraphQLClient
# OpenAI Gym
import gym
from gym import envs
import retro

# --- Constants

ID = "id"

STATE_CLIENT = "client"
STATE_CONFIG = "config"
STATE_THREAD = "thread"
STATE_ENVIRONMENT = "environment"
STATE_OBSERVATION = "observation"
STATE_EPISODE = "episode"
STATE_STEP = "step"
STATE_ACTION = "action"
STATE_REWARD = "reward"
STATE_STATUS = "status"

CONFIG_MODE = "mode"
CONFIG_AGENT_URI = "agentUri"
CONFIG_TOKEN = "token"
CONFIG_ENVIRONMENT = "environment"

STATUS_CODE = "code"
STATUS_ERRORS = "errors"

MODE_TRAINING = "Training"
MODE_PERFORMING = "Performing"

CODE_IDLE = "Idle"
CODE_STARTING = "Starting"
CODE_RUNNING = "Running"
CODE_STOPPED = "Stopped"
CODE_ENDED = "Ended"
CODE_ERROR = "Error"

# --- Simulation


def set_sim_status(code, errors=[]):
    ts = time.time()
    app.state[STATE_STATUS] = {
        ID: "gym@" + str(ts),
        STATUS_CODE: code,
        STATUS_ERRORS: errors}
    return app.state[STATE_STATUS]


def create_state():
    state = {
        STATE_CLIENT: None,
        STATE_CONFIG: None,
        STATE_THREAD: None,
        STATE_ENVIRONMENT: None,
        STATE_EPISODE: 0,
        STATE_STEP: 0,
        STATE_OBSERVATION: (0,),
        STATE_STATUS: None
    }
    app.state = state
    set_sim_status(CODE_IDLE)
    return state


def execute_client_request(graphql, variables=None):
    try:
        client = app.state[STATE_CLIENT]
        if (client == None):
            raise Exception("No client.  Running?")
        result = client.execute(graphql, variables)
        print("result: " + result)
        json_result = json.loads(result)
        if ("errors" in json_result):
            errors = json_result["errors"]
            if (errors != None):
                error_messages = [e["message"] for e in errors]
                set_sim_status(CODE_ERROR, error_messages)
                return None
        return json_result["data"]
    except Exception as e:
        print("exception: " + repr(e))
        set_sim_status(CODE_ERROR, [str(e)])
        return None


def agent_on_reset():
    result = execute_client_request('''
    {
        onReset
    }
    ''')
    # print("agent_on_reset: " + str(result))
    return str(result)


def agent_on_step(state, last_reward, last_action, done, context):
    result = execute_client_request('''
        mutation onStep($state: [Float!]!, $lastReward: Float!, $lastAction: Int!, $isDone: Boolean!, $context: String) {
            onStep(state: $state, lastReward: $lastReward, lastAction: $lastAction, isDone: $isDone, context: $context) {
                id
                action
                context
            }
        }
    ''', {
        "state": state, "lastReward": last_reward, "lastAction": last_action, "isDone": done, "context": context
    })
    # print("agent_on_step: " + str(result))
    return str(result)


def run_simulation(config):
    set_sim_status(CODE_STARTING)
    env = try_make_env(config[STATE_ENVIRONMENT])
    if (env == None):
        set_sim_status(
            CODE_ERROR, ["Can't load environment: " + config[CONFIG_ENVIRONMENT]])
        return app.state[STATE_STATUS]
    app.state[STATE_ENVIRONMENT] = env

    client = GraphQLClient(config[CONFIG_AGENT_URI])
    client.inject_token("Bearer " + config[CONFIG_TOKEN])
    app.state[STATE_CLIENT] = client

    thread = threading.Thread(target=run_episodes, args=(99,))
    print("thread: " + repr(thread))
    app.state[STATE_THREAD] = thread
    thread.start()

    return app.state[STATE_STATUS]


def stop_simulation():
    set_sim_status(CODE_STOPPED)

    # Close the env and write monitor result info to disk
    env = app.state[STATE_ENVIRONMENT]
    if (env != None):
        env.close()

    thread = app.state[STATE_THREAD]
    if (thread != None):
        thread.join()

    return app.state[STATE_STATUS]


# --- OpenAI Gym

def try_make_env(environmentId):
    try:
        return gym.make(environmentId)
    except:
        return retro.make(environmentId)
    return None


def run_episodes(episode_count):
    print("run_episodes:" + str(episode_count))
    app.state[STATE_EPISODE] = 0

    set_sim_status(CODE_RUNNING)

    env = app.state[STATE_ENVIRONMENT]

    for i in range(episode_count):
        if (app.state[STATE_STATUS][STATUS_CODE] != CODE_RUNNING):
            break

        app.state[STATE_EPISODE] = i

        done = False
        last_reward = 0
        last_action = 0

        ob = env.reset()
        print("episode #" + str(i) + ": " + repr(ob))

        agent_context = agent_on_reset()

        step = 0
        while app.state[STATE_STATUS][STATUS_CODE] == CODE_RUNNING:
            app.state[STATE_STEP] = step
            step += 1

            state = ob
            if (isinstance(ob, np.ndarray)):
                state = ob.tolist()
            elif (isinstance(ob, np.int64) or isinstance(ob, int)):
                state = (float(ob),)
            else:
                print("type of state`: " + repr(type(state)))

            app.state[STATE_OBSERVATION] = state

            on_step_result = agent_on_step(
                state, last_reward, last_action, done, agent_context)

            if (app.state[STATE_STATUS][STATUS_CODE] == CODE_ERROR):
                break

            last_action = env.action_space.sample()

            ob, last_reward, done, _ = env.step(last_action)
            if done:
                agent_on_step(ob, last_reward, last_action,
                              done, agent_context)
                print("- DONE!")
                break

            print("- step = " + str(step) + ", reward = " +
                  str(last_reward) + ", ob = " + repr(ob))

            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

            # render = env.render('rgb_array')
            # print("- render " + repr(render))

    status = app.state[STATE_STATUS]
    if (status[STATUS_CODE] != CODE_ERROR and status[STATUS_CODE] != CODE_STOPPED):
        set_sim_status(CODE_ENDED)

    # Close the env and write monitor result info to disk
    env.close()

# --- GraphQL


# Map resolver functions to Query and Mutation fields
query = QueryType()
mutation = MutationType()

# Define types using Schema Definition Language (https://graphql.org/learn/schema/)
# Wrapping string in gql function provides validation and better error traceback
type_defs = gql("""

    enum StatusCode {
        Idle
        Starting
        Running
        Stopped
        Ended
        Error
    }

    enum Mode {
        Training
        Performing
    }

    type SimStatus {
        id: ID!
        code: StatusCode!
        mode: Mode!
        errors: [String!]!
    }

    input ConfigInput {
        environment: ID!
        mode: Mode!
        agentUri: String!
        token: String!
    }

    type Environment {
        id: ID!
    }

    type Observation {
        episode: Int!
        step: Int!
        data: [Float!]!
        simStatus: SimStatus!
    }

    type Query {
        listEnvironments: [Environment!]!
        simStatus: SimStatus!
        observe: Observation!
        test: String!
    }
    type Mutation {
        run(config: ConfigInput!): SimStatus!
        stop: SimStatus!
    }
""")


# Resolvers are simple python functions
@query.field("listEnvironments")
def resolve_listEnvironments(*_):
    envids = [spec.id for spec in envs.registry.all()]
    res = map(lambda x: {"id": x}, sorted(envids))
    return res


@query.field("simStatus")
def resolve_simStatus(*_):
    return app.state[STATE_STATUS]


@query.field("observe")
def resolve_observe(*_):
    observation = {
        "episode": app.state[STATE_EPISODE],
        "step": app.state[STATE_STEP],
        "data": app.state[STATE_OBSERVATION],
        "simStatus": app.state[STATE_STATUS]
    }
    print('observe: ' + repr(observation))
    return observation


@query.field("test")
def resolve_test(*_):
    result = execute_client_request('''
        {
            allSimulators {
                id
            }
        }
    ''')
    print("result: " + str(result))
    return str(result)


@mutation.field("stop")
def resolve_stop(*_):
    return stop_simulation()


@mutation.field("run")
def resolve_run(*_, config):
    return run_simulation(config)


# Create executable GraphQL schema
schema = make_executable_schema(type_defs, [query, mutation])

# --- ASGI app

# 'Lifespan' is a standalone ASGI app.
# It implements the lifespan protocol,
# and allows registering lifespan event handlers.
lifespan = Lifespan()


@lifespan.on_event("startup")
async def startup():
    print("Starting up...")
    print("... done!")


@lifespan.on_event("shutdown")
async def shutdown():
    print("Shutting down...")
    stop_simulation()
    print("... done!")

# Create an ASGI app using the schema, running in debug mode
app = GraphQL(schema, debug=True)

# 'LifespanMiddleware' returns an ASGI app.
# It forwards lifespan requests to 'lifespan',
# and anything else goes to 'app'.
app = LifespanMiddleware(app, lifespan=lifespan)

# Create shared state on the app object
create_state()

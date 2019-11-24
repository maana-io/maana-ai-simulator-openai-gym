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

ACTION = "action"
AGENT_URI = "agentUri"
CLIENT = "client"
CODE = "code"
CONFIG = "config"
CONTEXT = "context"
DATA = "data"
ENDED = "Ended"
ENVIRONMENT = "environment"
EPISODE = "episode"
ERROR = "Error"
ERRORS = "errors"
ID = "id"
IDLE = "Idle"
MESSAGE = "message"
MODE = "mode"
OBSERVATION = "observation"
PERFORMING = "Performing"
REWARD = "reward"
RUNNING = "Running"
SIM_STATUS = "simStatus"
STARTING = "Starting"
STATUS = "status"
STEP = "step"
STOPPED = "Stopped"
THREAD = "thread"
TOKEN = "token"
TRAINING = "Training"

# --- Simulation


def set_sim_status(code, errors=[]):
    ts = time.time()
    app.state[STATUS] = {
        ID: "gym@" + str(ts),
        CODE: code,
        ERRORS: errors}
    return app.state[STATUS]


def create_state():
    state = {
        CLIENT: None,
        CONFIG: None,
        THREAD: None,
        ENVIRONMENT: None,
        EPISODE: 0,
        STEP: 0,
        OBSERVATION: (0,),
        REWARD: 0,
        STATUS: None
    }
    app.state = state
    set_sim_status(IDLE)
    return state


def execute_client_request(graphql, variables=None):
    try:
        client = app.state[CLIENT]
        if (client == None):
            raise Exception("No client.  Running?")
        result = client.execute(graphql, variables)
        # print("result: " + result)
        json_result = json.loads(result)
        if (ERRORS in json_result):
            errors = json_result[ERRORS]
            if (errors != None):
                error_messages = [e[MESSAGE] for e in errors]
                set_sim_status(ERROR, error_messages)
                return None
        return json_result[DATA]
    except Exception as e:
        print("exception: " + repr(e))
        set_sim_status(ERROR, [str(e)])
        return None


def agent_on_reset():
    result = execute_client_request('''
    {
        onReset
    }
    ''')
    if (result == None):
        return None
    return result["onReset"]


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
    if (result == None):
        return None
    return result["onStep"]


def run_simulation(config):
    set_sim_status(STARTING)
    env = try_make_env(config[ENVIRONMENT])
    if (env == None):
        set_sim_status(
            ERROR, ["Can't load environment: " + config[ENVIRONMENT]])
        return app.state[STATUS]
    app.state[ENVIRONMENT] = env

    client = GraphQLClient(config[AGENT_URI])
    client.inject_token("Bearer " + config[TOKEN])
    app.state[CLIENT] = client

    thread = threading.Thread(target=run_episodes, args=(99,))
    print("thread: " + repr(thread))
    app.state[THREAD] = thread
    thread.start()

    return app.state[STATUS]


def stop_simulation():
    set_sim_status(STOPPED)

    # Close the env and write monitor result info to disk
    env = app.state[ENVIRONMENT]
    if (env != None):
        env.close()

    thread = app.state[THREAD]
    if (thread != None):
        thread.join()

    return app.state[STATUS]


# --- OpenAI Gym

def try_make_env(environmentId):
    try:
        return gym.make(environmentId)
    except:
        return retro.make(environmentId)
    return None


def run_episodes(episode_count):
    try:
        print("run_episodes:" + str(episode_count))
        app.state[EPISODE] = 0
        app.state[REWARD] = 0

        set_sim_status(RUNNING)

        env = app.state[ENVIRONMENT]

        for i in range(episode_count):
            if (app.state[STATUS][CODE] != RUNNING):
                break

            app.state[EPISODE] = i

            done = False
            last_reward = 0
            last_action = 0

            ob = env.reset()
            print("episode #" + str(i) + ": " + repr(ob))

            agent_context = agent_on_reset()

            step = 0
            while app.state[STATUS][CODE] == RUNNING:
                app.state[STEP] = step
                step += 1

                state = ob
                if (isinstance(ob, np.ndarray)):
                    state = ob.tolist()
                elif (isinstance(ob, np.int64) or isinstance(ob, int)):
                    state = (float(ob),)
                else:
                    print("type of state`: " + repr(type(state)))

                app.state[OBSERVATION] = state

                on_step_result = agent_on_step(
                    state, last_reward, last_action, done, agent_context)

                # print("on_step_result " + repr(on_step_result))

                if (app.state[STATUS][CODE] == ERROR):
                    break

                # last_action = env.action_space.sample()
                last_action = on_step_result[ACTION]
                agent_context = on_step_result[CONTEXT]

                ob, last_reward, done, _ = env.step(last_action)
                app.state[REWARD] += last_reward

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

        status = app.state[STATUS]
        if (status[CODE] != ERROR and status[CODE] != STOPPED):
            set_sim_status(ENDED)

    except Exception as e:
        print("exception: " + repr(e))
        set_sim_status(ERROR, [str(e)])

    finally:
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
        reward: Float!
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
    return app.state[STATUS]


@query.field("observe")
def resolve_observe(*_):
    observation = {
        EPISODE: app.state[EPISODE],
        STEP: app.state[STEP],
        REWARD: app.state[REWARD],
        DATA: app.state[OBSERVATION],
        SIM_STATUS: app.state[STATUS]
    }
    print('observe: ' + repr(observation))
    return observation


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

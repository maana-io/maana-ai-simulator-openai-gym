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
AGENT_STATS = "agentStats"
AGENTS = "agents"
AGENT_URI = "agentUri"
CLIENT = "client"
CODE = "code"
CONFIG = "config"
CONTEXT = "context"
DATA = "data"
ENDED = "Ended"
ENVIRONMENT = "environment"
ENVIRONMENT_ID = "environmentId"
EPISODE = "episode"
ERROR = "Error"
ERRORS = "errors"
ID = "id"
IDLE = "Idle"
MESSAGE = "message"
MODE = "mode"
OBSERVATION = "observation"
PERFORMING = "Performing"
LAST_ACTION = "lastAction"
LAST_REWARD = "lastReward"
RUNNING = "Running"
STARTING = "Starting"
STATUS = "status"
STEP = "step"
STOPPED = "Stopped"
THREAD = "thread"
TOKEN = "token"
TOTAL_REWARD = "totalReward"
TRAINING = "Training"
URI = "uri"

# --- Simulation


def set_status(code, errors=[]):
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
        LAST_ACTION: (0,),
        LAST_REWARD: (0,),
        TOTAL_REWARD: (0,),
        STATUS: None
    }
    app.state = state
    set_status(IDLE)
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
                set_status(ERROR, error_messages)
                return None
        return json_result[DATA]
    except Exception as e:
        print("exec exception: " + repr(e))
        set_status(ERROR, [str(e)])
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
        mutation onStep($state: [Float!]!, $lastReward: [Float!]!, $lastAction: [Float!]!, $isDone: Boolean!, $context: String) {
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
    set_status(STARTING)

    app.state[CONFIG] = config

    env = try_make_env(config[ENVIRONMENT_ID])
    if (env == None):
        set_status(
            ERROR, ["Can't load environment: " + config[ENVIRONMENT_ID]])
        return app.state[STATUS]
    app.state[ENVIRONMENT] = env

    agents = config[AGENTS]
    uri = agents[0][URI]
    token = agents[0][TOKEN]

    client = GraphQLClient(uri)
    client.inject_token("Bearer " + token)
    app.state[CLIENT] = client

    thread = threading.Thread(target=run_episodes, args=(99,))
    print("thread: " + repr(thread))
    app.state[THREAD] = thread
    thread.start()

    return app.state[STATUS]


def stop_simulation():
    set_status(STOPPED)

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
        app.state[LAST_ACTION] = (0,)
        app.state[LAST_REWARD] = (0,)

        set_status(RUNNING)

        env = app.state[ENVIRONMENT]

        for i in range(episode_count):
            if (app.state[STATUS][CODE] != RUNNING):
                break

            app.state[EPISODE] = i

            done = False
            last_action = (0,)
            last_reward = (0,)

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
                app.state[LAST_ACTION] = last_action
                app.state[LAST_REWARD] = last_reward

                on_step_result = agent_on_step(
                    state, last_reward, last_action, done, agent_context)

                # print("on_step_result " + repr(on_step_result))

                if (app.state[STATUS][CODE] == ERROR):
                    break

                # last_action = env.action_space.sample()
                last_action = on_step_result[ACTION]
                agent_context = on_step_result[CONTEXT]

                ob, last_reward, done, _ = env.step(last_action[0])
                last_reward = (last_reward,)
                app.state[TOTAL_REWARD] = (app.state[TOTAL_REWARD][0] +
                                           last_reward[0],)

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
            set_status(ENDED)

    except Exception as e:
        print("run exception: " + repr(e))
        set_status(ERROR, [str(e)])

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

# Boilerplate
type Info {
  id: ID!
  name: String!
  description: String
}

# Unkown
# Idle
# Starting
# Running
# Stopped
# Ended
# Error
type StatusCode {
  id: ID!
}

# Training
# Performing
type Mode {
  id: ID!
}

input AgentInput {
  type: String
  uri: String
  token: String
  features: [Float!]
}

type Agent {
  type: String
  uri: String
  token: String
  features: [Float!]
}

input ConfigInput {
  episodes: Int
  environmentId: ID!
  modeId: ID
  agents: [AgentInput!]!
}

type Config {
  episodes: Int!
  environment: Environment!
  mode: Mode!
  agents: [Agent!]!
}

type Environment {
  id: ID!
  name: String
  observationSpace: FeatureSpace
  actionSpace: FeatureSpace
  rewardSpace: FeatureSpace
}

type Dimension {
  id: ID!
  isContinuous: Boolean
  rangeMin: Float
  rangeMax: Float
}

type FeatureSpace {
  id: ID!
  name: String
  dimensions: [Dimension]
}

type Status {
  id: ID!
  code: StatusCode!
  errors: [String!]!
}

type AgentStats {
  lastReward: [Float!]!
  lastAction: [Float!]!
  totalReward: [Float!]!
}

type Observation {
  episode: Int!
  step: Int!
  mode: Mode
  data: [Float!]!
  agentStats: [AgentStats!]!
  status: Status!
}

type Query {
  listEnvironments: [Environment!]!
  configuration: Config!
  status: Status!
  observe: Observation!
}

type Mutation {
  run(config: ConfigInput!): Status!
  stop: Status!
}

""")


def transformStatus(status):
    print(repr(status))
    return {"id": status["id"], "code": {"id": status["code"]}, "errors": status["errors"]}

# Resolvers are simple python functions
@query.field("listEnvironments")
def resolve_listEnvironments(*_):
    envids = [spec.id for spec in envs.registry.all()]
    res = map(lambda x: {"id": x}, sorted(envids))
    return res


@query.field("status")
def resolve_status(*_):
    return transformStatus(app.state[STATUS])


@query.field("observe")
def resolve_observe(*_):
    observation = {
        EPISODE: app.state[EPISODE],
        STEP: app.state[STEP],
        AGENT_STATS: ({
            LAST_ACTION: app.state[LAST_ACTION],
            LAST_REWARD: app.state[LAST_REWARD],
            TOTAL_REWARD: app.state[TOTAL_REWARD],
        },),
        DATA: app.state[OBSERVATION],
        STATUS: transformStatus(app.state[STATUS])
    }
    print('observe: ' + repr(observation))
    return observation


@mutation.field("stop")
def resolve_stop(*_):
    return transformStatus(stop_simulation())


@mutation.field("run")
def resolve_run(*_, config):
    return transformStatus(run_simulation(config))


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

# --- External imports
from asgi_lifespan import Lifespan, LifespanMiddleware
import json
import time
import datetime
import threading
import numpy as np
# GraphQL
from ariadne import ObjectType, QueryType, MutationType, gql, make_executable_schema
from ariadne.asgi import GraphQL
from graphqlclient import GraphQLClient
# OpenAI Gym
import gym
from gym import envs
# import retro

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
MODE_ID = "modeId"
OBSERVATION = "observation"
PERFORMING = "Performing"
LAST_ACTION = "lastAction"
LAST_REWARD = "lastReward"
RENDER = "render"
RUNNING = "Running"
SESSION_ID = "sessionId"
STARTING = "Starting"
STATUS = "status"
STATE = "state"
STEP = "step"
STOPPED = "Stopped"
THREAD = "thread"
TOKEN = "token"
TOTAL_REWARD = "totalReward"
TRAINING = "Training"
UNKNOWN = "Unknown"
URI = "uri"

# --- Simulation


def set_status(session_id, code, errors=[]):
    app_state = get_app_state(session_id)

    ts = time.time()
    app_state[STATUS] = {
        ID: str(session_id) + ":" + str(ts),
        CODE: code,
        ERRORS: errors}
    return app_state[STATUS]


def create_state(session_id):
    state = {
        ID: session_id,
        CLIENT: None,
        CONFIG: None,
        THREAD: None,
        ENVIRONMENT: None,
        EPISODE: 0,
        STEP: 0,
        STATE: (0,),
        LAST_ACTION: (0,),
        LAST_REWARD: (0,),
        TOTAL_REWARD: (0,),
        RENDER: "",
        STATUS: None
    }
    return state


def get_app_state(session_id):
    if (not session_id in app.sessions):
        app_state = create_state(session_id)
        app.sessions[session_id] = app_state
        set_status(session_id, UNKNOWN)
    return app.sessions[session_id]


def execute_client_request(session_id, graphql, variables=None):
    try:
        app_state = get_app_state(session_id)
        client = app_state[CLIENT]
        if (client == None):
            raise Exception("No client.  Running?")
        result = client.execute(graphql, variables)
        # print("result: " + result)
        json_result = json.loads(result)
        if (ERRORS in json_result):
            errors = json_result[ERRORS]
            if (errors != None):
                error_messages = [e[MESSAGE] for e in errors]
                set_status(session_id, ERROR, error_messages)
                return None
        return json_result[DATA]
    except Exception as e:
        print("exec exception: " + repr(e))
        set_status(session_id, ERROR, [str(e)])
        return None


def agent_on_reset(session_id, state_space, action_space, model_id, is_training):
    result = execute_client_request(session_id, '''
        mutation onReset($stateSpace: Int, $actionSpace: Int, $modelId: ID, $isTraining: Boolean) {
            onReset(stateSpace: $stateSpace, actionSpace: $actionSpace, modelId: $modelId, isTraining: $isTraining)
        }
    ''', {
        "stateSpace": state_space, "actionSpace": action_space, "modelId": model_id, "isTraining": is_training
    })
    if (result == None):
        return None
    # print("onReset: " + repr(result))
    return result["onReset"]


def agent_on_step(session_id, state, last_reward, last_action, step, context):
    t1 = time.time()
    result = execute_client_request(session_id, '''
        mutation onStep($state: [Float!]!, $lastReward: [Float!]!, $lastAction: [Float!]!, $step: Int!, $context: String) {
            onStep(state: $state, lastReward: $lastReward, lastAction: $lastAction, step: $step, context: $context) {
                id
                action
                context
            }
        }
    ''', {
        "state": state, "lastReward": last_reward, "lastAction": last_action, "step": step, "context": context
    })
    t2 = time.time()
    print("onStep took " + str(t2 - t1) + " seconds")
    if (result == None):
        return None
    # print("onStep: " + repr(result))
    return result["onStep"]


def agent_on_done(session_id, last_state, last_reward, last_action, total_steps, context):
    result = execute_client_request(session_id, '''
        mutation onDone($lastState: [Float!]!, $lastReward: [Float!]!, $lastAction: [Float!]!, $totalSteps: Int!, $context: String) {
            onDone(lastState: $lastState, lastReward: $lastReward, lastAction: $lastAction, totalSteps: $totalSteps, context: $context)
        }
    ''', {
        "lastState": last_state, "lastReward": last_reward, "lastAction": last_action, "totalSteps": total_steps, "context": context
    })
    if (result == None):
        return None
    # print("onDone: " + repr(result))
    return result["onDone"]


def run_simulation(config):

    session_id = config[SESSION_ID]
    app_state = get_app_state(session_id)
    app_state[CONFIG] = config

    set_status(session_id, STARTING)

    env = try_make_env(config[ENVIRONMENT_ID])
    if (env == None):
        set_status(
            ERROR, ["Can't load environment: " + config[ENVIRONMENT_ID]])
        return app_state[STATUS]
    app_state[ENVIRONMENT] = env

    agents = config[AGENTS]
    uri = agents[0][URI]
    token = agents[0][TOKEN]

    client = GraphQLClient(uri)
    client.inject_token("Bearer " + token)
    app_state[CLIENT] = client

    thread = threading.Thread(target=run_episodes, args=(session_id, 99,))
    print("thread: " + repr(thread))
    app_state[THREAD] = thread
    thread.start()

    return app_state[STATUS]


def stop_simulation(session_id):

    app_state = get_app_state(session_id)

    set_status(session_id, STOPPED)

    # Close the env and write monitor result info to disk
    env = app_state[ENVIRONMENT]
    if (env != None):
        env.close()

    thread = app_state[THREAD]
    if (thread != None):
        thread.join()

    return app_state[STATUS]


# --- OpenAI Gym

def try_make_env(environmentId):
    try:
        return gym.make(environmentId)
    except:
        # return retro.make(environmentId)
        print("Invalid environment: " + environmentId)
    return None


def run_episodes(session_id, episode_count):
    try:
        app_state = get_app_state(session_id)
        config = app_state[CONFIG]
        app_state[EPISODE] = 0
        app_state[LAST_ACTION] = (0,)
        app_state[LAST_REWARD] = (0,)
        app_state[TOTAL_REWARD] = (0,)

        set_status(session_id, RUNNING)

        env = app_state[ENVIRONMENT]

        is_training = config[MODE_ID] == TRAINING
        action_space = env.action_space.n
        state_space = env.observation_space.n

        for i in range(episode_count):
            if (app_state[STATUS][CODE] != RUNNING):
                break

            app_state[EPISODE] = i

            done = False
            last_action = (0,)
            last_reward = (0,)

            ob = env.reset()
            print("episode #" + str(i) + ": " + repr(ob))

            agent_reset_result = agent_on_reset(
                session_id, state_space, action_space, session_id, is_training)
            agent_context = agent_reset_result

            step = 0
            while app_state[STATUS][CODE] == RUNNING:
                app_state[STEP] = step
                step += 1

                state = ob
                if (isinstance(ob, np.ndarray)):
                    state = ob.tolist()
                elif (isinstance(ob, np.int64) or isinstance(ob, int)):
                    state = (float(ob),)
                else:
                    print("type of state`: " + repr(type(state)))

                app_state[STATE] = state
                app_state[LAST_ACTION] = last_action
                app_state[LAST_REWARD] = last_reward

                on_step_result = agent_on_step(
                    session_id,
                    state,
                    last_reward,
                    last_action,
                    step,
                    agent_context)

                # print("on_step_result " + repr(on_step_result))

                if (app_state[STATUS][CODE] == ERROR):
                    break

                # last_action = env.action_space.sample()
                last_action = on_step_result[ACTION]
                agent_context = on_step_result[CONTEXT]

                ob, last_reward, done, _ = env.step(last_action[0])
                last_reward = (last_reward,)
                app_state[TOTAL_REWARD] = (app_state[TOTAL_REWARD][0] +
                                           last_reward[0],)

                if done:
                    on_done_result = agent_on_done(
                        session_id,
                        ob,
                        last_reward,
                        last_action,
                        step,
                        agent_context)
                    print("- DONE!")
                    break

                # print("- step = " + str(step) + ", reward = " +
                #       str(last_reward) + ", ob = " + repr(ob))

                # Note there's no env.render() here. But the environment still can open window and
                # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
                # Video is not recorded every episode, see capped_cubic_video_schedule for details.

                # render = env.render('rgb_array')
                # print("- render " + repr(render))

        status = app_state[STATUS]
        if (status[CODE] != ERROR and status[CODE] != STOPPED):
            set_status(session_id, ENDED)

    except Exception as e:
        print("run exception: " + repr(e))
        set_status(session_id, ERROR, [str(e)])

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
  sessionId: ID!
  episodes: Int
  environmentId: ID!
  modeId: ID
  agents: [AgentInput!]!
}

type Config {
  sessionId: ID!
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
  render: String!
  status: Status!
}

type Query {
  listEnvironments: [Environment!]!
  status(sessionId: ID!): Status!
  observe(sessionId: ID!): Observation!
}

type Mutation {
  run(config: ConfigInput!): Status!
  stop(sessionId: ID!): Status!
}

""")


def transformStatus(status):
    return {"id": status["id"], "code": {"id": status["code"]}, "errors": status["errors"]}

# Resolvers are simple python functions
@query.field("listEnvironments")
def resolve_listEnvironments(*_):
    envids = [spec.id for spec in envs.registry.all()]
    res = map(lambda x: {"id": x}, sorted(envids))
    return res


@query.field("status")
def resolve_status(*_, sessionId):
    return transformStatus(get_app_state(sessionId)[STATUS])


@query.field("observe")
def resolve_observe(*_, sessionId):

    app_state = get_app_state(sessionId)

    env = app_state[ENVIRONMENT]

    render = env.render("ansi") if env else ""

    observation = {
        EPISODE: app_state[EPISODE],
        STEP: app_state[STEP],
        AGENT_STATS: ({
            LAST_ACTION: app_state[LAST_ACTION],
            LAST_REWARD: app_state[LAST_REWARD],
            TOTAL_REWARD: app_state[TOTAL_REWARD],
        },),
        DATA: app_state[STATE],
        RENDER: render,
        STATUS: transformStatus(app_state[STATUS])
    }
    # print('observe: ' + repr(observation))
    return observation


@mutation.field("stop")
def resolve_stop(*_, sessionId):
    return transformStatus(stop_simulation(sessionId))


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
    print("... done!")

# Create an ASGI app using the schema, running in debug mode
app = GraphQL(schema, debug=True)

# 'LifespanMiddleware' returns an ASGI app.
# It forwards lifespan requests to 'lifespan',
# and anything else goes to 'app'.
app = LifespanMiddleware(app, lifespan=lifespan)

app.sessions = {}

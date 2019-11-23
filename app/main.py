# --- External imports
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
    status = app.state[STATE_STATUS]
    print('set_sim_status before: ' + repr(status))
    ts = time.time()
    app.state[STATE_STATUS] = {
        ID: "gym@" + str(ts),
        STATUS_CODE: code,
        STATUS_ERRORS: errors}
    status = app.state[STATE_STATUS]
    print('set_sim_status after: ' + repr(status))
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
        STATE_ACTION: (0,),
        STATE_REWARD: 0,
        STATE_STATUS: None
    }
    app.state = state
    set_sim_status(CODE_IDLE)
    print("**** new state created ****")
    return state


def execute_client_request(graphql):
    try:
        client = app.state[STATE_CLIENT]
        if (client == None):
            raise Exception("No client.  Running?")
        result = client.execute(graphql)
        print("result: " + result)
        return result
    except Exception as e:
        print("exception: " + repr(e))
        set_sim_status(CODE_ERROR, [str(e)])
        return None


def agent_on_reset():
    result = execute_client_request('''
        {
            allSimulators {
                id
            }
        }
    ''')
    print("agent_on_reset: " + str(result))
    return str(result)


def agent_on_step(ob, reward):
    result = execute_client_request('''
        {
            allSimulators {
                id
            }
        }
    ''')
    print("agent_on_step: " + str(result))
    return str(result)


def agent_on_end():
    result = execute_client_request('''
        {
            allSimulators {
                id
            }
        }
    ''')
    print("agent_on_end: " + str(result))
    return str(result)

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
    app.state[STATE_ACTION] = (0,)
    app.state[STATE_REWARD] = 0

    set_sim_status(CODE_RUNNING)

    env = app.state[STATE_ENVIRONMENT]
    client = app.state[STATE_CLIENT]

    for i in range(episode_count):
        if (app.state[STATE_STATUS][STATUS_CODE] != CODE_RUNNING):
            break

        app.state[STATE_EPISODE] = i

        reward = 0
        ob = env.reset()
        print("episode #" + str(i) + ": " + repr(ob))

        reset_result = agent_on_reset()
        print("- reset_result: " + repr(reset_result))

        step = 0
        while app.state[STATE_STATUS][STATUS_CODE] == CODE_RUNNING:
            app.state[STATE_STEP] = step
            step += 1

            data = ob
            if (isinstance(ob, np.ndarray)):
                data = ob.tolist()
            else:
                print("type of ob: " + repr(type(ob)))

            app.state[STATE_OBSERVATION] = data

            status = app.state[STATE_STATUS]
            print('- status: ' + repr(status))

            on_step_result = agent_on_step(ob, reward)
            print("- on_step_result: " + repr(on_step_result))

            action = env.action_space.sample()  # agent.act(ob, reward, done)
            print("-- action: " + repr(action))
            print("type of action: " + repr(type(action)))
            act = action
            if (type(action) is int):
                act = (action,)
            app.state[STATE_ACTION] = act

            ob, reward, done, _ = env.step(action)
            if done:
                print("- DONE!")
                break

            print("- step = " + str(step) + ", reward = " +
                  str(reward) + ", ob = " + repr(ob))
            app.state[STATE_REWARD] = reward

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
        action: [Float!]!
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
    return app.state[STATE_STATUS]


@query.field("observe")
def resolve_observe(*_):
    observation = {
        "episode": app.state[STATE_EPISODE],
        "step": app.state[STATE_STEP],
        "action": app.state[STATE_ACTION],
        "reward": app.state[STATE_REWARD],
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
    set_sim_status(CODE_STOPPED)

    # Close the env and write monitor result info to disk
    env = app.state[STATE_ENVIRONMENT]
    if (env != None):
        env.close()

    client = app.state[STATE_CLIENT]

    return app.state[STATE_STATUS]


@mutation.field("run")
def resolve_run(*_, config):
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


# Create executable GraphQL schema
schema = make_executable_schema(type_defs, [query, mutation])

# --- ASGI app

# Create an ASGI app using the schema, running in debug mode
app = GraphQL(schema, debug=True)

# Create shared state on the app object
create_state()

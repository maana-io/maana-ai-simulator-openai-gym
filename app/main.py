from ariadne import ObjectType, QueryType, MutationType, gql, make_executable_schema
from ariadne.asgi import GraphQL
from gym import envs
from graphqlclient import GraphQLClient

# Map resolver functions to Query and Mutation fields
query = QueryType()
mutation = MutationType()

# Define types using Schema Definition Language (https://graphql.org/learn/schema/)
# Wrapping string in gql function provides validation and better error traceback
type_defs = gql("""
    
    type SimStatus {
        id: ID!
        errors: [String!]!
    }
    
    input ConfigInput {
        uri: String!
        token: String!
    }

    type Environment {
        id: ID!
    }

    type Observation {
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
    return app.state["simStatus"]


@query.field("observe")
def resolve_observe(*_):
    return {"simStatus": app.state["simStatus"]}


@query.field("test")
def resolve_test(*_):
    try:
        client = app.state["client"]

        result = client.execute('''
            {
                allSimulators {
                    id
                }
            }
        ''')
        print("result: " + result)
        return result
    except Exception as e:
        print("exception: " + repr(e))
        return str(e)


@mutation.field("run")
def resolve_run(*_, config):
    client = GraphQLClient(config["uri"])
    client.inject_token("Bearer " + config["token"])
    app.state["client"] = client
    app.state["simStatus"] = {"id": "running", "errors": []}

    return app.state["simStatus"]


# Create executable GraphQL schema
schema = make_executable_schema(type_defs, [query, mutation])

# Create an ASGI app using the schema, running in debug mode
app = GraphQL(schema, debug=True)

# Create shared state on the app object
app.state = {"client": None, "simStatus": {"id": "idle", "errors": []}}

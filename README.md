# Speech2Plan

Speech2Plan is the ROS4HRI node that turns what a person says into an
`hri_actions_msgs/Intent`. A nurse says "bring this to the kitchen"; the node
publishes an intent on `/intents` naming the speaker, what was asked for, and
which parts of the request are still open.

This is the first of three stages. Stage two verbalizes the belief state from
the semantic digital twin, stage three builds the plan. 

## Status

Prototype. Parsing works end to end against the Claude command line. The node
follows tracked voices and resolves the speaker to a person id. Gaze, pointing
and engagement are not read yet, so a deictic "this" stays unfilled.

## What it publishes

The output type is `hri_actions_msgs/Intent`:

```
intent     __intent_bring_object__
data       {"goal": "Kitchen", "input": "bring this to the kitchen"}
source     <ROS4HRI person id, or __unknown_agent__>
modality   __modality_speech__
confidence 0.9
```

`source` is the person who spoke. In a
hospital it also decides whether an instruction carries authority, and it is how
a clarification question finds its way back to the person who asked.

The intent names and the thematic roles come from the message definition
itself. `contract.py` reads them off `hri_actions_msgs.msg.Intent`, so there is
one vocabulary and a test that fails if it drifts.

## Topics

```
in   /humans/voices/tracked              hri_msgs/IdsList
in   /humans/voices/<id>/speech          hri_msgs/LiveSpeech
in   /humans/persons/tracked             hri_msgs/IdsList
in   /humans/persons/<id>/voice_id       std_msgs/String
out  /intents                            hri_actions_msgs/Intent
```

Per-voice and per-person subscriptions follow the tracked lists, so ids that
appear and disappear are handled rather than hardcoded.

## Where the world comes from

Grounding needs a `World`. There are two sources behind one interface:

`PrototypeWard` builds a two storey building in process: a kitchen on each
floor, plus a living room and a bathroom. The two kitchens are the point, since
they are what makes "take it to the kitchen" ambiguous. It needs no robot and
no map, so the parser can be run and tested on a laptop.

`FetchedWorld` asks the graph. Any process holding a world exposes it by running
semdt's `FetchWorldServer`, which answers a `std_srvs/Trigger` with the whole
world as a JSON modification list. In a real run that process is the one that
built the map. The client discovers the service by suffix, so it does not need
to know which node owns it, and `is_available()` reports whether anything is
serving one. The node prefers the fetched world and falls back to the prototype,
which is why it runs either way.

What comes back is a snapshot. For a world that keeps changing, semdt's
`Synchronizer` subscribes to further modifications. Speech2Plan does not use it,
because the room list is enough for parsing and rooms do not move.

## Grounding stays separate

The model names an entity type and never a room. It answers `Kitchen`; if the
world holds two kitchens the grounder returns both, and the clarification
question follows from that. 

Two kinds of uncertainty are kept apart. A role the speaker left open is listed
in `unfilled_roles`, and "bring this" lands there until gaze or a question fills
it. A type matching several entities is not uncertainty, it is a fact about the
world, and the answer is to ask.

## Layout

```
src/speech2plan/
  contract.py                      intents and roles, read from hri_actions_msgs
  vocabulary.py                    builds the symbol set from a World
  grounding.py                     constraint + world -> resolved | ambiguous | unmatched
  exceptions.py
  parsing/
    backend.py                     the LanguageModel interface
    claude_cli.py                  prototype backend, shells out to `claude`
    instruction_parser.py          prompt, call, validation
  world_source.py                  PrototypeWard, and the WorldSource interface
  placement.py                     which storey a room is on, which room an object is in
  ros/
    intent_node.py                 the ROS4HRI node
    fetched_world.py               fetches the live world off the graph
scripts/parse_once.py              parse one utterance from the shell
test/
  golden_utterances.py             the eval set
  test_golden_utterances.py        runs the eval set against a model, off by default
  test_contract.py, test_grounding.py, test_placement.py
  test_intent_message.py, test_reply_schema.py
```

Everything outside `ros/` runs without rclpy and without a robot. The message
package imports on its own, so the contract tests need no ROS runtime.
Speech2Plan depends on `semantic_digital_twin` and nothing else from the CRAM
workspace, and the dependency runs one way.

## Running the parser

```bash
source /opt/ros/jazzy/setup.bash
PYTHONPATH=src:$PYTHONPATH python3 scripts/parse_once.py "Bring this to the kitchen"
```

```
intent    : __intent_bring_object__
unfilled  : ['object']
grounding :
  goal       ambiguous between kitchen_ground, kitchen_first
```

Naming the floor resolves the goal, and the world holds two cups:

```
Bring the cup to the kitchen on the first floor
  object     ambiguous between cup, second_cup
  goal       resolved to kitchen_first
```

The backend calls the locally installed `claude` executable atm. 
It is slow per call and carries the command line's own system prompt, so its just for prototyping for now.
An HTTP backend and a local model both satisfy `LanguageModel` without the parser changing, and the deictic case
will want one that accepts images.

## Tests

```bash
source /opt/ros/jazzy/setup.bash
PYTHONPATH=src:$PYTHONPATH python3 -m pytest test -q
```

The grounding tests use mimics instead of a loaded world, so the ambiguity cases
run without a model and without the twin.

`test_golden_utterances.py` calls a model and is skipped unless
`SPEECH2PLAN_RUN_MODEL_TESTS=1`:

```bash
SPEECH2PLAN_RUN_MODEL_TESTS=1 PYTHONPATH=src:$PYTHONPATH python3 -m pytest test -q
```

## Known gaps

The vocabulary covers rooms only, so objects are never resolved. Confidence comes from the model and is not
calibrated. Grounding is not yet in the node's path, so `data` carries entity
types rather than resolved entity ids. There is no clarification question
generator, which is where stage two attaches.

`/humans/persons/<id>/voice_id` is assumed to be `std_msgs/String` per REP-155.
Confirm against the robot before relying on it.

`Room.floor` is the room's floor surface, not the building storey. The storey is
a `Level`, and the twin has no relation tying one to the other, so
`StoreyResolver` compares elevations. Replace it once the real map establishes a
relation; nothing else in grounding changes.

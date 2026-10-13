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
  ros/
    intent_node.py                 the ROS4HRI node
scripts/parse_once.py              parse one utterance from the shell
test/
  golden_utterances.py             the eval set
  test_contract.py, test_grounding.py
```

Everything outside `ros/` runs without rclpy and without a robot. The message
package imports on its own, so the contract tests need no ROS runtime.
Speech2Plan depends on `semantic_digital_twin` and nothing else from the CRAM
workspace, and the dependency runs one way.

## Running the parser

```bash
source /opt/ros/jazzy/setup.bash
PYTHONPATH=src:$PYTHONPATH python3 scripts/parse_once.py "Bring this to the kitchen on the first floor"
```

```
speaker   : __unknown_agent__
intent    : __intent_bring_object__
  goal      : EntityConstraint(entity_type='Kitchen', qualifiers=[Qualifier(attribute='floor', value='FirstFloor')], ...)
unfilled  : ['object']
confidence: 0.9
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

## Known gaps

`load_world()` in the node raises until a world is wired in; `scripts/parse_once.py`
builds a vocabulary by hand meanwhile. The vocabulary covers rooms only, so
objects are never resolved. Confidence comes from the model and is not
calibrated. Grounding is not yet in the node's path, so `data` carries entity
types rather than resolved entity ids. There is no clarification question
generator, which is where stage two attaches.

`/humans/persons/<id>/voice_id` is assumed to be `std_msgs/String` per REP-155.
Confirm against the robot before relying on it.

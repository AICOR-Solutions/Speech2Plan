# Speech2Plan

Speech2Plan is the ROS4HRI node that turns what a person says into an
`hri_actions_msgs/Intent`. A nurse says "bring this to the kitchen"; the node
publishes an intent on `/intents` naming the speaker, what was asked for, and
which parts of the request are still open.

This is the first of three stages. Stage two verbalizes the belief state from
the semantic digital twin, stage three builds the plan. 

## Status

Prototype. Parsing, grounding and the clarification dialogue run end to end on
a laptop: the node asks which kitchen was meant, takes the answer, grounds the
instruction again and publishes the instance. The model behind it is the Claude
command line for now. Gaze, pointing and engagement are not read yet, so a
deictic "this" stays unfilled unless a question fills it.

## What it publishes

The output type is `hri_actions_msgs/Intent`, with no custom message anywhere.
This is "bring the cup to the kitchen" with the cup settled and the kitchen
still open:

```
intent     __intent_bring_object__
data       {"object": "second_cup",
            "input": "bring the cup to the kitchen",
            "other_data": {"constraints": [
              {"role": "object", "status": "resolved", "confidence": 1.0,
               "entity": "second_cup", "entity_type": "Cup",
               "qualifiers": [{"attribute": "name", "value": "second_cup"}],
               "candidates": []},
              {"role": "goal", "status": "ambiguous", "confidence": 0.9,
               "entity": null, "entity_type": "Kitchen", "qualifiers": [],
               "candidates": ["kitchen_ground", "kitchen_first"]}]}}
source     <ROS4HRI person id, or __unknown_agent__>
modality   __modality_speech__
priority   128
confidence 0.9
```

A thematic role is published only once it names a world entity. `goal` is absent
above because two kitchens matched, and a consumer reading the standard keys
cannot tell an entity called Kitchen from the type Kitchen. Every role appears
among the constraints instead, with its own confidence and a status of resolved,
ambiguous, unmatched or unfilled, plus whatever matched. Those travel under
`other_data`, which the message defines for exactly this.

`source` is the person who spoke. In a
hospital it also decides whether an instruction carries authority, and it is how
a clarification question finds its way back to the person who asked.

The intent names and the thematic roles come from the message definition
itself. `contract.py` reads them off `hri_actions_msgs.msg.Intent`, so there is
one vocabulary and a test that fails if it drifts.

## What ROS4HRI already covers

The task, the constraints and the dialogue all fit standard interfaces:

| | Interface |
|---|---|
| the utterance | `hri_msgs/LiveSpeech` on the person's voice topic |
| the speaker | the REP-155 person id, in `Intent.source` |
| the task | `Intent.intent` plus the thematic roles in `Intent.data` |
| the constraints | `other_data` inside the same JSON object |
| saying and asking | `/skill/say` and `/skill/ask` |
| asking for help | `/skill/ask_human_for_help` |

The constraint block is an extension, not a replacement: a consumer that
ignores `other_data` still reads a valid intent, and acts on roles that name
instances. What ROS4HRI does not cover is the belief state and composite
tasks. ROS4HRI has no
belief representation, so the twin holds it and verbalisation leaves as speech.
The skills catalogue defines single skills, so "bring the cup upstairs" has no
standard action to be submitted as; that is the one interface the plan executive
side still has to settle. `Intent` also has no header and no id field, so a task
id, if one is needed, belongs in `other_data` too.

## Topics

```
in   /humans/voices/tracked              hri_msgs/IdsList
in   /humans/voices/<id>/speech          hri_msgs/LiveSpeech
in   /humans/persons/tracked             hri_msgs/IdsList
in   /humans/persons/<id>/voice_id       std_msgs/String
out  /intents                            hri_actions_msgs/Intent
```

None of those subscriptions are ours. `hri.HRIListener` from `libhri` holds
them, and the node asks it three things:

```python
self.listener.on_voice(self.on_voice)        # a voice has started being tracked
voice.on_speech(...)                         # that voice said something final
self.listener.tracked_persons                # who is here, and whose voice is whose
```

There is no way to enumerate the per-id topics and no need to: REP-155 publishes
the ids in the tracked lists, every other name follows from an id, and libhri
turns that into objects. It also gets the QoS right per topic, which is the part
that fails silently. The tracked lists are volatile, republished by
`hri_person_manager` at its `rate` parameter, 30 Hz by default. The person
feature bindings are latched, published once when the association is made, so
`/humans/persons/<id>/voice_id` has to be subscribed `TRANSIENT_LOCAL`. A
volatile subscription to it is compatible, receives nothing, logs nothing, and
leaves every utterance attributed to `__unknown_agent__`, which is exactly the
bug the hand-written version had.

The listener spawns its own node, which this one spins from a timer rather than
letting it run a thread of its own, so every callback belongs to one executor
and a blocking dialogue cannot be interrupted by a second one. `main` spins a
`MultiThreadedExecutor`, because the dialogue blocks inside that timer while the
action clients need threads of their own.

## Where the world comes from

Grounding needs a `World`. There are two sources behind one interface:

`PrototypeWard` builds a two storey building in process: a kitchen on each
floor, plus a living room and a bathroom. The nurse stands on the ground floor,
which is what makes "the kitchen" resolve without asking. Two of the three cups
are on her floor in different rooms, so words separate those, while the two
plates share one kitchen and nothing but perception can. It needs no robot and
no map, so the parser can be run and tested on a laptop.

## What the floor settles

A nurse who says "the kitchen" in a two kitchen building means the one she is
standing in. `AgentStoreyPreference` narrows the candidates to the storey the
`Agent` annotations of the twin are on, which is the robot and whoever
perception has put in the world, and resolves the role when one candidate is
left. The intent records it:

```
{"role": "goal", "status": "resolved", "entity": "kitchen_ground",
 "assumed": true, "candidates": ["kitchen_first"]}
```

`assumed` says the words did not settle it and `candidates` says what was ruled
out, so nothing downstream has to trust the guess blindly. The robot also says
what it ruled out, and the confirmation that follows is what makes the
assumption safe:

```
I understood: I will bring a cup to the kitchen on the ground floor.
There is also the kitchen on the first floor.
```

`NoAssumption` is the other `CandidatePreference`, and with it every ambiguity
reaches the dialogue.

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

## Asking, through ROS4HRI skills

The node conducts no dialogue of its own. When an instruction cannot be
grounded it calls the standard communication skills:

| Skill | Action type | For |
|---|---|---|
| `/skill/say` | `communication_skills/action/Say` | stating what was understood |
| `/skill/ask` | `communication_skills/action/Ask` | asking which one |

`ask` takes a JSON schema for the answers and returns structured JSON, so a
reply needs no second parse.

Three things trigger a question, and they produce different schemas: an
ambiguous role gives an enum of the options, an unmatched one asks where to
look, and an unfilled one asks which is meant. Whatever happened, the task is
then put up for confirmation before it is published, so the nurse always hears
what the robot is about to do and can say no:

```
I will bring the cup in the living room on the ground floor to the kitchen on the ground floor. Is that correct?
  {"confirmed": {"type": "boolean"}}
```

Nothing is published until that comes back true. `confirm_before_acting=False`
turns it off for tests and for a caller that confirms elsewhere.

`DialogueSkills` is the interface. `Ros4HriDialogueSkills` calls the actions,
and tests use a fake that answers from a script.

## Questions a person can answer

A question never contains an entity id. `kitchen_ground` and `second_cup` are
names the twin uses internally, and asking "cup or second_cup" puts the robot's
bookkeeping on the nurse. `description.py` turns an entity into the words a
person would use for it: a room is told apart by its storey, an object by the
room it stands in, and the twin's own `NaturalLanguageDescription` wins over
both when the entity carries one.

```
Which kitchen do you mean: the one on the ground floor, or the one on the first floor?
  {"goal": {"type": "string", "enum": ["the one on the ground floor", "the one on the first floor"]}}

Which cup do you mean: the one in the kitchen on the ground floor, or the one in the kitchen on the first floor?
```

The option strings are also the enum values, so the answer comes back as one of
them and the clarifier maps it to the entity behind it. An answer in the
person's own words works too, since the describer matches a phrase against what
each entity would be called.

An answer is only ever read against the options of the question it answers. That
matters: "the one on the ground floor" is a sensible answer to "which cup" and
to "which kitchen", and reading it against everything in the world once bound a
kitchen to the object role. When neither the enum value nor the words match,
`AnswerInterpreter` sends the question, its options and what the person said to
the model, which picks one of the options or none. "ground flor" then still
lands, and a model that picks nothing leaves the role open rather than guessing.
On the robot the chatbot behind `/skill/ask` does this mapping itself; the
interpreter is for backends that hand back raw words.

When words cannot separate the candidates, there is no choice to offer. Two
plates on the same table are one entity type in one room, and a nurse cannot
answer "plate or other plate" any more than the robot can ask it:

```
I know of 2 plates and cannot tell them apart in words. Which plate do you mean?
```

Settling that needs perception, not dialogue: which plate the nurse means is
whatever they are looking at or pointing to, and resolving "this one" is what
the vision half of the model is for. Until it is wired up, the role stays
ambiguous and the intent says so.

## What the robot says before it asks

The task comes first, as a sentence rather than an intent constant. A role that
matched several entities stays indefinite, a resolved one is named, and
`SpokenTask` holds one pattern per intent.

Then `belief.py` adds what the twin holds about the parts the robot cannot act
on. Only those parts: a resolved role is already named in the task, so it says
nothing further about it.

```
I understood: I will bring a cup to a kitchen.
I know of 2 cups: one in the kitchen on the ground floor and one in the kitchen on the first floor.
I know of 2 kitchens: one on the ground floor and one on the first floor.

I understood: I will bring a plate to a bathroom on the ground floor.
I know of 2 plates, all in the kitchen on the ground floor.
I know of 1 bathroom, the one on the first floor.
```

Namesakes in one place are counted and placed together, since that is the fact
that makes them unanswerable. An entity the world does not hold is answered with
what it does hold of that kind, which is what the following question needs. Past
`most_entities` the report counts instead of listing, so a cupboard of cups does
not become a monologue. It all goes out in one `say` call.

## What comes back goes in again

The answer arrives under the key it was asked for, so `{"object": "the one in
the kitchen on the first floor"}` needs no second parse. `Clarifier.clarify`
rebuilds the instruction from it: the
role gets a `name` qualifier pinning that instance, a role the speaker left open
becomes a binding, the answered role is scored 1.0, and grounding runs again.
An answer naming something the world does not hold changes nothing, so the
question comes back, at most `maximum_rounds` times. A refused confirmation
withdraws the instruction and the node publishes nothing.

A skill call blocks inside the speech callback, so the node puts its
subscriptions in a reentrant callback group and `main` spins a
`MultiThreadedExecutor`. The clarifier is optional. Without one the node grounds
in silence, and when the skill servers are missing it logs that and publishes
the intent as parsed.

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
  description.py                   entities in the words a person uses for them
  belief.py                        what the twin holds about the open roles
  parsing/answer_interpreter.py    reads an answer that matches no option literally
  intent_message.py                the published Intent, roles and constraints
  exceptions.py
  parsing/
    backend.py                     the LanguageModel interface
    claude_cli.py                  prototype backend, shells out to `claude`
    instruction_parser.py          prompt, call, validation
  dialogue.py                      questions from grounding outcomes, answers back in
  world_source.py                  PrototypeWard, and the WorldSource interface
  placement.py                     which storey a room is on, which room an object is in
  ros/
    intent_node.py                 the ROS4HRI node, built on hri.HRIListener
    fetched_world.py               fetches the live world off the graph
    skill_client.py                calls /skill/say and /skill/ask
scripts/
  parse_once.py                    parse one utterance from the shell
  clarify_once.py                  parse it, then clarify it in the terminal
  console_skills.py                serves /skill/say and /skill/ask on the terminal
  dialogue_roundtrip.py            node, fake skill servers and a probe in one process
test/
  golden_utterances.py             the eval set
  test_golden_utterances.py        runs the eval set against a model, off by default
  test_contract.py, test_grounding.py, test_placement.py
  test_dialogue.py, test_node_dialogue.py, test_description.py, test_belief.py
  test_intent_message.py, test_reply_schema.py, test_api_key.py
```

Everything outside `ros/` runs without rclpy and without a robot. `ros/` needs
the workspace sourced, for `pyhri` and the skill action types. The message
package imports on its own, so the contract tests need no ROS runtime.
Speech2Plan depends on `semantic_digital_twin` and nothing else from the CRAM
workspace, and the dependency runs one way.

## Running the parser

```bash
source /opt/ros/jazzy/setup.bash
python3 scripts/parse_once.py "Bring this to the kitchen"
```

Every script puts `src` on the path itself, so none of them needs `PYTHONPATH`.
What they do need is ROS: `hri_actions_msgs` for the contract, and for the ones
that talk to the graph also `~/aicore/install/setup.bash`, which is where
`pyhri` and the skill interfaces live. An IDE run configuration has to carry
that environment too, which is what `scripts/run_node.sh` exists for.

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

## Running the dialogue

`clarify_once.py` puts the questions on the terminal and reads the answers from
stdin, so the loop can be exercised without a robot:

```bash
python3 scripts/clarify_once.py "Bring the cup to the kitchen"
```

```
robot: I understood: I will bring a cup to a kitchen. I know of 2 cups: ...
robot: Which cup do you mean: the one in the kitchen on the ground floor, or the one in the kitchen on the first floor?
       schema {"object": {"type": "string", "enum": ["the one in the kitchen on the ground floor", ...]}}
you  : the one in the kitchen on the first floor
robot: Which kitchen do you mean: the one on the ground floor, or the one on the first floor?
you  : first floor
grounding :
  object     resolved to second_cup
  goal       resolved to kitchen_first
```

`dialogue_roundtrip.py` does the same over ROS. It runs the node, a stand-in for
`/skill/say` and `/skill/ask` that answers from a script, and a probe that
announces one tracked person and utters one sentence. With no argument the parse
is canned and no model is called; with an utterance the real parser runs.

```bash
source ~/aicore/install/setup.bash
python3 scripts/dialogue_roundtrip.py
```

```
person: bring the cup to the kitchen
robot : I understood: I will bring a cup to a kitchen. I know of 2 cups: one in the kitchen
        on the ground floor and one in the kitchen on the first floor. I know of 2 kitchens:
        one on the ground floor and one on the first floor.
robot : Which cup do you mean: the one in the kitchen on the ground floor, or the one in the kitchen on the first floor?
person: {'object': 'the one in the kitchen on the first floor'}
robot : Which kitchen do you mean: the one on the ground floor, or the one on the first floor?
person: {'goal': 'the one on the first floor'}
intent: __intent_bring_object__
data  : {"object": "second_cup", "goal": "kitchen_first", ...}
source: person_demo
```

This is the one check that covers the action clients, the callback groups and
the executor, which no unit test can reach.

## Running the node by hand

The node itself, driven over the real topics. Four terminals.
`scripts/run_node.sh` sources ROS and the workspace by itself, so terminal 2
needs nothing prepared; the others want both setup files sourced.

```bash
# 1  the robot's mouth and ears, on your terminal
python3 scripts/console_skills.py

# 2  the node
./scripts/run_node.sh

# 3  what the robot publishes
ros2 topic echo /intents hri_actions_msgs/msg/Intent --full-length
```

The fourth terminal plays `hri_person_manager` and the speech recogniser. The
first three keep publishing, so leave them running, each in its own shell or
backgrounded with `&`:

```bash
ros2 topic pub -r 1 /humans/persons/tracked hri_msgs/msg/IdsList "{ids: [nurse_1]}"
ros2 topic pub -r 1 /humans/voices/tracked hri_msgs/msg/IdsList "{ids: [voice_1]}"
ros2 topic pub -r 1 --qos-durability transient_local --qos-reliability reliable \
  /humans/persons/nurse_1/voice_id std_msgs/msg/String "{data: voice_1}"
```

Then say something, once:

```bash
ros2 topic pub --once -w 1 /humans/voices/voice_1/speech hri_msgs/msg/LiveSpeech \
  "{final: 'bring the cup to the kitchen', locale: 'en_US'}"
```

Terminal 1 asks which cup and then asks you to confirm, terminal 3 prints the
intent with `source: nurse_1`. The binding has to be published with
`transient_local`, because that is how the person manager publishes it, and
`ros2 topic pub --once` would drop it again when it exits. The type in the echo
command is needed because `/intents` has no publisher until the node is up.

## Tests

```bash
source /opt/ros/jazzy/setup.bash
python3 -m pytest test -q
```

68 tests, none of which needs a model or a robot. `test_dialogue.py` covers the
questions and the answers going back in, `test_node_dialogue.py` drives the node
with a stand-in listener, a canned parse and a recording publisher, so the path
from a tracked voice to a published intent is tested without spinning anything.

If the laptop is configured for the robot, CycloneDDS logs a failed write per
port, several times a second, while the robot is off the network:
`tev: ddsi_udp_conn_write to udp/10.0.0.37:8410 failed with retcode -3`. The
static peer is unreachable, usually because the Ethernet profile is down and the
interface has no address. It goes to stderr through Cyclone's error sink, which
`<Tracing>` does not control, so `<Verbosity>none</Verbosity>` in the config
does not stop it. `CYCLONEDDS_URI=` in front of the command drops the peer
along with the rest of the config, and `2>/dev/null` discards the noise while
keeping the output.

`test_golden_utterances.py` calls a model and is skipped unless
`SPEECH2PLAN_RUN_MODEL_TESTS=1`:

```bash
SPEECH2PLAN_RUN_MODEL_TESTS=1 python3 -m pytest test -q
```

## Known gaps

Confidence comes from the model and is not calibrated. The semantic constraints
are `floor`, `room` and `name`, with no spatial relations, negation or quantity.
Nothing fills `CompletionRequest.images` yet, so the vision half of the design
is unexercised, which is the same gap as the unfilled deictic "this" and as two
plates the robot cannot tell apart in words.

`goal` is where something goes, a room or a surface, and `recipient` is a person
who receives something. The intents table in `Intent.msg` lists `recipient` as
required for `BRING_OBJECT` and `PLACE_OBJECT`, but the examples in its own
header use `goal` for a place, so the table and the examples disagree. This node
follows the examples.

`/skill/ask` has only run against the local stand-in: on TIAGo only
`/skill/say` and `/skill/chat` were seen, so the ask path on the robot is
unverified.

`libhri` offers more than the node reads. `person.engagement_status`,
`person.anonymous`, `person.transform` and the face and body features are all
there, and gating on engagement or filling a deictic "this" from gaze is a
matter of using them rather than subscribing to anything.

`Room.floor` is the room's floor surface, not the building storey. The storey is
a `Level`, and the twin has no relation tying one to the other, so
`PlacementResolver` compares elevations. Replace it once the real map
establishes a relation; nothing else in grounding changes.

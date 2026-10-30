# Speech2Plan

A ROS4HRI node that turns what a person says into an `hri_actions_msgs/Intent`.
It parses the utterance with an LLM, grounds it in the semantic digital twin,
clarifies what is open through the ROS4HRI skills, confirms the task with the
speaker, and publishes it on `/intents`. No custom messages.

Prototype. The LLM is the `claude` command line.

## How it works

1. `hri.HRIListener` reports each tracked voice and its final speech, and which
   person the voice belongs to.
2. World changes from other processes are applied, then the LLM gets the
   utterance and one line per entity type in the world (rooms, furniture,
   graspable objects, not parts such as handles), with what sets the instances
   of a type apart, but no instance names:
   `- Cup (5): on Table floor=GroundFloor | in LivingRoom | on CounterTop | ...`.
   It answers with an intent and, per role, a description: a type, a `floor`,
   and relations to other described entities (`on` a Table, `in` a Bedroom or a
   Cabinet). Types and floors are restricted to those in the world; a thing the
   speaker leaves unnamed ("the thing on the sofa") is a `Thing`, which any
   movable thing fits, and a thing of no known type keeps only the speaker's
   words. A reply that breaks the rules
   goes back to the LLM once, with what was wrong. With `query_world` the LLM
   gets only the type names and queries the world instead, see below.
3. The grounder turns each description into an entity query (krrood's EQL): one
   variable over the instances of the type, one more per relation over the
   instances of the reference, joined by `SupportedBy` / `InsideOf` and the
   twin's storage spaces. It resolves the role to one instance, or marks it
   ambiguous, unmatched or unfilled. Several candidates on different floors are
   settled by the floor the people and the robot are on, recorded as `assumed`.
4. The robot says what it understood and what the twin holds about the open
   parts (`/skill/say`), asks each question once in words a person can answer
   (never entity ids), and takes the next sentence a tracked voice speaks as the
   answer, feeding it back through the LLM and the grounder. Namesakes the room
   cannot tell apart are offered by what they stand on or lie in ("the one on the
   counter top"). For something the twin does not hold it says so ("I know of no
   gloves at all") and asks where it is: an answer naming a place is added to the
   description, which may then find it, or else is published for the executive to
   look there. It does not search itself. Fillers such as
   "huh", which TIAGo's speech recognition makes of noise, are no answer and no
   instruction. TIAGo's `/skill/ask` is not used: its
   only chatbot, Rasa, cannot fill an answer schema, so it never returns.
5. It asks for confirmation, stating the task again only if the answers changed
   it, and publishes only if the speaker confirms.
   Without reachable skill servers nothing is published.

## Interface

```
in   /humans/voices/tracked, /humans/voices/<id>/speech     via libhri
in   /humans/persons/tracked, /humans/persons/<id>/voice_id via libhri
call /skill/say                                             communication_skills
in   /humans/voices/<id>/speech                           answers to its questions
out  /intents                                               hri_actions_msgs/Intent
```

```
intent     __intent_bring_object__
data       {"object": "second_cup", "input": "bring the cup to the kitchen",
            "other_data": {"constraints": [
              {"role": "object", "status": "resolved", "entity": "second_cup", ...},
              {"role": "goal", "status": "ambiguous", "entity": null,
               "entity_type": "Kitchen", "candidates": ["kitchen_ground", "kitchen_first"]}]}}
source     <ROS4HRI person id, or __unknown_agent__>
```

A role key appears in `data` only once it names a world entity. Every role,
with status, confidence, qualifiers, relations, the speaker's words and
candidates, is under `other_data`, so a consumer that ignores it still reads a
valid intent. An unmatched role whose place the speaker gave carries it as a
relation to a named entity:

```
{"role": "object", "status": "unmatched", "entity_type": null,
 "spoken_form": "the gloves",
 "relations": [{"relation": "on", "reference": {"entity_type": "SideTable",
   "qualifiers": [{"attribute": "name", "value": "bedside_table"}], ...}}], ...}
```

## Worlds

- `FetchedWorld`: the live world from semdt's `FetchWorldServer`, used when one is
  served, then kept current by a `WorldSynchronizer` whose updates are applied
  before each utterance
- `CareWard`: the fallback, the prototype ward plus a patient room, a lounge,
  furniture (a counter, sofas, a bedside table, a bathroom cabinet) and everyday
  objects resting on or lying in it, several of them duplicated
- `PrototypeWard`: the small two storey ward the tests use
- `ElevatorBuilding`: the CRAM elevator test building with named rooms and a TIAGo

## Planning

Speech2Plan only publishes intents. Planning and carrying them out happens in
CRAM, in `experiments.intent_execution`: its executor waits on
`/speech2plan/intents`, turns `MOVE_TO`, `GRAB_OBJECT` and `BRING_OBJECT` into
coraplex plans and performs them, on the real robot or in a simulated two storey
building it serves for this node to fetch:

```bash
python -m experiments.intent_execution.executor   # in the CRAM environment
```

## Running

```bash
python3 scripts/parse_once.py "Bring this to the kitchen"     # parse only
python3 scripts/clarify_once.py "Bring the cup to the kitchen" # dialogue on the terminal
python3 scripts/evaluate.py                                    # golden utterances, per stage
python3 scripts/dialogue_roundtrip.py                          # node + stand-in skills over DDS
```

The node over real topics, one command per terminal:

```bash
python3 scripts/console_skills.py      # serves /skill/say and /skill/ask on the terminal
./scripts/run_node.sh                  # the node
./scripts/run_node.sh --ros-args -p query_world:=true   # the node, LLM queries the world
ros2 topic echo /intents hri_actions_msgs/msg/Intent --full-length

# on TIAGo, PAL's communication_hub publishes on /intents too, so remap ours:
./scripts/run_node.sh --ros-args -r /intents:=/speech2plan/intents
ros2 topic echo /speech2plan/intents hri_actions_msgs/msg/Intent --full-length

ros2 topic pub -r 1 /humans/persons/tracked hri_msgs/msg/IdsList "{ids: [nurse_1]}"
ros2 topic pub -r 1 /humans/voices/tracked hri_msgs/msg/IdsList "{ids: [voice_1]}"
ros2 topic pub -r 1 --qos-durability transient_local --qos-reliability reliable \
  /humans/persons/nurse_1/voice_id std_msgs/msg/String "{data: voice_1}"
ros2 topic pub --once -w 1 /humans/voices/voice_1/speech hri_msgs/msg/LiveSpeech \
  "{final: 'bring the cup to the kitchen', locale: 'en_US'}"
```

## Tests

```bash
python3 -m pytest test -q                                   # no model, no robot
SPEECH2PLAN_RUN_MODEL_TESTS=1 python3 -m pytest test -q     # plus the golden utterances
```

`scripts/evaluate.py` scores the golden utterances of `speech2plan.evaluation`
on the care ward stage by stage (parsed, intent, entity types, unfilled roles,
grounding), so a prompt or model change shows where it helps or hurts:

```bash
python3 scripts/evaluate.py --openrouter openai/gpt-4.1-mini
python3 scripts/evaluate.py --query-world
```

The gated golden test runs both parsers; a case marked `needs_query` is an
expected failure for the one that reads a listing.

## Querying the world

`QueryingInstructionParser` does not list the world in the prompt, only its
type names, floors and relations, so the prompt stays one size however much the
world holds. Before answering, the LLM may put up to three queries, each a reply
of its own in the shape of a role description:

```
{"query": {"find": {"entity_type": "Cup"}}}
<result>Cup: 5 matches
- on Table floor=GroundFloor: cup on the table in the kitchen on the ground floor
- in LivingRoom: cup in the living room on the ground floor
...</result>

{"query": {"contents": {"entity_type": "SideTable"}}}
<result>SideTable: 1 match
- side table in the bedroom on the ground floor: on it 1 Cup, 1 Book, 1 Bottle</result>
```

`find` lists the matches with what singles each out, `contents` what rests on or
lies in them. Both are entity queries answered by the grounder (`WorldQueries`),
never with instance names. The loop is plain text over `CompletionRequest`, the
transcript resent on every call, so it works with the `claude` command line as
with OpenRouter. The prompt tells the LLM to query only when the words do not
plainly name a type, and never to single out an instance the speaker did not.

On the 14 golden utterances (claude command line, care ward) both parsers pass
the 13 that name their types plainly, nearly all in one call; the querying one
asks what is on the coffee table for "the thing from the coffee table". "Bring my pills from the
nightstand" fails with the listing, where nothing says what holds pills, and
passes with queries: the LLM asks what is on the side table, finds a Bottle and
answers `Bottle on SideTable`. The listing prompt grows with the world (3.4k
characters for the prototype ward, 4.9k with 100 more objects), the querying
prompt stays at 4.8k; the querying parser also skips describing every instance
when the world changes, which takes seconds in a large world.
If CycloneDDS floods
stderr with `ddsi_udp_conn_write ... retcode -3`, the robot peer is unreachable;
prefix `CYCLONEDDS_URI=` or append `2>/dev/null`.

## Known gaps

- Confidence is not calibrated; descriptions are a `floor` and `on` / `in`
  relations, no directions such as "left of the plate" yet
- Language only: a deictic "this", or two identical objects on one surface, stays ambiguous
- Nothing submits the task to the plan executive by ROS 2 action yet, and there
  is no assistance dialogue

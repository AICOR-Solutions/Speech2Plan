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
   utterance and one line per referable entity in the world (rooms, furniture,
   graspable objects, not parts such as handles; type, name, floor or room). It answers with an intent and, per role, an
   entity type and qualifiers. The type is restricted to those in the world.
3. The grounder resolves each role to an instance, or marks it ambiguous,
   unmatched or unfilled. Several candidates on different floors are settled by
   the floor the people and the robot are on, recorded as `assumed`.
4. The robot says what it understood and what the twin holds about the open
   parts (`/skill/say`), asks each question once in words a person can answer
   (never entity ids), and takes the next sentence a tracked voice speaks as the
   answer, feeding it back through the LLM and the grounder. Fillers such as
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
with status, confidence, qualifiers and candidates, is under `other_data`, so a
consumer that ignores it still reads a valid intent.

## Worlds

- `FetchedWorld`: the live world from semdt's `FetchWorldServer`, used when one is
  served, then kept current by a `WorldSynchronizer` whose updates are applied
  before each utterance
- `CareWard`: the fallback, the prototype ward plus a patient room, a lounge and
  everyday objects (books, bottles, fruit, a towel, sofas), several of them duplicated
- `PrototypeWard`: the small two storey ward the tests use
- `ElevatorBuilding`: the CRAM elevator test building with named rooms and a TIAGo

## Planning

`planning.PlanBuilder` stands in for the `/intents` consumer. It turns
`MOVE_TO`, `GRAB_OBJECT` and `BRING_OBJECT` into coraplex plans, riding the
elevator between floors, and `KinematicSimulation` runs them. Each step carries
a summary, which `plan_once.py` prints:

```
plan:
  1. pick up cup in kitchen_ground
  2. ride the elevator to the first floor
  3. put cup on bedroom_first_table
```

 Needs the
optional `planning` extra (`coraplex`); the node does not import it.

## Running

```bash
python3 scripts/parse_once.py "Bring this to the kitchen"     # parse only
python3 scripts/clarify_once.py "Bring the cup to the kitchen" # dialogue on the terminal
python3 scripts/plan_once.py "Go to the bedroom upstairs"      # parse, clarify, plan, simulate
python3 scripts/dialogue_roundtrip.py                          # node + stand-in skills over DDS
```

The node over real topics, one command per terminal:

```bash
python3 scripts/console_skills.py      # serves /skill/say and /skill/ask on the terminal
./scripts/run_node.sh                  # the node
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

`test_planning.py` takes about two and a half minutes. If CycloneDDS floods
stderr with `ddsi_udp_conn_write ... retcode -3`, the robot peer is unreachable;
prefix `CYCLONEDDS_URI=` or append `2>/dev/null`.

## Known gaps

- Carrying something to another floor fails in coraplex: its occupancy costmap
  samples standing poses at z=0 (expected failure in `test_planning.py`)
- Confidence is not calibrated; constraints are only `floor`, `room` and `name`
- Language only: a deictic "this", or two identical objects in one place, stays ambiguous
- Nothing submits the task to the plan executive by ROS 2 action yet, and there
  is no assistance dialogue

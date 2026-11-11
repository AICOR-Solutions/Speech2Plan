# Speech2Plan

A ROS4HRI node that turns what a person says into an `hri_actions_msgs/Intent`.
It parses the utterance with an LLM, grounds it in the semantic digital twin,
clarifies what is open through the ROS4HRI skills, confirms the task with the
speaker, and hands it to the plan executive as a `CarryOutIntent` goal. While
the robot carries a task out, the plan executive asks people for help through this
node's `AskForHelp` skill. Those two actions, in `speech2plan_msgs`, are its only
interfaces of its own.

The intents are those of the medicine-transport mission (RA-FETCH D2.1): going
somewhere (`MOVE_TO`) and bringing something (`BRING_OBJECT`); putting something
somewhere or fetching it is bringing it. Anything else is `RAW_USER_INPUT` and
dropped. Only places are grounded: people put what the robot brings on its tray
and take it off again, so the item is never looked for and stays the speaker's
words.

Prototype. The LLM is the `claude` command line.

## How it works

1. `hri.HRIListener` reports each tracked voice and its final speech, and which
   person the voice belongs to.
2. World changes from other processes are applied, then the LLM gets the
   utterance and one line per place type in the world (rooms and furniture, not
   not the things lying about), with what sets the
   instances of a type apart, but no instance names:
   `- Table (2): in Kitchen floor=GroundFloor | floor=FirstFloor`. Places that
   carry a number, as the rooms of the IAI office do, are listed by it
   (`- Room (82): number=0.30 | ... | number=2.58 | ...`), and "room two point
   five eight" becomes a Room with `number=2.58`.
   It answers with an intent, the item in the speaker's words
   (`{"spoken_form": "the insulin"}`), and the goal as a description: a type, a
   `floor` or a `number`, and the room it lies `in`. Types and
   floors are restricted to those in the world; a place of no known type keeps
   only the speaker's words. A reply that breaks the rules goes back to the LLM
   once, with what was wrong.
3. The grounder turns the goal into an entity query (krrood's EQL): one variable
   over the instances of the type, one more over the rooms it is said to lie in,
   joined by which room each place stands in. It resolves the goal to
   one instance, or marks it ambiguous, unmatched or unfilled. Several candidates
   on different floors are settled by the floor the people and the robot are on,
   recorded as `assumed`.
4. The robot says what it understood and what the twin holds about an open goal
   (`/skill/say`), asks each question once in words a person can answer (never
   entity ids), and takes the next sentence a tracked voice speaks as the answer,
   feeding it back through the LLM and the grounder. For a place the twin does not
   hold it says so ("I know of no pharmacy at all") and asks which place is meant;
   an item the speaker left open is asked for and kept in their words. Fillers
   such as "huh", which TIAGo's speech recognition makes of noise, are no answer
   and no instruction. TIAGo's `/skill/ask` is not used: its only chatbot, Rasa,
   cannot fill an answer schema, so it never returns.
5. It asks for confirmation, stating the task again only if the answers changed
   it, and submits it only if the speaker confirms. Without reachable skill
   servers nothing is submitted. Submitting waits until the executive took the
   task on, not until it is done; the steps it reports and how it ended are
   logged. If no executive is there, or it is busy with another task, the robot
   says it cannot take the task on.
6. While the robot carries a task out, the plan executive asks people for help
   through `/skill/ask_for_help`: to put the item on the tray, to take it off,
   whether it may board the elevator with the people there, or whether to try a
   failed step again. This node says the question and reads the
   yes or no from the words alone, without the LLM, so it works where the robot
   has no network, as in the elevator. No instruction dialogue runs while a
   question is out.

## Interface

```
in   /humans/voices/tracked, /humans/voices/<id>/speech     via libhri
in   /humans/persons/tracked, /humans/persons/<id>/voice_id via libhri
call /skill/say                                             communication_skills
in   /humans/voices/<id>/speech                           answers to its questions
call /skill/carry_out_intent                       speech2plan_msgs/CarryOutIntent
serve /skill/ask_for_help                               speech2plan_msgs/AskForHelp
```

Both follow PAL's skill conventions (`std_skills`). `CarryOutIntent`: the goal is
`meta` and the `hri_actions_msgs/Intent`; the feedback names each step as it
starts (`step`, `step_number`, `step_count`) and what the robot does (`RUNNING`,
`WAITING_FOR_PEOPLE`, `ASKING_FOR_HELP` with the `problem`); the result is a
`std_skills/Result` (`ROS_ENOERR`, `ROS_ENOTSUP` for an intent no plan is built
for, `ROS_EBADMSG` for roles naming nothing to act on, `ROS_ECANCELED`,
`ROS_EOTHER` when a step failed and people could not help) and the `records` of
what happened at the tray ("delivered the insulin to 2.58"). `AskForHelp`: the
goal is the `problem` (`load_tray`, `unload_tray`, `board_elevator`,
`step_failed`) and the `text`
to say, ending in a yes or no question; the result is whether the answer was
`yes`. Build them into the colcon workspace next to this repository:

```bash
cd ~/aicore && colcon build --packages-select speech2plan_msgs \
  --base-paths Speech2Plan/speech2plan_msgs && source install/setup.bash
```

```
intent     __intent_bring_object__
data       {"object": "insulin", "goal": "kitchen_first",
            "input": "bring the insulin to the kitchen upstairs",
            "other_data": {"constraints": [
              {"role": "goal", "status": "resolved", "entity": "kitchen_first",
               "entity_type": "Kitchen", "qualifiers": [{"attribute": "floor", ...}], ...}]}}
source     <ROS4HRI person id, or __unknown_agent__>
```

The goal appears in `data` only once it names a world entity; the item always
appears as the speaker's word for it, without "the" or "my", as REP-155's own
example sends `"groceries"`. Every place role, with status, confidence,
qualifiers, relations, the speaker's words and candidates, and every role left
open, is under `other_data`, so a consumer that ignores it still reads a valid
intent.

## Worlds

- `FetchedWorld`: the live world from semdt's `FetchWorldServer`, used when one is
  served, then kept current by a `WorldSynchronizer` whose updates are applied
  before each utterance
- `CareWard`: the fallback, the prototype ward plus a patient room, a lounge and
  furniture (a counter, sofas, a bedside table, a bathroom cabinet)
- `PrototypeWard`: the small two storey ward the tests use, with a cup on its
  table, which is no place

## Planning

Speech2Plan only submits intents. Planning and carrying them out happens in
CRAM, in `experiments.intent_execution`: its demonstration serves
`/skill/carry_out_intent`, one task at a time, turns `MOVE_TO` and `BRING_OBJECT`
into coraplex plans and performs them, on the real robot or in a simulated two
storey building it serves for this node to fetch. A bring drives to the pickup
room, a stand-in for the pharmacy, asks people to put the item on the tray and
goes on once they say they did, drives to the goal, riding the elevator between
storeys, asks people to take it off again and logs the delivery. When a step
fails it asks whether to try again, and on a yes plans anew from where it
stands, without collecting the item again:

```bash
python -m experiments.intent_execution.demo   # in the CRAM environment
```

## Running

```bash
python3 scripts/parse_once.py "Bring this to the kitchen"     # parse only
python3 scripts/clarify_once.py "Bring the insulin to the kitchen" # dialogue on the terminal
python3 scripts/evaluate.py                                    # golden utterances, per stage
python3 scripts/dialogue_roundtrip.py                          # node + stand-in skills over DDS
```

The node over real topics, one command per terminal:

```bash
python3 scripts/console_skills.py      # serves /skill/say on the terminal
python3 scripts/console_skills.py --with-executive   # and prints the submitted tasks
./scripts/run_node.sh                  # the node

ros2 topic pub -r 1 /humans/persons/tracked hri_msgs/msg/IdsList "{ids: [nurse_1]}"
ros2 topic pub -r 1 /humans/voices/tracked hri_msgs/msg/IdsList "{ids: [voice_1]}"
ros2 topic pub -r 1 --qos-durability transient_local --qos-reliability reliable \
  /humans/persons/nurse_1/voice_id std_msgs/msg/String "{data: voice_1}"
ros2 topic pub --once -w 1 /humans/voices/voice_1/speech hri_msgs/msg/LiveSpeech \
  "{final: 'bring the cup to the kitchen', locale: 'en_US'}"
```

## Settings

The prompts and the word lists (filler words, determiners, yes and no words)
live in `src/speech2plan/settings.toml`, read by `speech2plan.settings`. Every
class that uses them takes them as a parameter, defaulting to that file, so a
test or experiment can pass its own.

## Tests

```bash
python3 -m pytest test -q                                   # no model, no robot
SPEECH2PLAN_RUN_MODEL_TESTS=1 python3 -m pytest test -q     # plus the golden utterances
```

`scripts/evaluate.py` scores the golden utterances of `speech2plan.evaluation`
on the care ward stage by stage (parsed, intent, roles: place types and the item
word, unfilled roles, grounding), so a prompt or model change shows where it helps
or hurts. The cases are go-to and delivery requests, an item or a goal left open,
a place the twin lacks and one request outside the mission:

```bash
python3 scripts/evaluate.py --openrouter openai/gpt-4.1-mini
```

If CycloneDDS floods
stderr with `ddsi_udp_conn_write ... retcode -3`, the robot peer is unreachable;
prefix `CYCLONEDDS_URI=` or append `2>/dev/null`.

## Known gaps

- Confidence is not calibrated; a place is described by its type, `floor`,
  `number` and the room it lies `in`, no directions such as "the room next to
  the lift" yet
- Help is a yes or no: people can let the robot try again or stop it, not tell
  it what to do instead, and the executive acts only on roles that name an
  entity (`data`), not on the constraints under `other_data`
- The elevator call has no fallback of its own yet: a drive that fails at the
  elevator is asked about like any failed step, not with "please press the
  button"

"""Vocabulary specific to tabletop_synth (names, extra spells, features, surface variants, extra phrase banks)."""

from __future__ import annotations

from ._borrow import W

PLAYERS = [
    "Bettina", "Calloway", "Desmond", "Elspeth", "Fitz", "Giulia", "Hamish", "Imogen", "Jasper", "Kofi", "Lorna",
    "Mateo", "Nadia", "Osric", "Priya", "Quentin", "Rosalind", "Soren", "Thea", "Ulrich", "Viv", "Wendell",
    "Xiomara", "Yusuf", "Zelda", "Anouk", "Bram", "Cleo", "Dario", "Esme", "Florian", "Greer", "Henrik", "Ingrid",
]
NEAR_PAIRS = [("Jaycee", "Jaycie"), ("Corin", "Corrin"), ("Teagan", "Tegan"), ("Mikayla", "Makayla"),
              ("Rylee", "Riley"), ("Devin", "Devon"), ("Jaimie", "Jamie"), ("Katelyn", "Kaitlyn")]
DM_NAMES = ["Gideon", "Marisol", "Thaddeus", "Leontine", "Ambrose", "Philippa", "Cassius", "Winona"]

EXTRA_LEVEL = {
    "Light": 0, "Prestidigitation": 0, "Message": 0, "Mending": 0,
    "Find Familiar": 1, "Alarm": 1, "Identify": 1, "Comprehend Languages": 1, "Speak with Animals": 1,
    "Tenser's Floating Disk": 1, "Hail of Thorns": 1, "Absorb Elements": 1, "Tasha's Caustic Brew": 1,
    "Melf's Acid Arrow": 2, "Snilloc's Snowball Swarm": 2, "Maximilian's Earthen Grasp": 2,
    "Leomund's Tiny Hut": 3, "Otiluke's Resilient Sphere": 4, "Evard's Black Tentacles": 4, "Bigby's Hand": 5,
}
LEVEL = {**W.SPELL_LEVEL, **EXTRA_LEVEL}
CLASS_EXTRA = {
    "wizard": ["Light", "Prestidigitation", "Find Familiar", "Alarm", "Identify", "Tenser's Floating Disk",
               "Melf's Acid Arrow", "Leomund's Tiny Hut", "Otiluke's Resilient Sphere", "Evard's Black Tentacles",
               "Bigby's Hand", "Absorb Elements", "Tasha's Caustic Brew"],
    "sorcerer": ["Light", "Prestidigitation", "Message", "Snilloc's Snowball Swarm", "Maximilian's Earthen Grasp",
                 "Absorb Elements"],
    "warlock": ["Prestidigitation", "Find Familiar", "Comprehend Languages"],
    "bard": ["Prestidigitation", "Message", "Comprehend Languages", "Identify", "Leomund's Tiny Hut"],
    "cleric": ["Light", "Mending", "Detect Magic"],
    "druid": ["Mending", "Speak with Animals", "Absorb Elements"],
    "ranger": ["Hail of Thorns", "Speak with Animals", "Absorb Elements"],
    "paladin": ["Detect Magic"],
}
RITUALS = {"Detect Magic", "Find Familiar", "Alarm", "Identify", "Comprehend Languages", "Speak with Animals",
           "Leomund's Tiny Hut", "Tenser's Floating Disk"}
REACTIONS = {"Shield", "Counterspell", "Hellish Rebuke", "Absorb Elements"}
PASSING = {
    "Light": ["I cast {s} on the end of my staff so we can see.", "Quick {s} on my dagger, I want to see the floor."],
    "Prestidigitation": ["I use {s} to get the mud off my boots.", "With {s} I warm up this terrible stew."],
    "Mage Hand": ["I send a {s} over to lift the latch.", "Can my {s} grab the key? I cast it."],
    "Message": ["I cast {s} to whisper to {pc}: be careful.", "{s} to {pc}: 'Don't say anything yet.'"],
    "Mending": ["I cast {s} on the torn strap of my pack.", "Let me fix that cracked lantern with {s}."],
    "Guidance": ["I cast {s} on {pc}, just in case.", "{s} on {pc}, for luck."],
    "Thaumaturgy": ["I use {s} to make my voice boom across the room.", "{s} -- the candles flicker when I talk."],
    "Druidcraft": ["I cast {s} to see what the weather will be tomorrow.", "Little {s}, I make a flower bloom."],
    "Minor Illusion": ["I make a {s} of a crate to hide behind.", "I cast {s}: the sound of footsteps down the hall."],
}
SHORT = {"Tasha's Hideous Laughter": "hideous laughter", "Leomund's Tiny Hut": "tiny hut",
         "Tenser's Floating Disk": "floating disk", "Melf's Acid Arrow": "acid arrow",
         "Otiluke's Resilient Sphere": "resilient sphere", "Evard's Black Tentacles": "black tentacles",
         "Bigby's Hand": "Bigby's hand", "Snilloc's Snowball Swarm": "snowball swarm",
         "Maximilian's Earthen Grasp": "earthen grasp", "Tasha's Caustic Brew": "caustic brew"}
ATTACK_SPELLS = W.ATTACK_SPELLS | {"Melf's Acid Arrow", "Bigby's Hand"}
SAVE_SPELLS = W.SAVE_SPELLS | {"Snilloc's Snowball Swarm", "Maximilian's Earthen Grasp", "Evard's Black Tentacles",
                               "Tasha's Caustic Brew", "Hail of Thorns"}
HEAL_SPELLS = W.HEAL_SPELLS

FEATURES = {
    "cleric": {"Channel Divinity: Turn Undead": ["I use my Channel Divinity -- Turn Undead!",
                                                 "Channel Divinity: Turn Undead. I raise my holy symbol."],
               "Channel Divinity: Preserve Life": ["Channel Divinity, Preserve Life, spread across everyone.",
                                                   "I use Channel Divinity to Preserve Life on the wounded."]},
    "druid": {"Wild Shape": ["I Wild Shape into a giant spider.", "I'm going to Wild Shape. Brown bear."]},
    "paladin": {"Channel Divinity: Sacred Weapon": ["Channel Divinity: Sacred Weapon, my {w} starts to glow.",
                                                    "I use my Channel Divinity for Sacred Weapon."]},
}
MARTIAL = {
    "barbarian": ["I rage. Full rage, I scream at the {m}.", "Reckless Attack, I don't care anymore."],
    "fighter": ["Action Surge! I go again.", "I use my Second Wind, that's a bonus action."],
    "monk": ["Flurry of Blows, two more punches.", "I spend a ki point on Stunning Strike."],
    "rogue": ["Cunning Action, I disengage behind the pillar.", "I'll get Sneak Attack on this one, right?"],
    "bard": ["I give {pc} a Bardic Inspiration die.", "Bardic Inspiration to {pc}: 'You got this!'"],
}
PET_KINDS = {"wizard": ["owl", "raven", "cat"], "warlock": ["imp-shaped familiar", "pseudodragon"],
             "ranger": ["wolf", "panther", "hawk"]}
PET_NAMES = ["Thistle", "Morrow", "Pip", "Ember", "Quill", "Bramble", "Soot", "Juniper", "Wick", "Fennel"]

DM_NARR2 = [
    "The wind picks up as you reach {place}; lanterns swing on their hooks and {sound} rises over the rooftops.",
    "{npc_cap} finishes wiping down a mug, sets it on the shelf, and says, \"You're not from around here, are you?\"",
    "You spend the next hour asking around {place}. Most people shrug, but {npc} mentions {obj} went missing.",
    "Okay, so the path narrows here. Roots cross the ground, and somewhere ahead of you there's {sound}.",
    "It's late afternoon by the time you get to {place}. The light is {adj}, everything smells like rain.",
    "Behind the counter, {npc} keeps glancing at the back door like they're expecting someone else to come in.",
    "You notice scratch marks along the doorframe, old ones, and next to them someone carved a little symbol.",
]
PC_TALK2 = [
    "Okay, before we go anywhere, can we talk about what {npc} said? Because that did not sound right to me.",
    "I'm going to sit with {pc} for a bit and ask how they're doing after everything at {place}.",
    "Can I look at {obj} more closely? Like, turn it over, see if there's writing on the bottom of it?",
    "I think we split up. {pc} and I take {place}, everybody else goes and talks to {npc}.",
    "Honestly I just want a bath and a real bed. Is there an inn near {place} that isn't terrible?",
    "Wait, did we ever pay {npc} back? I feel like we owe them money from the last time we were here.",
    "I keep my hand on my weapon the whole time, I don't trust this place at all.",
]
OOC2 = [
    "My dice have been rolling ones all night, I'm putting them in dice jail after this.",
    "Hold on, I need to find my pencil, it rolled under the couch again.",
    "We're about an hour in, right? Okay, I just want to know when the snacks come out.",
    "Did anyone else see that the map has a little dragon drawn in the corner? I love that.",
    "Sorry, my phone keeps buzzing, it's my sister, she wants to know who brought the good chips.",
]

DELIMS = {
    "real": ("[START OF EPISODE]", "[END OF EPISODE]"),
    "session": ("=== SESSION {k} ===", "=== END OF SESSION {k} ==="),
    "hash": ("### Episode {k}", None),
    "dash": ("--- begin transcript {k} ---", "--- end transcript {k} ---"),
    "xml": ('<episode n="{k}">', "</episode>"),
}
DELIM_W = {"real": 0.40, "session": 0.15, "hash": 0.15, "dash": 0.15, "xml": 0.15}
DELIM_DESC = {
    "real": ["{each} delimited by [START OF EPISODE] and [END OF EPISODE]."],
    "session": ["{each} opens with a banner line such as === SESSION 1 === and closes with a matching END OF SESSION "
                "banner."],
    "hash": ["{each} starts with a heading line (### Episode 1, ### Episode 2, and so on)."],
    "dash": ["{each} wrapped in --- begin transcript N --- and --- end transcript N --- lines."],
    "xml": ["{each} enclosed in <episode> ... </episode> tags."],
}
SPEAKER_W = {"colon": 0.5, "upper": 0.2, "bracket": 0.15, "dash": 0.15}
HEADER_W = {"plays": 0.4, "paren": 0.2, "arrow": 0.2, "table": 0.2}
DM_LABEL_W = {"DM": 0.4, "GM": 0.2, "name": 0.4}
INSTR = [
    "The following lines contain {what} of a Dungeons and Dragons game played by a group of friends. {delim} "
    "A question about the game statistics follows. Answer it from the transcripts only. Do not guess, estimate or "
    "approximate the result, and do not ask for clarification. Reason step by step if needed. "
    "Return the final answer in \\boxed{{}}.",
    "Below you will find {what} from a tabletop role-playing campaign (Dungeons & Dragons, fifth edition). {delim} After them "
    "comes a question about dice rolls or spells. Base your answer on the transcripts alone; do not approximate. "
    "Give the final answer in \\boxed{{}}.",
    "You will read {what} of a recorded D&D game night. {delim} Answer the statistics question at the end exactly "
    "-- no estimates, no follow-up questions -- and put the final answer inside \\boxed{{}}.",
]
MAP_INTRO = ["The following lines contain the mapping between player names and character names.",
             "Who plays whom at this table:", "Players and the characters they play are listed below.",
             "Cast list (player and character):"]
MAP_OUTRO = ["Use this mapping when answering the questions below.", "Refer to this list when a question names a "
             "player or a character.", ""]

DELIMS2 = {**DELIMS,
           "title": ("[[Episode {k}: {title}]]", None),
           "words": ("EPISODE {K}", None),
           "dated": ("Session {k} — {date}", None),
           "angle": ("<<< transcript {k} begins >>>", "<<< transcript {k} ends >>>"),
           "prose": ("Here begins the record of session {k}.", "Here ends the record of session {k}.")}
DELIM_W2 = {"real": 0.12, "session": 0.06, "hash": 0.06, "dash": 0.06, "xml": 0.06,
            "title": 0.13, "words": 0.13, "dated": 0.13, "angle": 0.12, "prose": 0.13}
DELIM_DESC2 = {**DELIM_DESC,
               "title": ["{each} starts with a bracketed title line such as [[Episode 1: ...]]."],
               "words": ["{each} starts with a heading in capitals (EPISODE ONE, EPISODE TWO, ...)."],
               "dated": ["{each} starts with a dated session heading."],
               "angle": ["{each} wrapped in <<< transcript N begins >>> and <<< transcript N ends >>> lines."],
               "prose": ["{each} introduced by a sentence announcing the session and closed by one saying it ends."]}
VERBLESS_DESC = {"session", "hash", "title", "words", "dated", "prose"}
NUM_CAPS = ["ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE", "TEN"]
TITLE_A = ["Salt", "Lantern", "Hollow", "Ash", "Copper", "Thorn", "Mire", "Glass", "Ember", "Iron", "Willow", "Fog"]
TITLE_B = ["Road", "Crown", "Bargain", "Bridge", "Well", "Market", "Vigil", "Tide", "Ledger", "Orchard", "Bell"]
DATE_FMT = ["%d %B %Y", "%B %d, %Y", "%Y-%m-%d", "%a %d %b %Y"]

SPEAKER_W2 = {"colon": 0.12, "upper": 0.06, "bracket": 0.06, "dash": 0.06,
              "stamp": 0.14, "charname": 0.14, "bold": 0.14, "quote": 0.14, "wrap": 0.14}
WRAP_P = 0.4
HEADER_W2 = {"plays": 0.12, "paren": 0.1, "arrow": 0.1, "table": 0.1, "prose": 0.2, "json": 0.18, "bullet": 0.2}
PROSE_CLAUSE = ["{p} is playing {c}", "{p} plays {c}", "{c} is played by {p}", "{p} is back as {c}",
                "{p} takes the role of {c}"]
INSTR2 = INSTR + [
    "What follows is {what} from a long-running Dungeons & Dragons campaign. {delim} A single question about dice "
    "rolls or spellcasting comes after the transcripts; answer it exactly from the text and put the final answer in "
    "\\boxed{{}}.",
    "Context: {what} of a fifth-edition D&D group. {delim} Count carefully -- the question at the end needs an "
    "exact answer, given in \\boxed{{}}.",
    "These are {what} of friends playing Dungeons and Dragons. Read them and answer the question that follows "
    "exactly; estimates are not accepted. Final answer in \\boxed{{}}.",
    "Below: {what} of a D&D table. Answer the question about the game statistics using only what is said in them, "
    "and write the final answer as \\boxed{{}}.",
    "Transcribed audio from a tabletop game night follows. Use it to answer the statistics question exactly. "
    "Return the final answer in \\boxed{{}}.",
]
MAP_INTRO2 = MAP_INTRO + ["Table roster:", "Who's who at the table:",
                          "Before the logs, a note on the players and their characters."]

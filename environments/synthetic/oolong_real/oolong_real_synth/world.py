"""Static vocabulary for the synthetic D&D sessions: names, classes, spells, roll types, phrase banks."""

from __future__ import annotations

PLAYER_NAMES = [
    "Avery", "Blake", "Casey", "Dana", "Eli", "Frankie", "Gabe", "Harper", "Iris", "Jordan", "Kai", "Logan",
    "Morgan", "Noel", "Oakley", "Parker", "Quinn", "Reese", "Sage", "Tatum", "Uma", "Wren", "Yara", "Zane",
    "Bryn", "Cole", "Drew", "Emery", "Finn", "Gray", "Hollis", "Indy", "Jules", "Lane", "Mika", "Nico", "Remy",
    "Shay", "Toby", "Vic", "Rory", "Skye", "Teddy", "Mara", "Ollie", "Pia", "Joss", "Ines",
]
DM_NAMES = ["Rowan", "Marcus", "Brennan", "Aabria", "Evan", "Celeste", "Hugo", "Lionel", "Dahlia", "Kendra"]

_SYL_A = ["Bra", "Ilv", "Qui", "Sef", "Dor", "Ka", "Mor", "Tha", "Ves", "Ori", "Zan", "Eli", "Gru", "Fen", "Hal",
          "Ny", "Pell", "Ro", "Syl", "Tor", "Ul", "Wy", "Xan", "Yse", "Cael", "Del", "Ari", "Bel", "Cor", "Lu"]
_SYL_B = ["ra", "vin", "dra", "llo", "mir", "rik", "the", "nna", "zor", "wen", "ga", "ric", "lyn", "dus", "mae",
          "sha", "tor", "quin", "bel", "dax", "ren", "vos", "lia", "mon", "kar", "ith", "sio", "fey", "nox", "ett"]
_SYL_C = ["", "", "", "n", "s", "th", "a", "o", "is", "en", "ar", "us"]

FAMILIAR_NAMES = ["Pebble", "Mister Whiskers", "Gloam", "Sprocket", "Tumble", "Biscuit", "Nib", "Cinder"]
FAMILIAR_KINDS = ["owl", "cat", "raven", "ferret", "frog", "hawk"]

CLASSES: dict[str, dict] = {
    "wizard": {"spells": ["Fire Bolt", "Mage Hand", "Ray of Frost", "Magic Missile", "Shield", "Mage Armor",
                          "Detect Magic", "Sleep", "Scorching Ray", "Misty Step", "Web", "Fireball", "Counterspell",
                          "Haste", "Polymorph", "Dimension Door", "Cone of Cold"],
               "weapons": ["dagger", "quarterstaff"]},
    "cleric": {"spells": ["Sacred Flame", "Guidance", "Toll the Dead", "Thaumaturgy", "Cure Wounds", "Healing Word",
                          "Bless", "Guiding Bolt", "Spiritual Weapon", "Lesser Restoration", "Spirit Guardians",
                          "Revivify", "Banishment", "Mass Healing Word"],
               "weapons": ["mace", "warhammer"]},
    "warlock": {"spells": ["Eldritch Blast", "Minor Illusion", "Hex", "Armor of Agathys", "Hellish Rebuke",
                           "Darkness", "Hold Person", "Misty Step", "Hunger of Hadar", "Counterspell", "Banishment"],
                "weapons": ["pact blade", "sickle"]},
    "druid": {"spells": ["Produce Flame", "Shillelagh", "Druidcraft", "Guidance", "Entangle", "Faerie Fire",
                         "Thunderwave", "Cure Wounds", "Healing Word", "Moonbeam", "Pass without Trace", "Call Lightning",
                         "Wall of Fire", "Polymorph"],
              "weapons": ["scimitar", "staff"]},
    "bard": {"spells": ["Vicious Mockery", "Mage Hand", "Minor Illusion", "Dissonant Whispers", "Healing Word",
                        "Charm Person", "Tasha's Hideous Laughter", "Faerie Fire", "Suggestion", "Shatter",
                        "Hypnotic Pattern", "Dimension Door"],
             "weapons": ["rapier", "hand crossbow"]},
    "sorcerer": {"spells": ["Fire Bolt", "Ray of Frost", "Mage Hand", "Chromatic Orb", "Witch Bolt", "Shield",
                            "Magic Missile", "Scorching Ray", "Mirror Image", "Misty Step", "Fireball",
                            "Lightning Bolt", "Haste"],
                 "weapons": ["light crossbow", "dagger"]},
    "paladin": {"spells": ["Divine Favor", "Bless", "Shield of Faith", "Thunderous Smite", "Cure Wounds",
                           "Find Steed", "Aura of Vitality", "Revivify"],
                "weapons": ["longsword", "glaive"]},
    "ranger": {"spells": ["Hunter's Mark", "Ensnaring Strike", "Cure Wounds", "Pass without Trace", "Spike Growth",
                          "Conjure Barrage"],
               "weapons": ["longbow", "shortsword"]},
    "fighter": {"spells": [], "weapons": ["greatsword", "falchion", "heavy crossbow"]},
    "barbarian": {"spells": [], "weapons": ["greataxe", "maul", "handaxe"]},
    "rogue": {"spells": ["Mage Hand", "Disguise Self", "Charm Person"], "weapons": ["rapier", "shortbow", "dagger"]},
    "monk": {"spells": [], "weapons": ["quarterstaff", "fists", "darts"]},
}
CASTER_CLASSES = ["wizard", "cleric", "warlock", "druid", "bard", "sorcerer", "paladin", "ranger"]
MARTIAL_CLASSES = ["fighter", "barbarian", "rogue", "monk"]

SPELL_LEVEL: dict[str, int] = {
    "Fire Bolt": 0, "Mage Hand": 0, "Ray of Frost": 0, "Sacred Flame": 0, "Guidance": 0, "Toll the Dead": 0,
    "Thaumaturgy": 0, "Eldritch Blast": 0, "Minor Illusion": 0, "Produce Flame": 0, "Shillelagh": 0,
    "Druidcraft": 0, "Vicious Mockery": 0,
    "Magic Missile": 1, "Shield": 1, "Mage Armor": 1, "Detect Magic": 1, "Sleep": 1, "Cure Wounds": 1,
    "Healing Word": 1, "Bless": 1, "Guiding Bolt": 1, "Hex": 1, "Armor of Agathys": 1, "Hellish Rebuke": 1,
    "Entangle": 1, "Faerie Fire": 1, "Thunderwave": 1, "Dissonant Whispers": 1, "Charm Person": 1,
    "Tasha's Hideous Laughter": 1, "Chromatic Orb": 1, "Witch Bolt": 1, "Divine Favor": 1, "Shield of Faith": 1,
    "Thunderous Smite": 1, "Hunter's Mark": 1, "Ensnaring Strike": 1, "Disguise Self": 1,
    "Scorching Ray": 2, "Misty Step": 2, "Web": 2, "Spiritual Weapon": 2, "Lesser Restoration": 2, "Darkness": 2,
    "Hold Person": 2, "Moonbeam": 2, "Pass without Trace": 2, "Suggestion": 2, "Shatter": 2, "Mirror Image": 2,
    "Find Steed": 2, "Spike Growth": 2,
    "Fireball": 3, "Counterspell": 3, "Haste": 3, "Spirit Guardians": 3, "Revivify": 3, "Hunger of Hadar": 3,
    "Call Lightning": 3, "Hypnotic Pattern": 3, "Lightning Bolt": 3, "Aura of Vitality": 3, "Conjure Barrage": 3,
    "Mass Healing Word": 3,
    "Polymorph": 4, "Dimension Door": 4, "Banishment": 4, "Wall of Fire": 4,
    "Cone of Cold": 5,
}
ATTACK_SPELLS = {"Fire Bolt", "Ray of Frost", "Eldritch Blast", "Produce Flame", "Chromatic Orb", "Witch Bolt",
                 "Guiding Bolt", "Scorching Ray", "Spiritual Weapon"}
SAVE_SPELLS = {"Sacred Flame", "Toll the Dead", "Vicious Mockery", "Fireball", "Thunderwave", "Dissonant Whispers",
               "Shatter", "Lightning Bolt", "Hold Person", "Moonbeam", "Call Lightning", "Cone of Cold", "Entangle",
               "Faerie Fire", "Hypnotic Pattern", "Banishment", "Polymorph", "Tasha's Hideous Laughter", "Sleep",
               "Hunger of Hadar", "Wall of Fire", "Conjure Barrage", "Charm Person", "Suggestion"}
HEAL_SPELLS = {"Cure Wounds", "Healing Word", "Mass Healing Word", "Aura of Vitality"}
DAMAGE_TYPE = {"Fire Bolt": "fire", "Ray of Frost": "cold", "Eldritch Blast": "force", "Produce Flame": "fire",
               "Chromatic Orb": "acid", "Witch Bolt": "lightning", "Guiding Bolt": "radiant", "Scorching Ray": "fire",
               "Spiritual Weapon": "force", "Sacred Flame": "radiant", "Toll the Dead": "necrotic",
               "Vicious Mockery": "psychic", "Fireball": "fire", "Thunderwave": "thunder", "Shatter": "thunder",
               "Lightning Bolt": "lightning", "Moonbeam": "radiant", "Call Lightning": "lightning",
               "Cone of Cold": "cold", "Dissonant Whispers": "psychic", "Hunger of Hadar": "cold",
               "Wall of Fire": "fire", "Conjure Barrage": "piercing"}

SKILLS = ["Perception", "Stealth", "Investigation", "Insight", "Persuasion", "Athletics", "Deception",
          "Acrobatics", "History", "Arcana", "Survival", "Sleight of Hand", "Medicine", "Religion", "Intimidation",
          "Nature", "Animal Handling", "Performance"]
SKILL_W = [12, 7, 5, 1.8, 2, 2, 1.5, 1.3, 0.9, 0.9, 0.8, 0.6, 0.5, 0.5, 0.45, 0.4, 0.35, 0.25]
ABILITY_CHECKS = ["Strength", "Dexterity", "Intelligence", "Wisdom", "Constitution", "Charisma"]
SAVES = ["Constitution Save", "Wisdom Save", "Dexterity Save", "Strength Save", "Charisma Save",
         "Intelligence Save"]
SAVE_W = [6, 5, 4, 1, 0.4, 0.3]
TOOLS = ["Thieves' Tools"]
CONFUSABLE = {
    "Perception": ["Investigation", "Insight"], "Investigation": ["Perception"], "Insight": ["Perception"],
    "Attack": ["Spell Attack"], "Spell Attack": ["Attack"], "Stealth": ["Sleight of Hand"],
    "Persuasion": ["Deception"], "Deception": ["Persuasion"], "Athletics": ["Strength", "Acrobatics"],
    "Acrobatics": ["Athletics"], "Constitution Save": ["Constitution"], "Wisdom Save": ["Wisdom"],
    "Dexterity Save": ["Dexterity"], "Strength Save": ["Strength"], "Damage": ["Healing"], "Healing": ["Damage"],
    "Initiative": ["Dexterity"], "Arcana": ["History"], "History": ["Arcana"], "Nature": ["Survival"],
    "Survival": ["Nature"],
}

SPOKEN = {
    "Perception": ["perception", "a perception check", "perception check"],
    "Stealth": ["stealth", "a stealth check", "stealth check"],
    "Investigation": ["investigation", "an investigation check", "investigation check"],
    "Insight": ["insight", "an insight check", "insight check"],
    "Persuasion": ["persuasion", "a persuasion check", "persuasion check"],
    "Athletics": ["athletics", "an athletics check", "athletics check"],
    "Deception": ["deception", "a deception check", "deception check"],
    "Acrobatics": ["acrobatics", "an acrobatics check", "acrobatics check"],
    "History": ["history", "a history check", "history check"],
    "Arcana": ["arcana", "an arcana check", "arcana check"],
    "Survival": ["survival", "a survival check", "survival check"],
    "Sleight of Hand": ["sleight of hand", "a sleight of hand check", "sleight-of-hand check"],
    "Medicine": ["medicine", "a medicine check", "medicine check"],
    "Religion": ["religion", "a religion check", "religion check"],
    "Intimidation": ["intimidation", "an intimidation check", "intimidation check"],
    "Nature": ["nature", "a nature check", "nature check"],
    "Animal Handling": ["animal handling", "an animal handling check", "animal handling check"],
    "Performance": ["performance", "a performance check", "performance check"],
    "Strength": ["strength", "a strength check", "straight strength check"],
    "Dexterity": ["dexterity", "a dexterity check", "flat dexterity check"],
    "Intelligence": ["intelligence", "an intelligence check", "intelligence check"],
    "Wisdom": ["wisdom", "a wisdom check", "flat wisdom check"],
    "Constitution": ["constitution", "a constitution check", "constitution check"],
    "Charisma": ["charisma", "a charisma check", "charisma check"],
    "Constitution Save": ["a constitution saving throw", "a con save", "a constitution save"],
    "Wisdom Save": ["a wisdom saving throw", "a wisdom save", "a wis save"],
    "Dexterity Save": ["a dexterity saving throw", "a dex save", "a dexterity save"],
    "Strength Save": ["a strength saving throw", "a strength save", "a str save"],
    "Charisma Save": ["a charisma saving throw", "a charisma save", "a cha save"],
    "Intelligence Save": ["an intelligence saving throw", "an int save", "an intelligence save"],
    "Thieves' Tools": ["thieves' tools", "a thieves' tools check", "a lockpicking check with thieves' tools"],
    "Initiative": ["initiative"],
    "Death Save": ["a death saving throw", "a death save"],
}

NUM_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
             "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
             "twenty-one", "twenty-two", "twenty-three", "twenty-four", "twenty-five", "twenty-six", "twenty-seven",
             "twenty-eight", "twenty-nine", "thirty", "thirty-one", "thirty-two", "thirty-three", "thirty-four",
             "thirty-five", "thirty-six", "thirty-seven", "thirty-eight", "thirty-nine", "forty"]
ORDINAL = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh", 8: "eighth",
           9: "ninth"}

PLACES = ["the tavern", "the old mill", "the docks", "the market square", "the temple steps", "the sewer grate",
          "the bridge", "the watchtower", "the cellar", "the forest edge", "the caravan", "the inn's back room",
          "the library", "the crypt entrance", "the canyon rim", "the river bank", "the guard post", "the bazaar",
          "the shrine", "the mine shaft", "the lighthouse", "the barracks", "the courtyard", "the ruined chapel"]
NPCS = ["the barkeep", "a hooded stranger", "the captain", "an old fisherman", "the high priestess", "a young courier",
        "the merchant", "two guards", "the blacksmith", "a tiefling bard", "the harbormaster", "a goblin scout",
        "the innkeeper", "a dwarven miner", "the city magistrate", "a street urchin", "the cartographer"]
MONSTERS = ["the ghoul", "the bandit captain", "the ogre", "the giant spider", "the cultist", "the gnoll",
            "the wraith", "the owlbear", "the hobgoblin", "the troll", "the bugbear", "the zombie", "the harpy",
            "the wererat", "the animated armor", "the dire wolf", "the basilisk", "the hag"]
OBJECTS = ["a brass key", "a torn map", "a sealed letter", "a glowing gem", "a bloodstained dagger", "a music box",
           "a ledger", "a silver locket", "a cracked mirror", "a bundle of herbs", "an iron lockbox", "a bone flute",
           "a strange coin", "a wax seal", "a lantern", "a coil of rope", "a vial of ink", "a carved idol"]
ADJ = ["dim", "cold", "crowded", "quiet", "smoky", "damp", "narrow", "loud", "dusty", "flickering", "cramped",
       "drafty", "musty", "bright", "eerie", "rain-soaked", "gilded", "crumbling"]
SOUNDS = ["a distant bell", "footsteps above", "the creak of a door", "muffled voices", "a dog barking",
          "dripping water", "wind through the shutters", "a cart rattling past", "laughter from the street",
          "a low growl", "the crackle of the fire"]
DM_NARR = [
    "You make your way to {place}. The air is {adj} and you hear {sound}.",
    "As you step into {place}, {npc} looks up from a table and squints at you.",
    "{npc_cap} leans in close. \"I wouldn't go to {place} after dark, if I were you.\"",
    "The path bends toward {place}; somewhere nearby you hear {sound}.",
    "Inside, the room is {adj}. On a shelf you notice {obj}.",
    "{npc_cap} slides {obj} across the counter and waits for your answer.",
    "It takes you the better part of an hour to reach {place}. Nobody follows you, as far as you can tell.",
    "The {adj} corridor opens into a wider chamber, and {sound} echoes off the stone.",
    "{npc_cap} shrugs. \"Folks around {place} don't talk much. You'll want to ask {npc2}.\"",
    "You find {obj} wedged behind a loose board near {place}.",
    "Rain starts to fall as you cross toward {place}, and {sound} fades behind you.",
    "{npc_cap} counts out the coins slowly, then pockets {obj} without a word.",
    "Okay. So you head back toward {place}; the streets are {adj} this time of night.",
    "Glancing around {place}, you spot {npc} watching you from across the way.",
    "There's a long pause. {npc_cap} finally says, \"Fine. Meet me at {place} at dawn.\"",
]
PC_TALK = [
    "I want to ask {npc} about {obj}.", "Can we go to {place} first?", "I'm going to keep an eye on {npc}.",
    "Do I recognize {obj}?", "Let's not go to {place} tonight.", "I put {obj} in my pack.",
    "I walk over to {npc} and say hello.", "What does {place} look like from here?",
    "I'd like to buy {obj}, if they'll sell it.", "Is {npc} armed?", "I hand {obj} to {pc}.",
    "Wait, where is {pc} right now?", "I'm going to stay close to {pc}.", "Do we trust {npc}?",
    "I lean against the wall near {place} and listen.", "I ask {npc} what happened at {place}.",
    "Can I tell if {obj} is magical just by looking?", "I'm going to follow {npc} quietly.",
    "How far is {place} from here?", "I tuck {obj} under my cloak.",
]
CHATTER = [
    "Oh, that's good.", "Wait, really?", "I love this.", "That's so creepy.", "No, no, no.", "Oh my god.",
    "Okay, okay.", "Hold on.", "That's fair.", "Yeah, let's do it.", "I don't like this.", "Ooh.", "Oh no.",
    "Sure, why not.", "That's a lot.", "Good call.", "Wow.", "I'm scared.", "Classic.", "Nice.",
    "Are you serious?", "Same.", "Gross.", "That tracks.", "Bold move.", "I'm so confused.", "Love it.",
    "Not again.", "We should go.", "Is that bad?", "Okay, cool.", "Rude!", "Fine, fine.", "Amazing.",
]
DM_SHORT = ["Okay.", "Sure.", "Correct.", "All right.", "You can, yeah.", "Go for it.", "Mm-hmm.", "Fair enough.",
            "You got it.", "Yes.", "Not quite.", "Understood."]
OOC = [
    "Can someone pass the chips?", "I need more coffee.", "My character sheet is a mess, hold on.",
    "Who's got the pencil?", "Did everyone see the fan art this week?", "My cat walked across my keyboard earlier.",
    "Is it hot in here or is it just me?", "Hydrate, everybody.", "We're going to be here all night, aren't we?",
    "I forgot to eat lunch.", "Did someone bring the snacks?", "This table is so pretty.",
    "Shout-out to everyone in the chat.", "I'm writing that down.", "That's going on a T-shirt.",
]
TAILS = ["", "", "", "", " Easy.", " Hopefully.", " Come on.", " Ugh.", " Nice!", " Not great.", " Yes!",
         " Okay.", " That's fine.", " Is that good?", " Boom.", " Meh.", " Ha!", " Woo!", " Damn.", " Please?"]
LEADS = ["", "", "", "", "Uh, ", "Okay, ", "Um, ", "So, ", "Oh, ", "Ooh, ", "Wait, ", "Alright, ", "Hmm, "]

SPELL_ALIASES = {
    "Eldritch Blast": ["EB", "eldritch blast", "Eldritch Blast"], "Fire Bolt": ["firebolt", "fire bolt", "a Fire Bolt"],
    "Healing Word": ["healing word", "a Healing Word"], "Cure Wounds": ["cure wounds", "Cure Wounds"],
    "Hunter's Mark": ["hunter's mark", "Hunter's Mark"], "Spiritual Weapon": ["spiritual weapon", "my Spiritual Weapon"],
    "Magic Missile": ["magic missile", "Magic Missile"], "Sacred Flame": ["sacred flame", "Sacred Flame"],
    "Toll the Dead": ["toll the dead", "Toll the Dead"], "Vicious Mockery": ["vicious mockery", "a Vicious Mockery"],
    "Guiding Bolt": ["guiding bolt", "a Guiding Bolt"], "Scorching Ray": ["scorching ray", "Scorching Ray"],
    "Misty Step": ["misty step", "Misty Step"], "Hold Person": ["hold person", "Hold Person"],
    "Dissonant Whispers": ["dissonant whispers"], "Tasha's Hideous Laughter": ["Tasha's", "hideous laughter"],
    "Armor of Agathys": ["armor of agathys", "Armor of Agathys"], "Pass without Trace": ["pass without trace"],
    "Detect Magic": ["detect magic", "Detect Magic"], "Mage Hand": ["mage hand", "a Mage Hand"],
    "Ray of Frost": ["ray of frost", "Ray of Frost"], "Chromatic Orb": ["chromatic orb", "a Chromatic Orb"],
}
VERB_SPELLS = {"Hex": "Hex", "Bless": "Bless", "Shield": "Shield", "Counterspell": "Counterspell",
               "Banishment": "banish", "Polymorph": "polymorph", "Haste": "Haste", "Misty Step": "Misty Step",
               "Healing Word": "Healing Word", "Hunter's Mark": "Hunter's Mark", "Charm Person": "charm",
               "Guiding Bolt": "Guiding Bolt", "Entangle": "entangle", "Faerie Fire": "Faerie Fire",
               "Thunderwave": "Thunderwave", "Shatter": "Shatter", "Fireball": "Fireball", "Darkness": "Darkness",
               "Sleep": "Sleep", "Web": "Web", "Moonbeam": "Moonbeam", "Suggestion": "Suggestion"}
LINGERING = {"Spiritual Weapon", "Moonbeam", "Call Lightning", "Witch Bolt", "Hex", "Hunter's Mark", "Spirit Guardians",
             "Bless", "Haste", "Armor of Agathys", "Faerie Fire", "Hunger of Hadar", "Wall of Fire", "Shillelagh"}

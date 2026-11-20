"""Real-world vocabulary the synthetic encyclopedia is written in."""

from __future__ import annotations

PROFESSIONS: dict[str, tuple[list[str], list[str]]] = {
    "a woodwind player": (["oboist", "bassoonist", "clarinettist", "flautist"],
                          ["played second reed in a touring orchestra"]),
    "a brass player": (["trumpeter", "trombonist", "cornet player", "tuba player"],
                       ["played in the brass section of a regimental band"]),
    "a keyboard player": (["organist", "harpsichordist", "pianist"],
                          ["gave recitals at the keyboard in concert halls"]),
    "a string player": (["violinist", "cellist", "violist", "harpist"],
                        ["played in the string section of the civic orchestra"]),
    "a maker of musical instruments": (["luthier", "organ builder", "piano maker"],
                                       ["built and repaired instruments in a small workshop"]),
    "a medical doctor": (["surgeon", "physician", "cardiologist", "paediatrician", "anaesthetist"],
                         ["practised medicine at the county infirmary", "treated patients as a qualified doctor"]),
    "a nurse or pharmacist": (["nurse", "midwife", "pharmacist", "apothecary"],
                              ["dispensed remedies and cared for the sick"]),
    "a metalworker": (["blacksmith", "goldsmith", "silversmith", "tinsmith"],
                      ["worked metal at a forge"]),
    "a natural scientist": (["botanist", "geologist", "entomologist", "ornithologist", "mycologist"],
                            ["collected and classified specimens for a natural history museum"]),
    "a legal professional": (["barrister", "solicitor", "magistrate", "notary"],
                             ["practised law in the district courts"]),
    "a seafarer": (["sea captain", "navigator", "boatswain", "harbour pilot"],
                   ["spent decades at sea on merchant vessels"]),
    "a member of the clergy": (["vicar", "chaplain", "rabbi", "parish priest"],
                               ["served a congregation for most of a lifetime"]),
    "a food tradesperson": (["baker", "butcher", "brewer", "cheesemaker", "confectioner"],
                            ["kept a shop selling bread and pastries", "ran a family business making food"]),
    "a printing tradesperson": (["printer", "typesetter", "bookbinder", "lithographer"],
                                ["worked the presses at a printing house"]),
    "a textile worker": (["weaver", "draper", "tailor", "milliner", "dyer"],
                         ["worked in the cloth trade"]),
    "a dealer or financier": (["banker", "auctioneer", "pawnbroker", "stockbroker", "money changer"],
                              ["sold estates and furniture at public sales", "lent money against goods"]),
    "a builder": (["stonemason", "carpenter", "bricklayer", "roofer"],
                  ["raised houses and barns across the district"]),
    "a visual artist": (["painter", "sculptor", "engraver", "watercolourist", "illustrator"],
                        ["made pictures for a living"]),
    "a writer": (["novelist", "poet", "playwright", "essayist"],
                 ["made a living from the pen"]),
    "a teacher": (["schoolmaster", "governess", "lecturer", "tutor"],
                  ["taught children at the village school"]),
    "a farmer": (["shepherd", "dairy farmer", "orchardist", "beekeeper"],
                 ["worked the land on a family holding"]),
    "a person who worked with horses": (["farrier", "jockey", "horse trainer", "coachman"],
                                        ["looked after horses at a coaching inn"]),
    "a surveyor or mapmaker": (["cartographer", "surveyor", "hydrographer"],
                               ["drew maps of the coast for the admiralty"]),
    "a maker of timepieces": (["clockmaker", "watchmaker"],
                              ["repaired clocks and watches"]),
    "a composer": (["composer"], ["wrote music for the stage and the concert hall"]),
    "an architect": (["architect"], ["designed public buildings"]),
}

TERM_CATEGORY: dict[str, str] = {t: c for c, (ts, _) in PROFESSIONS.items() for t in ts}

CREATIVE = {
    "a writer": ["novel", "novel", "poetry collection", "children's book", "play"],
    "a visual artist": ["painting", "painting", "etching"],
    "a composer": ["symphony", "opera", "song cycle"],
}

THEMES: dict[str, list[str]] = {
    "a bird of prey": ["falcon", "kestrel", "osprey", "buzzard", "sparrowhawk"],
    "a sea creature": ["octopus", "grey seal", "humpback whale", "lobster", "jellyfish"],
    "a flowering plant": ["orchid", "tulip", "foxglove", "magnolia", "poppy"],
    "a precious stone": ["sapphire", "emerald", "ruby", "opal", "garnet"],
    "a celestial event": ["comet", "solar eclipse", "meteor shower", "aurora"],
    "a coastal structure": ["lighthouse", "harbour wall", "sea fort", "pier"],
    "a sailing vessel": ["schooner", "frigate", "brigantine", "clipper"],
    "a wild cat": ["lynx", "leopard", "ocelot", "jaguar"],
    "a kind of tree": ["willow", "yew", "birch", "chestnut tree"],
    "a board game": ["chess", "backgammon", "draughts", "mahjong"],
    "a kind of storm": ["blizzard", "hailstorm", "thunderstorm", "sandstorm"],
    "a musical instrument": ["harpsichord", "bagpipes", "hurdy-gurdy", "zither"],
}
THEME_CATEGORY: dict[str, str] = {t: c for c, ts in THEMES.items() for t in ts}

ORG_KINDS = {
    "gallery": ["{x} Gallery", "the {x} Academy of Fine Arts", "the {x} Art Society"],
    "press": ["{x} Press", "{x} and Sons Publishers", "the {x} Publishing House"],
    "school": ["{x} Grammar School", "{x} College", "the {x} Academy for Girls", "{x} School of Music"],
    "society": ["the {x} Society of Naturalists", "the {x} Philharmonic Society", "the {x} Literary Society",
                "the {x} Rowing Club"],
    "prize": ["the {x} Medal", "the {x} Prize", "the {x} Award for Fiction"],
}

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
NUMWORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
            "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
            "twenty-one", "twenty-two", "twenty-three", "twenty-four", "twenty-five", "twenty-six", "twenty-seven",
            "twenty-eight", "twenty-nine", "thirty"]
ORDINALS = ["zeroth", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]

FIRST_F = ["Ada", "Agnes", "Alma", "Amelie", "Anouk", "Astrid", "Beatrix", "Bettina", "Brigid", "Camille", "Cecily",
           "Clara", "Corin", "Dagny", "Delphine", "Edda", "Elin", "Elise", "Elowen", "Esme", "Eulalie", "Fenna",
           "Freya", "Greta", "Hedda", "Helka", "Ida", "Ilse", "Imke", "Ines", "Isolde", "Jutta", "Karin", "Kirsi",
           "Lene", "Liesel", "Linnea", "Lorna", "Maud", "Mareike", "Marit", "Mina", "Nell", "Nora", "Odile", "Ottilie",
           "Petra", "Rhoswen", "Rosalind", "Saskia", "Selma", "Sigrid", "Solveig", "Tamsin", "Thea", "Tilda", "Una",
           "Vera", "Wanda", "Wilhelmina", "Ysolde", "Zelda", "Aurelie", "Berit", "Carys", "Dorothea", "Eira",
           "Frieda", "Gunnhild", "Henrike", "Ingrid", "Jolanda", "Ketil", "Leocadia", "Malin", "Nanna", "Orla"]
FIRST_M = ["Aldo", "Anselm", "Arvid", "Bastian", "Benedikt", "Bertil", "Casimir", "Cyprian", "Dag", "Desmond",
           "Edvard", "Egon", "Emrys", "Evander", "Fabian", "Falk", "Florian", "Gideon", "Gunnar", "Hagen", "Hartwig",
           "Ignaz", "Ivo", "Jannik", "Jasper", "Joris", "Kasimir", "Konrad", "Lambert", "Leopold", "Linus", "Lorcan",
           "Magnus", "Matthias", "Mikkel", "Niall", "Nikolai", "Odo", "Osric", "Otto", "Piet", "Quentin", "Rafael",
           "Roald", "Rupert", "Severin", "Sigurd", "Silas", "Stellan", "Tadeo", "Teodor", "Thorvald", "Tobias",
           "Ulrich", "Valentin", "Vidar", "Wendel", "Wolfram", "Xaver", "Yannick", "Zeno", "Alaric", "Bram",
           "Caius", "Dorian", "Eamon", "Fridolin", "Gervase", "Hallvard", "Ingmar", "Jarle", "Kjell", "Leander"]

SYL_A = ["al", "an", "ar", "bel", "bor", "bran", "cal", "cor", "dal", "dun", "el", "fal", "fen", "gar", "hal", "hol",
         "ir", "jan", "kel", "kes", "lan", "lor", "mar", "mel", "mor", "nar", "nor", "ost", "pel", "quen", "rad",
         "ren", "sal", "sel", "tal", "tor", "ul", "val", "vel", "wen", "yar", "zel", "brin", "drav", "gwen", "thal",
         "vor", "isk", "esk", "ard"]
SYL_B = ["a", "e", "i", "o", "u", "ia", "ae", "ou", "ei", "y"]
SYL_C = ["bek", "by", "dale", "den", "dorf", "fell", "ford", "gard", "ham", "holm", "ick", "ley", "lin", "mar", "mont",
         "more", "ner", "nik", "rath", "ric", "ridge", "rook", "sen", "stad", "ston", "thorpe", "ton", "vik", "wald",
         "well", "wick", "worth", "zar", "mund", "lett", "rand", "kov", "vane", "mere", "brook"]

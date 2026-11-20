"""Flag traces that contain a sentence the pool decontamination dropped.

The sentence list ($CTC_DATAGEN_DIR/xabsence_paraphrase/dropped_sentences.json) holds eval text and is not shipped."""
import json, os, re
HERE = os.path.dirname(os.path.abspath(__file__))
DROP_FILE = os.path.join(os.environ.get("CTC_DATAGEN_DIR") or os.path.join(HERE, "../../../outputs/synth-datagen/ctcbench"),
                         "xabsence_paraphrase", "dropped_sentences.json")
DROP = None
norm = lambda s: re.sub(r"[^a-z0-9]+", " ", s.lower())
def contaminated(obj) -> bool:
    global DROP
    if DROP is None:
        DROP = json.load(open(DROP_FILE))
    blob = norm(json.dumps(obj, ensure_ascii=False).replace("\\n", " "))
    return any(s in blob for s in DROP)

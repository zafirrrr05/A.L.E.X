"""
Label configuration for the Action Detection Module.

Maps your clip_dataset classes to the paper's event taxonomy.
Background (index 0) is reserved — no clip should be labelled 0 at training
time; it is emitted at inference when no foreground event is detected.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Your clip_dataset folder names  →  canonical event name  →  class index
# ──────────────────────────────────────────────────────────────────────────────
#
# Mapping rationale (paper Table / Section 3.1 classes vs your labels):
#   corner      → corner          (dead-ball restart from corner arc)
#   foul        → tackle          (closest paper equivalent; foul arises from tackle/challenge)
#   freekick    → free-kick       (dead-ball restart)
#   goal        → shot            (a goal IS a successful shot — paper uses "shot" as the action)
#   goalkick    → goal-kick       (dead-ball restart)
#   longpass    → pass            (long variant of pass; paper has unified "pass" class)
#   ontarget    → shot            (shot on target — same action class as goal above)
#   penalty     → penalty         (specific shot subtype, kept separate)
#   shortpass   → pass            (short variant; merged with longpass into "pass")
#   substitution → substitution   (non-ball event, kept for completeness)
#   throw-in    → throw-in        (paper has this explicitly)
#
# NOTE: goal and ontarget both map to CLASS_SHOT.  If you later want to
#       distinguish them, simply assign them separate indices here.

BACKGROUND   = 0

# Foreground classes (1-indexed to leave 0 for background)
CLASS_PASS         = 1   # shortpass + longpass
CLASS_SHOT         = 2   # goal + ontarget
CLASS_CORNER       = 3
CLASS_FREEKICK     = 4
CLASS_GOALKICK     = 5
CLASS_PENALTY      = 6
CLASS_SUBSTITUTION = 7
CLASS_THROW_IN     = 8
CLASS_TACKLE       = 9   # foul

NUM_CLASSES = 10         # 0..9  (background + 9 foreground)

# Human-readable labels (index → string)
IDX_TO_LABEL = {
    BACKGROUND:         "background",
    CLASS_PASS:         "pass",
    CLASS_SHOT:         "shot",
    CLASS_CORNER:       "corner",
    CLASS_FREEKICK:     "free-kick",
    CLASS_GOALKICK:     "goal-kick",
    CLASS_PENALTY:      "penalty",
    CLASS_SUBSTITUTION: "substitution",
    CLASS_THROW_IN:     "throw-in",
    CLASS_TACKLE:       "tackle",
}

LABEL_TO_IDX = {v: k for k, v in IDX_TO_LABEL.items()}

# Maps raw folder name  →  class index
FOLDER_TO_CLASS = {
    "corner":       CLASS_CORNER,
    "foul":         CLASS_TACKLE,
    "freekick":     CLASS_FREEKICK,
    "goal":         CLASS_SHOT,
    "goalkick":     CLASS_GOALKICK,
    "longpass":     CLASS_PASS,
    "ontarget":     CLASS_SHOT,
    "penalty":      CLASS_PENALTY,
    "shortpass":    CLASS_PASS,
    "substitution": CLASS_SUBSTITUTION,
    "throw-in":     CLASS_THROW_IN,
}

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline API-2 event type strings  (Part 5 §5.3 / §5.10)
# Used in the JSON output emitted to the Formation module.
# ──────────────────────────────────────────────────────────────────────────────
API2_EVENT_TYPE = {
    BACKGROUND:         "background",
    CLASS_PASS:         "pass",
    CLASS_SHOT:         "shot",
    CLASS_CORNER:       "corner",
    CLASS_FREEKICK:     "free-kick",
    CLASS_GOALKICK:     "goal-kick",
    CLASS_PENALTY:      "penalty",
    CLASS_SUBSTITUTION: "substitution",
    CLASS_THROW_IN:     "throw-in",
    CLASS_TACKLE:       "tackle",
}

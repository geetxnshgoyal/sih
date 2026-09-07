"""Shared deterministic vocabulary priority for resumable extraction."""
import re

CLINICAL = """
please help water pain body injury hurt wound blood bandage injection medicine
tablet doctor nurse hospital emergency ambulance operation patient sick fever
headache stomach dizzy weak breathe heart head hand eye ear back throat tooth
bone skin yes no stop wait more less sorry thank you
""".split()

DAILY = """
hello how are you good morning good afternoon good evening good night name time
today tomorrow yesterday morning evening night day week month year food eat drink
hungry thirsty sleep bathroom understand know give take come sit stand walk fall
mother father sister brother family child baby man woman friend house school work
money phone car bus train ticket station airport hotel police road right left up
down open close hot cold good bad big small happy sad afraid deaf hearing
""".split()

def normalise(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", value.lower())).strip()

_CLINICAL = set(CLINICAL)
_DAILY = set(DAILY)

def priority_key(label: str):
    word = normalise(label)
    tokens = set(word.split())
    if word in _CLINICAL or tokens & _CLINICAL:
        tier = 0
    elif word in _DAILY or tokens & _DAILY:
        tier = 1
    else:
        tier = 2
    return tier, word

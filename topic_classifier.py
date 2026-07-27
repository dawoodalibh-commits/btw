"""Phase 9 — Topic Classification.

Rule-based: score each candidate topic by how many of its keywords appear in
the question text, keep anything that scores above zero. This is
deliberately swappable for an LLM classifier later (same input/output
contract: question text in, topic name list out) once the keyword lists stop
being good enough for messier real-world questions.

Usage:
    python topic_classifier.py --built output/built --output-dir output/topics
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from schemas import read_json, write_json

# Public, well-known Cambridge International subject codes -- used only to
# narrow keyword search to the right subject and avoid cross-subject false
# positives (e.g. "equilibrium" means something different in physics vs
# chemistry). Unknown codes fall back to searching every subject.
_SUBJECT_CODES = {"9702": "Physics", "9709": "Mathematics", "9701": "Chemistry"}

_TOPIC_KEYWORDS: dict[str, dict[str, list[str]]] = {
    "Physics": {
        "Measurement": ["uncertaint", "precision", "accuracy", "significant figure", "estimate", "scalar", "vector quantity", "si unit", "base quantit"],
        "Mechanics": ["velocity", "acceleration", "decelerat", "displacement", "projectile", "friction", "newton's", "equilibrium of forces", "resultant force"],
        "Moments": ["moment of a force", "torque", "couple", "centre of gravity", "principle of moments"],
        "Momentum": ["momentum", "collision", "impulse", "conservation of momentum"],
        "Circular Motion": ["circular motion", "centripetal", "angular velocity"],
        "Gravitational Fields": ["gravitational field", "gravitational potential", "orbit", "satellite", "kepler"],
        "Waves": ["wave", "wavelength", "amplitude", "diffraction", "interference", "refraction", "reflection", "doppler", "stationary wave", "polaris"],
        "Electricity": ["current", "voltage", "resistance", "resistor", "circuit", "potential difference", "capacitor", "e.m.f", "emf", "ohm"],
        "Thermal Physics": ["thermal", "specific heat capacity", "internal energy", "ideal gas", "kelvin", "kinetic theory"],
        "Nuclear Physics": ["nucleus", "nuclear", "radioactive", "decay", "half-life", "alpha particle", "beta particle", "gamma", "isotope", "fission", "fusion", "binding energy"],
        "Particle Physics": ["quark", "lepton", "boson", "neutrino", "standard model"],
        "Quantum Physics": ["photon", "photoelectric", "wave-particle", "de broglie", "energy level", "quantum"],
        "Magnetic Fields": ["magnetic field", "magnetic flux", "electromagnetic induction", "solenoid", "flemin"],
    },
    "Mathematics": {
        "Algebra": ["quadratic", "polynomial", "factoris", "simultaneous equation", "inequalit"],
        "Trigonometry": ["sine", "cosine", "tangent", "trigonometric", "radian"],
        "Differentiation": ["differentiat", "derivative", "stationary point", "gradient of the curve", "rate of change"],
        "Integration": ["integrat", "area under the curve", "definite integral"],
        "Vectors": ["vector", "magnitude of", "unit vector", "position vector", "dot product"],
        "Statistics": ["mean", "median", "variance", "standard deviation", "probability", "distribution", "binomial", "normal distribution"],
        "Mechanics": ["velocity", "acceleration", "force", "projectile", "momentum"],
        "Series": ["arithmetic progression", "geometric progression", "series", "sum to infinity"],
    },
    "Chemistry": {
        "Atomic Structure": ["proton", "neutron", "electron", "atomic number", "mass number", "isotope", "electron configuration"],
        "Bonding": ["ionic bond", "covalent bond", "metallic bond", "intermolecular", "hydrogen bond", "van der waals"],
        "Organic Chemistry": ["alkane", "alkene", "alcohol", "carboxylic acid", "ester", "functional group", "isomer", "polymer", "hydrocarbon"],
        "Physical Chemistry": ["enthalpy", "entropy", "equilibrium constant", "rate of reaction", "activation energy", "le chatelier"],
        "Inorganic Chemistry": ["periodic table", "group 2", "transition element", "halogen"],
        "Electrochemistry": ["redox", "oxidation", "reduction", "electrode", "electrolysis", "half-equation"],
    },
}


def _subject_for_paper(paper: str) -> str | None:
    code = paper.split("/")[0] if paper else ""
    return _SUBJECT_CODES.get(code)


def classify_text(text: str, subject: str | None = None) -> list[str]:
    text_lower = text.lower()
    subjects = [subject] if subject in _TOPIC_KEYWORDS else _TOPIC_KEYWORDS.keys()

    scores: dict[str, int] = {}
    for subj in subjects:
        for topic, keywords in _TOPIC_KEYWORDS[subj].items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score:
                scores[topic] = scores.get(topic, 0) + score

    return [topic for topic, _ in sorted(scores.items(), key=lambda kv: -kv[1])]


def classify_questions(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    classified = []
    for q in questions:
        subject = _subject_for_paper(q.get("paper", ""))
        classified.append({**q, "topics": classify_text(q["text"], subject)})
    return classified


def run(built_dir: Path, output_dir: Path) -> list[dict[str, Any]]:
    questions = read_json(built_dir / "built_questions.json")
    classified = classify_questions(questions)
    write_json(classified, output_dir / "classified_questions.json")
    return classified


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 9: rule-based topic tagging for built questions.")
    parser.add_argument("--built", type=Path, default=Path("output/built"), help="Phase 8 output directory")
    parser.add_argument("--output-dir", type=Path, default=Path("output/topics"))
    args = parser.parse_args()

    classified = run(args.built, args.output_dir)
    n_tagged = sum(1 for q in classified if q["topics"])
    print(f"Classified {len(classified)} questions ({n_tagged} got at least one topic) -> {args.output_dir}")


if __name__ == "__main__":
    main()

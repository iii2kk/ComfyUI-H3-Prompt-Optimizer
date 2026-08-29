import copy
import re


SECTION_NAMES = (
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
)
REFERENCE_FIELDS = (
    "ref_images",
    "ref_videos",
    "ref_video_audios",
    "ref_audios",
)

_SHOT_RE = re.compile(r"\[Shot\s+(\d+)](?:\s+At\s+(\d{2}):(\d{2})\.(\d{3}),)?")
_TAG_RE = re.compile(r"<(Subject|Picture|Video|Audio)\s+(\d+)>")


def normalize_prompt(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("MiniMax H3 Ref2VA prompt is empty.")
    value = text.strip().lstrip("\ufeff").strip().replace("\r\n", "\n").replace("\r", "\n")
    if "```" in value:
        raise ValueError("MiniMax H3 Ref2VA prompt contains a Markdown code fence.")
    return value


def _split_sections(text):
    matches = list(re.finditer(r"(?m)^(%s):" % "|".join(map(re.escape, SECTION_NAMES)), text))
    found = [match.group(1) for match in matches]
    if found != list(SECTION_NAMES):
        raise ValueError("MiniMax H3 Ref2VA prompt sections are missing or out of order.")

    sections = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        value = text[match.end():end].strip()
        if not value:
            raise ValueError("MiniMax H3 Ref2VA prompt section '%s' is empty." % match.group(1))
        sections[match.group(1)] = value
    return sections


def _parse_shots(description, duration_seconds):
    matches = list(_SHOT_RE.finditer(description))
    if not matches or int(matches[0].group(1)) != 1:
        raise ValueError("MiniMax H3 Ref2VA prompt must begin its shot sequence with [Shot 1].")

    shots = []
    for index, match in enumerate(matches):
        number = int(match.group(1))
        if number != index + 1:
            raise ValueError("MiniMax H3 Ref2VA shot numbers must be sequential.")
        if index == 0:
            if match.group(2) is not None:
                raise ValueError("[Shot 1] must not have a timestamp.")
            start_sec = 0.0
        else:
            if match.group(2) is None:
                raise ValueError("[Shot %d] must begin with an MM:SS.mmm timestamp." % number)
            start_sec = int(match.group(2)) * 60 + int(match.group(3)) + int(match.group(4)) / 1000
            if start_sec <= shots[-1]["start_sec"] or start_sec >= duration_seconds:
                raise ValueError("MiniMax H3 Ref2VA shot timestamps must increase within the target duration.")

        end = matches[index + 1].start() if index + 1 < len(matches) else len(description)
        body = description[match.end():end].strip()
        if not body:
            raise ValueError("MiniMax H3 Ref2VA [Shot %d] is empty." % number)
        shots.append({
            "id": "shot_%d" % number,
            "number": number,
            "start_sec": start_sec,
            "heading": match.group(0),
            "body": body,
        })

    return description[:matches[0].start()].strip(), shots


def expected_reference_labels(package):
    if not isinstance(package, dict) or package.get("schema_version") != 1:
        raise ValueError("Invalid MiniMax H3 Ref2VA package.")
    if any(not isinstance(package.get(name), dict) for name in REFERENCE_FIELDS):
        raise ValueError("Invalid MiniMax H3 Ref2VA package.")

    labels = {
        *(('Picture', index) for index in range(1, len(package["ref_images"]) + 1)),
        *(('Video', index) for index in range(1, len(package["ref_videos"]) + 1)),
    }
    audio_count = len(package["ref_video_audios"]) + len(package["ref_audios"])
    labels.update(('Audio', index) for index in range(1, audio_count + 1))
    return labels


def defined_reference_labels(document):
    definitions = document["sections"]["subject_definitions"]
    return {
        (kind, int(number))
        for kind, number in _TAG_RE.findall(definitions)
        if kind != "Subject"
    }


def _validate_labels(sections, expected_labels):
    definitions = {(kind, int(number)) for kind, number in _TAG_RE.findall(sections["subject_definitions"])}
    missing = expected_labels - definitions
    if missing:
        labels = ", ".join("<%s %d>" % item for item in sorted(missing))
        raise ValueError("MiniMax H3 Ref2VA prompt is missing supplied reference labels: %s." % labels)

    for kind in ("Subject", "Picture", "Video", "Audio"):
        numbers = sorted({number for label_kind, number in definitions if label_kind == kind})
        if numbers and numbers != list(range(1, max(numbers) + 1)):
            raise ValueError("MiniMax H3 Ref2VA %s labels must be sequential from 1." % kind)

    later = "\n".join(sections[name] for name in SECTION_NAMES[1:])
    used = {(kind, int(number)) for kind, number in _TAG_RE.findall(later)}
    undefined = used - definitions
    if undefined:
        labels = ", ".join("<%s %d>" % item for item in sorted(undefined))
        raise ValueError("MiniMax H3 Ref2VA prompt uses undefined labels: %s." % labels)


def parse_ref2va_prompt(text, duration_seconds, expected_labels=None):
    duration_seconds = float(duration_seconds)
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive.")
    value = normalize_prompt(text)
    if not value.startswith("subject_definitions:"):
        raise ValueError("MiniMax H3 Ref2VA prompt must begin with subject_definitions:.")
    sections = _split_sections(value)
    preamble, shots = _parse_shots(sections["detailed_description"], duration_seconds)
    _validate_labels(sections, expected_labels or set())
    return {
        "schema_version": 1,
        "mode": "ref2va",
        "duration_seconds": duration_seconds,
        "sections": {
            "subject_definitions": sections["subject_definitions"],
            "summary": sections["summary"],
            "retention_analysis": sections["retention_analysis"],
            "detailed_description": {
                "preamble": preamble,
                "shots": shots,
            },
            "overall_soundscape": sections["overall_soundscape"],
            "non_diegetic_music": sections["non_diegetic_music"],
        },
    }


def compile_ref2va_prompt(document):
    sections = document["sections"]
    detailed = sections["detailed_description"]
    shot_text = []
    if detailed["preamble"]:
        shot_text.append(detailed["preamble"])
    shot_text.extend("%s %s" % (shot["heading"], shot["body"]) for shot in detailed["shots"])
    values = {
        "subject_definitions": sections["subject_definitions"],
        "summary": sections["summary"],
        "retention_analysis": sections["retention_analysis"],
        "detailed_description": "\n\n".join(shot_text),
        "overall_soundscape": sections["overall_soundscape"],
        "non_diegetic_music": sections["non_diegetic_music"],
    }
    return "\n\n".join("%s:\n%s" % (name, values[name]) for name in SECTION_NAMES)


def shot_index_for_time(document, time_sec):
    shots = document["sections"]["detailed_description"]["shots"]
    selected = 0
    for index, shot in enumerate(shots):
        if shot["start_sec"] <= time_sec:
            selected = index
        else:
            break
    return selected


def allowed_patch_paths(document, issue_type, localization=None):
    if issue_type in ("audio_sync", "dialogue"):
        return {
            "/sections/overall_soundscape",
            "/sections/non_diegetic_music",
        }
    midpoint = 0.0
    if localization:
        midpoint = (float(localization.get("start_sec", 0.0)) + float(localization.get("end_sec", 0.0))) / 2
    index = shot_index_for_time(document, midpoint)
    paths = {"/sections/detailed_description/shots/%d/body" % index}
    if issue_type in ("appearance", "identity_consistency", "object_consistency"):
        paths.update({
            "/sections/subject_definitions",
            "/sections/retention_analysis",
        })
    return paths


def apply_patch(document, operations, allowed_paths):
    updated = copy.deepcopy(document)
    for operation in operations:
        if operation.op != "replace" or operation.path not in allowed_paths:
            raise ValueError("Prompt patch attempted to change a locked path: %s." % operation.path)
        if operation.path == "/sections/subject_definitions":
            updated["sections"]["subject_definitions"] = operation.value.strip()
        elif operation.path == "/sections/retention_analysis":
            updated["sections"]["retention_analysis"] = operation.value.strip()
        elif operation.path == "/sections/overall_soundscape":
            updated["sections"]["overall_soundscape"] = operation.value.strip()
        elif operation.path == "/sections/non_diegetic_music":
            updated["sections"]["non_diegetic_music"] = operation.value.strip()
        else:
            match = re.fullmatch(r"/sections/detailed_description/shots/(\d+)/body", operation.path)
            if not match:
                raise ValueError("Unsupported Prompt IR patch path: %s." % operation.path)
            index = int(match.group(1))
            shots = updated["sections"]["detailed_description"]["shots"]
            if index >= len(shots):
                raise ValueError("Prompt IR patch references a missing shot.")
            shots[index]["body"] = operation.value.strip()
        if not operation.value.strip():
            raise ValueError("Prompt IR patch values must not be empty.")
    return updated

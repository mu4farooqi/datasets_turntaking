import re
from enum import Enum
from os.path import join

from datasets_turntaking.utils import read_txt

BACKCHANNEL_PHRASES = {
    "yeah", "um-hum", "uh-huh", "right", "oh", "um", "uh", "okay", "yes",
    "so", "oh yeah", "yeah yeah", "oh really", "um-hum um-hum", "and uh",
    "that's right", "oh okay", "uh-huh uh-huh", "right right", "but uh",
    "oh uh-huh", "i see", "oh wow", "oh no", "right yeah", "but um", "and um"
}

class Labels(Enum):
    TURN_TAKING = 0
    CONTINUING_SPEECH = 1
    BACKCHANNEL = 2


OmitText = [
    "[silence]",
    "[noise]",
    "[vocalized-noise]",
]

# Only used once, json file is included in repo
def extract_audio_mapping(audio_root):
    """
    Used to create `relative_audio_path.json`.

    The audio files requires manual download and the extracted files are
    organized in a "non-straight-forward" manner. This function maps the
    session ids to their relative audio_path.

    This should only be run once and the mapping should be "shipped" with the
    dataset. For now it is included in the git-repo but should be downloaded
    along with the transcripts.

    Then given the path to the audio-root (extracted audio files) the audio
    path is reconstructed.

    (I have changed the format from .sph -> .wav)
    ```
      AUDIO_ROOT
    ├──  swb1_d1
    │  └──  data
    |       |-- sw02285.{wav,sph}
    |       └-- ..
    ├──  swb1_d2
    │  └──  data
    |       └-- ...
    ├──  swb1_d3
    │  └──  data
    |       └-- ...
    └──  swb1_d4
       └──  data
            └-- ...
    ```

    ```python
    # Construct relative-audio-path-mappings:
    audio_root = "/Path/To/Extracted/Audio"
    map = extract_audio_mapping(audio_root)
    write_json(map, "swb_session_to_audio_map.json")
    ```

    RETURN:
        map:    dict, i.e. map['3002'] -> 'swb1_d1/data/sw03002'
    """
    map = {}
    for root, _, files in walk(audio_root):
        if len(files) > 0:
            rel_path = root.replace(audio_root, "")
            for f in files:
                if f.endswith(".wav") or f.endswith(".sph"):
                    # sw03815.{wav,sph} -> 3815
                    session = basename(f)
                    session = re.sub(r"^sw0(\w*).*", r"\1", session)
                    # sw03815.{wav,sph} ->sw03815
                    f_no_ext = re.sub(r"^(.*)\.\w*", r"\1", f)
                    map[session] = join(rel_path, f_no_ext)
    return map


def swb_regexp(s):
    """
    Switchboard annotation specific regexp.

    See:
        - `datasets_turntaking/features/dataset/switchboard.md`
        - https://www.isip.piconepress.com/projects/switchboard/doc/transcription_guidelines/transcription_guidelines.pdf

    """
    # Noise
    s = re.sub(r"\[noise\]", "", s)
    s = re.sub(r"\[vocalized-noise\]", "", s)

    # laughter
    s = re.sub(r"\[laughter\]", "", s)
    # laughing and speech e.g. [laughter-yeah] -> yeah
    s = re.sub(r"\[laughter-(\w*)\]", r"\1", s)
    s = re.sub(r"\[laughter-(\w*\'*\w*)\]", r"\1", s)

    # Partial words: w[ent] -> went
    s = re.sub(r"(\w+)\[(\w*\'*\w*)\]", r"\1\2", s)
    # Partial words: -[th]at -> that
    s = re.sub(r"-\[(\w*\'*\w*)\](\w+)", r"\1\2", s)

    # restarts
    s = re.sub(r"(\w+)-\s", r"\1 ", s)
    s = re.sub(r"(\w+)-$", r"\1", s)

    # Pronounciation variants
    s = re.sub(r"(\w+)\_\d", r"\1", s)

    # Mispronounciation [splace/space] -> space
    s = re.sub(r"\[\w+\/(\w+)\]", r"\1", s)

    # Coinage. remove curly brackets... keep word
    s = re.sub(r"\{(\w*)\}", r"\1", s)

    # remove double spacing on last
    s = re.sub(r"\s\s+", " ", s)
    return s.strip()  # remove whitespace start/end


def extract_vad_list(anno):
    vad = [[], []]
    for channel in [0, 1]:
        for utt in anno[channel]:
            s, e = utt["start"], utt["end"]
            vad[channel].append((s, e))
    return vad


def extract_vad_list_from_words(anno, min_word_diff=0.05):
    vad = []
    for utt in anno:
      s, e = utt["words"][0]["start"], utt["words"][0]["end"]
      for w in utt["words"][1:]:
        if w["start"] - e < min_word_diff:
          e = w["end"]
        else:
          vad.append((s, e))
          s = w["start"]
          e = w["end"]

      vad.append((s, e))
    return vad


def extract_word_level_annotations(session, speaker, session_dir, apply_regexp=True):
    def remove_multiple_whitespace(s):
        s = re.sub(r"\t", " ", s)
        return re.sub(r"\s\s+", " ", s)

    # Load word-level annotations
    words_filename = "sw" + session + speaker + "-ms98-a-word.text"
    words_list = read_txt(join(session_dir, words_filename))

    # process word-level annotation
    word_dict = {}
    for word_row in words_list:
        word_row = remove_multiple_whitespace(word_row).strip()
        try:
            idx, wstart, wend, word = word_row.split(" ")
        except Exception as e:
            print("word_row: ", word_row)
            print("word_split: ", word_row.split(" "))
            print(e)
            input()
            assert False

        if apply_regexp:
            word = swb_regexp(word)

        # correct channels for wavefiles
        speaker = 0 if speaker == "A" else 1

        if not (word in OmitText or word == ""):
            if idx in word_dict:
                word_dict[idx].append(
                    {"text": word, "start": float(wstart), "end": float(wend), "speaker": speaker, "label": None }
                )
            else:
                word_dict[idx] = [
                    {"text": word, "start": float(wstart), "end": float(wend),"speaker": speaker, "label": None }
                ]
    return word_dict


def combine_speaker_utterance_and_words(
    session, speaker, session_dir, apply_regexp=True
):
    """Combines word- and utterance-level annotations"""
    # Read word-level annotation and format appropriately
    word_dict = extract_word_level_annotations(
        session, speaker, session_dir, apply_regexp=apply_regexp
    )

    # Read utterance-level annotation
    trans_filename = "sw" + session + speaker + "-ms98-a-trans.text"
    trans_list = read_txt(join(session_dir, trans_filename))

    # correct channels for wavefiles
    speaker = 0 if speaker == "A" else 1

    # Combine word-/utterance- level annotations
    utterances = []
    for row in trans_list:
        # utt_start/end are padded so we use exact word timings
        utt_idx, _, _, *words = row.split(" ")

        if not (words[0] in OmitText and len(words) == 1):  # only noise/silence
            wd = word_dict.get(utt_idx, None)
            if wd is None:
                continue

            words = " ".join(words)
            if apply_regexp:
                words = swb_regexp(words)

            utterances.append(
                {
                    "speaker": speaker,
                    "text": words,
                    "words": wd,
                    "start": wd[0]["start"],
                    "end": wd[-1]["end"],
                }
            )
    return utterances


def extract_dialog(session, session_dir, apply_regexp=True):
    """Extract the annotated dialogs and label words appropriately."""
    # Extract utterances with word-level details for both speakers
    a_utterances = combine_speaker_utterance_and_words(
        session, speaker="A", session_dir=session_dir, apply_regexp=apply_regexp
    )
    b_utterances = combine_speaker_utterance_and_words(
        session, speaker="B", session_dir=session_dir, apply_regexp=apply_regexp
    )

    # Combine and sort utterances by start time
    all_utterances = sorted(a_utterances + b_utterances, key=lambda x: x['start'])

    # Initialize labeled words list
    labeled_utterances = []

    # Identify backchannel utterances and non-backchannel utterances
    backchannel_utterances = []
    for utt in all_utterances:
        if utt['text'].lower() in BACKCHANNEL_PHRASES:
            backchannel_utterances.append(utt)
        else:
            labeled_utterances.append(utt)

    # Mark turn-taking at the last word before speaker change
    current_speaker = labeled_utterances[0]['speaker']
    for i in range(1, len(labeled_utterances)):
        if labeled_utterances[i]['speaker'] != current_speaker:
            # Mark the last word of the previous speaker as 'Turn-taking'
            labeled_utterances[i - 1]['words'][-1]['label'] = Labels.TURN_TAKING.value
            current_speaker = labeled_utterances[i]['speaker']

    labeled_utterances += backchannel_utterances
    labeled_utterances = sorted(labeled_utterances, key=lambda x: x['start'])

    for i in range(1, len(labeled_utterances)):
      utt = labeled_utterances[i]
      if utt['text'].lower() not in BACKCHANNEL_PHRASES: continue

      j = i - 1
      while j >= 0 and labeled_utterances[j]['speaker'] == utt['speaker']: j -= 1
      if j < 0: continue

      for k in range(len(labeled_utterances[j]['words']) - 1, -1, -1):
        if labeled_utterances[j]['words'][k]['end'] <= utt['start'] or k == 0:
          labeled_utterances[j]['words'][k]['label'] = Labels.BACKCHANNEL.value
          break

    for i in range(len(labeled_utterances)):
      for j in range(len(labeled_utterances[i]['words'])):
        if labeled_utterances[i]['words'][j]['label'] is None:
          labeled_utterances[i]['words'][j]['label'] = Labels.CONTINUING_SPEECH.value
      labeled_utterances[i]['label'] = labeled_utterances[i]['words'][-1]['label']

    return extract_sample(labeled_utterances)

def extract_sample(labeled_utterances):
  sample_utterances = []
  for i in range(len(labeled_utterances)):
    utt = labeled_utterances[i]
    words = utt['words']
    for j in range(len(words)):
      sample_utterances.append(
        {
          "text": ' '.join([w['text'] for w in words[:j + 1]]),
          "start": words[0]['start'],
          "end": words[j]['end'],
          "label": words[j]['label'],
          "speaker": utt['speaker'],
          "words": words[:j + 1]
        }
      )
  return sample_utterances

def remove_words_from_dialog(dialog):
  new_dialog = []
  for utt in dialog:
    new_dialog.append(
      {
        "text": utt["text"],
        "start": utt["start"],
        "end": utt["end"],
        "label": utt["label"]
      }
    )
  return new_dialog


if __name__ == "__main__":

    from os import listdir
    from datasets_turntaking.utils import read_json

    extracted_path = "/home/erik/.cache/huggingface/datasets/downloads/extracted/3bb5f33eb413284d4ef4098cadaccfa92b81653428c64ea8f954fe77a21c687c"
    session = "2001"
    session = "4936"

    session = str(session)
    session_dir = join(extracted_path, "swb_ms98_transcriptions", session[:2], session)
    print(listdir(session_dir))
    dialog = extract_dialog(session, session_dir)
    vad = extract_vad_list_from_words(dialog)

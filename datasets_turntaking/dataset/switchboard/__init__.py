import random
from os.path import join, expanduser
from datasets import load_dataset
from datasets_turntaking.utils import repo_root


DATASET_SCRIPT = join(repo_root(), "datasets_turntaking/dataset/switchboard/switchboard.py")
AUDIO_DIR = join(expanduser("~"), "Downloads/swb1_LDC97S62")
EXT = ".wav"


def count_labels(dset):
  count = [0, 0, 0]
  for examples in dset:
    for label in examples['dialog']['label']: count[label] += 1

  return count

def downsample_continuing_speech(examples, downsampling_factor):
    new_dialogs = {'start': [], 'end': [], 'text': [], 'label': []}
    for i, label in enumerate(examples["dialog"]['label']):
        if (label == 1 and random.random() < downsampling_factor) or label != 1:
          new_dialogs['start'].append(examples['dialog']['start'][i])
          new_dialogs['end'].append(examples['dialog']['end'][i])
          new_dialogs['text'].append(examples['dialog']['text'][i])
          new_dialogs['label'].append(label)

    examples["dialog"] = new_dialogs
    return examples

def load_switchboard(
    split="train",
    audio_root=AUDIO_DIR,
    ext=EXT,
    train_files=None,
    val_files=None,
    test_files=None,
):
    if split == "val":
        split = "validation"

    def process_and_add_name(examples):
        examples["dataset_name"] = "switchboard"
        if audio_root is not None:
            examples["audio_path"] = join(audio_root, examples["audio_path"] + ext)

        return examples

    dset = load_dataset(
        DATASET_SCRIPT,
        name="default",
        split=split,
        train_files=train_files,
        val_files=val_files,
        test_files=test_files
    )
    # dset = dset.remove_columns(["speaker_id", "chapter_id"])
    total_counts = count_labels(dset)
    downsampling_factor = (total_counts[0] + total_counts[2]) / (2 * total_counts[1])
    dset = dset.map(lambda x: downsample_continuing_speech(x, downsampling_factor))
    dset = dset.map(process_and_add_name)
    return dset

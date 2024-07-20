from icecream import ic
import pandas as pd
import torch

from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

pln = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1", use_auth_token="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"
)
ic(type(pln))
ic(pln)
pln.to(torch.device("cuda:0"))

with ProgressHook() as progress_hook:
    diarization = pln(file="data/yoroshiku_male.wav", hook=progress_hook, num_speakers=2)

for segment, track, label in diarization.itertracks(yield_label=True):
    ic(segment, track, label)

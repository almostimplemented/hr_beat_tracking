# ECS7006 Music Informatics 2022: Coursework 1

Beat tracking system based off a piano transcription architecture [1].

A Python virtual environment is bundled with this project.

Please use `source venv/bin/activate` to enable the environment.

```
from beat_crnn import beatTracker

beats, downbeats = beatTracker(inputFile)
```

See `example.py` and `eval.py` for more usage.


[1] https://arxiv.org/abs/2010.01815

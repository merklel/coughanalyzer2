# Conv-Net Cough Analyzer

DO LISTENERS OF MOZART COUGH MORE OFTEN DURING A CONCERT THAN LISTENERS OF BEETHOVEN?
I tried to give an answer to this exhilarating question using freely available data.

## Objective
This tool aims to detect coughing in classical music concerts. This major annoyance (prior corona) is researched on quite a lot and not understood fully.
Goal is to detect coughs and relate them to the composer, the kind of music, and the athmosphere of the music.


## Method
Training a convolutional neuronal network using synthetic coughs sampled randomly on live recordings of classical music concerts obtained from youtube.
The augmented data helps generating enough samples for the algorithm to learn, since coughing is not too common.

- FFTs are created from 2 second chunks of music. Some of them are augmented with cough-sound files freely available on the internet.
- 2D-Conv Network using keras is fitted on the data.
- A testing dataset as well as real cough-samples from concerts that are manually labeled are used to test the approach.


## Results
Algoithm is learning to detect coughing during music, but false positives vs. true positives is not yet quite satisfying.


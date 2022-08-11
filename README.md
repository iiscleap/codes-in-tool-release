# codes-in-tool-release
This is the official github repository for the [screening-tool](https://coswara.iisc.ac.in/) developed by LEAP lab, IISc. The repository contains codes for the classifier development, statistical analysis of the classifier results and the bias and fairness analysis for various subgroups of the population.

**To run the BLSTM classifier**:
1. Clone this repo `codes-in-tool-release`.
2. Seperately, download and extract the [Coswara dataset](https://github.com/iiscleap/Coswara-Data).
3. Make sure to put `Coswara-Data` folder generated from step 2 inside `/home/data/`.
4. Verify that `classifier_BLSTM_model/data/<audio-cat>/dev.scp` paths are now consistent for your local machine.
5. Enter the folder `classifier_BLSTM_model` inside `codes-in-tool-release`.
6. Type `./run.sh` in your terminal and ENTER.

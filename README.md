# codes-in-tool-release
This is the official github repository for the [screening-tool](https://coswara.iisc.ac.in/) developed by LEAP lab, IISc. The repository contains codes for the classifier development, statistical analysis of the classifier results and the bias and fairness analysis for various subgroups of the population.

**To get the BLSTM classifier results**:
1. Clone this repo `codes-in-tool-release`.
2. Seperately, download and extract the [Coswara dataset](https://github.com/iiscleap/Coswara-Data).
3. Make sure to put `Coswara-Data` folder generated from step 2 inside `/home/data/`.
4. Verify that `classifier_BLSTM_model/data/<audio-cat>/dev.scp` paths are now consistent in your local machine.
5. Enter the folder `classifier_BLSTM_model` inside `codes-in-tool-release`.
6. In terminal, run `./prpare_feats.sh` for all the 9 audio categories: `breathing-deep`, `breathing-shallow`, `cough-heavy`, `cough-shallow`, `vowel-a`, `vowel-e`, `vowel-o`, `counting-normal`, `counting-fast`.
6. Type `./run.sh` in your terminal and ENTER. Once done, you can find the results inside `results` folder.

**To get the transformer classifier results**:

Follow the steps described in **To get the BLSTM classifier results** after replacing `classifier_BLSTM_model` by `classifier_transformer_model` in steps 4 and 5.

**To get the logistic regression, random forest, multi-layer perceptron classifiers results**:

After following steps 1,2,3 from **To get the BLSTM classifier results**,
1. Verify that `classifier_LR_RF_MLP_models/data/<audio-cat>/all.scp` paths are now consistent in your local machine.
2. Inside the folder `classifier_LR_RF_MLP_models`, run `./run.sh` in Terminal. It runs for a specific `<audio-cat>` (e.g. `cough-heavy`). Any of the 9 audio categories can be set at `audiocategory` field at the top of `run.sh`.

**To get the bias and fairness analysis results**:
1. Go to `bias_analysis.py`.
2. `work_dir` should be set to path where bias analysis results are to be saved.
3. `csv_metadata_path`, `test1_path`, `test2_path` and `gnd_truth_path` may be set based on the classifier model to be analyzed for bias and fairness.
4. Run `python bias_analysis.py` from Terminal.
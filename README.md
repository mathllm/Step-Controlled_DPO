# Step-Controlled DPO

This is a repository for the paper: Step-Controlled DPO: Leveraging Stepwise Error for Enhanced Mathematical Reasoning.

The paper is at `paper/SCDPO.pdf`.

<p align="center">
  <img src="./images/data.pdf">
</p>

### DPO and SCDPO Data Generation

The code for DPO and SCDPO data generation is in `src`. `src/positive_negative_lce_gen`, `src/positive_negative_lce_gen_internlm_ape` and `src/positive_negative_lce_gen_mathcode_mistral_addsys` are for the generation of DPO data, while `src/step_controled_dpo_lce_internlm`, `src/step_controled_dpo_lce_mathcoder` and `src/step_controled_dpo_lce` are for the generation of SCDPO data.

API inference of internlm models are deployed using vllm, with an example script at `src/positive_negative_lce_gen_internlm_ape/scripts_1/deploy_vllm.sh`. Other models are deployed using an image from [huggingface/text-generation-inference](https://github.com/huggingface/text-generation-inference), with an exampe deploy script at `src/deploy-tgi.sh`

### SFT, DPO, and SCDPO Training

The training code is based on the repository [huggingface/alignment-handbook](https://github.com/huggingface/alignment-handbook). The code for SFT training is at `alignment-handbook/scripts/run_sft_lce.py`, which is adapted for code-integrated training. The code for DPO and SCDPO training is at `alignment-handbook/scripts/run_dpo.py`. The config yaml files and training shell scripts are at `/mnt/cache/luzimu/open_source_repositories/Step-Controlled_DPO/alignment-handbook/recipes`. You can modify the paths in the `.yaml` files and `.sh` files to train the models.

### Inference

The inference code is at `src/inference`. `inference_g.py`, `inference_m1.py` and `inference_m2.py` are for generating the solutions. `compute_acc.py` is for computing the accuracy of the generated solutions.
# T5Chem_pKa
A Unified Deep Learning Method for pKa and Protonation prediction
<img width="736" alt="Screenshot 2024-03-14 at 3 24 27 PM" src="https://github.com/charlotteinfante/t5chem_pKa/assets/96793416/361bbd06-4c5e-4cda-b709-3f14ebc16a0c">


Inspired by the work of Jieyu Lu et al. (2022) {https://pubs.acs.org/doi/full/10.1021/acs.jcim.1c01467}, we use T5Chem--a T5 model built on HuggingFace Transformers--to predict macroscopic pKa, mircoscopic pKa, and protonation sites of small molecules. 

## Models
We leverage the multitasking ability of T5Chem, and we added the prefixes "acidic" and "basic" to train __one__ macroscopic pKa model. Depending on the type of pKa the user would like to predict, they can use the different prefixes to 

<img width="552" alt="Screenshot 2024-03-13 at 5 33 37 PM" src="https://github.com/charlotteinfante/t5chem_pKa/assets/96793416/3877e501-be9b-497d-9225-f632cd2f74c3">



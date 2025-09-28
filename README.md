# MyCleanDiffuser

This project is **based on [CleanDiffuser](https://github.com/CleanDiffuserTeam/CleanDiffuser)**, with some modifications and extensions on top of their excellent work.  
Big thanks to the CleanDiffuser team for their great contribution to the community 🙏.

---

## 🚀 Installation

It is recommended to create an isolated environment using **conda**:

```bash
conda create -n myclean python=3.10 -y
conda activate myclean
cd CleanDiffuser
pip install -r requirements.txt
```

If you prefer venv, please make sure Python 3.10 is already installed on your system.


---

## ▶️ Usage

To run the main program (adjust `main.py` to your actual entry script):

```bash
python pipelines/dql_d4rl_mujoco.py
```

<!-- If additional configuration is required, you can specify it via arguments, e.g.:

```bash
python main.py --config config.yaml
``` -->

---

## 📦 Dependencies

All dependencies are listed in [`requirements.txt`](requirements.txt).  

- For **critical dependencies** (e.g., `torch`, `mujoco`), versions are pinned to ensure compatibility.  
- For **general utilities** (e.g., `tqdm`, `requests`), version ranges are relaxed to make installation easier.  

We recommend using **conda** for complex packages (CUDA, MuJoCo, OpenCV) to avoid compilation issues.

---

## ❓ Issues & Support

If you encounter problems during installation or running the code:  

- Please open an [Issue] 
- I’ll try my best to help  
- Contributions via PRs are also very welcome 😃  

---

## 🙏 Acknowledgements

This project is built upon [CleanDiffuser](https://github.com/CleanDiffuserTeam/CleanDiffuser).  
All credits go to the original authors — this is just a customized extension.  

---

## 📄 License

This repository follows the original CleanDiffuser license.  
Please make sure to comply with the corresponding terms when using or distributing this project.  

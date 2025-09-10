
---

### **NOTE** :- Open Preview with Markdown Preview Enhanced :-
```powershell
Ctrl + Shift + V     # → Opens a split Markdown Preview (renders nicely).

Or 

Ctrl + Shift + K V   # → Opens the preview in a new tab.
```

# 🚀 Full Guide: Virtual Environment in VsCode

### 🔹 Step 1 : Go To CMD [ for opening your project folder in VsCode ]

```
# You Will Initially Here ;
C:\Users\aamir>

# Change The Directory [E:] and Folder in oneline ; 
cd /d E:\Campus-X\campusx_mlops\campusx_project

# Check Your Current Path on CMD ;
echo %cd%

# Now you will here ; 
E:\Campus-X\campusx_mlops\campusx_project

# List all (including hidden files) ;
dir /a

# Now Open VsCode just write this;
code .
```

- OR 

### 🔹 Step 1: Open Project in VS Code

* Open **VS Code**
* Go to **File → Open Folder** and select your project folder:

  ```
  E:\Campus-X\campusx_mlops\campusx_project
  ```

---

### 🔹 Step 2: Open Integrated Terminal
* In VS Code, open a terminal:

  ```
  Terminal → New Terminal
  ```
* It should open at:

  ```
  PS E:\Campus-X\campusx_mlops\campusx_project>
  ```

---

### 🔹 Step 3: Delete Old Virtual Environment (if exists)

Run:

```powershell
Remove-Item -Recurse -Force .venv
```

---

### 🔹 Step 4: Create a New Virtual Environment

Run:

```powershell
python -m venv .venv

# Remember :- You can give your virtual environment folder any name you like

python -m venv .venv      # common default
python -m venv .myvenv    # ur custom name : ,  So if you use [.myvenv] activation command also changes: .myvenv\Scripts\Activate.ps1
python -m venv venv       # without dot, also common
python -m venv env        # another common choice

```

👉 This creates `.venv` folder in your project.

---

### 🔹 Step 5: Activate the Virtual Environment

Run:

```powershell
# if your venv is named .venv
.venv\Scripts\Activate.ps1

# if your venv is named .myvenv
.myvenv\Scripts\Activate.ps1
```

✅ If successful, your terminal prompt will change to:

```powershell
(.venv) PS E:\Campus-X\campusx_mlops\campusx_project>
```

---

### 🔹 Step 6: check Python Version & Upgrade pip

```powershell
# check python version
python --version

# upgrade pip
pip install --upgrade pip
python.exe -m pip install --upgrade pip
```

---

### 🔹 Step 7: Install Required Packages


```python

# you are here ;
(.venv) PS E:\Campus-X\campusx_mlops\campusx_project>

# now installed these pakages 
pip install cookiecutter

# check version
cookiecutter --version


# Generate Project Structure
cookiecutter https://github.com/drivendata/cookiecutter-data-science -c v1 --output-dir .

# ls
----------------------------------------------------------------------------------
Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
d-----        09-09-2025  12:23 PM                .venv
d-----        09-09-2025  12:51 PM                mlops-mini-project
-a----        09-09-2025  12:51 PM           4353 python_venv_guide_miniproject.md
------------------------------------------------------------------------------------

# go to that folder
cd mlops-mini-project

---------------------------------------------------------------------------------------------
(.venv)  PS E:\Campus-X\campusx_mlops\campusx_project\mlops-mini-project> ls
    Directory: E:\Campus-X\campusx_mlops\campusx_project\mlops-mini-project
Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
d-----        09-09-2025  12:51 PM                data
d-----        09-09-2025  12:51 PM                docs
d-----        09-09-2025  12:51 PM                models
d-----        09-09-2025  12:51 PM                notebooks
d-----        09-09-2025  12:51 PM                references
d-----        09-09-2025  12:51 PM                reports
d-----        09-09-2025  12:51 PM                src
-a----        09-09-2025  12:51 PM            471 .env
-a----        09-09-2025  12:51 PM           1092 .gitignore
-a----        09-09-2025  12:51 PM              2 LICENSE
-a----        09-09-2025  12:51 PM           4767 Makefile
-a----        09-09-2025  12:51 PM           2959 README.md
-a----        09-09-2025  12:51 PM            113 requirements.txt
-a----        09-09-2025  12:51 PM            208 setup.py
-a----        09-09-2025  12:51 PM            657 test_environment.py
-a----        09-09-2025  12:51 PM             53 tox.ini
---------------------------------------------------------------------------------------------

# initialize git repo ;
git init

# create github repo 'mlops-mini-project'

# connect with github ;
git remote add origin https://github.com/aamir490/mlops-mini-project.git

# verify connect or not ;
git remote -v
-----------------------------------------------------------------
origin  https://github.com/aamir490/mlops-mini-project.git (fetch)
origin  https://github.com/aamir490/mlops-mini-project.git (push)
------------------------------------------------------------------

# check status ;
git status

# git add
git add .

# commit
git commit -m "initail commit"

# check barnch ;
git branch

# push to master ;
git push origin -u main

# connect with Dagshub ;


# after connect go to 'expermint tab' and copy some impr link 
------------------------------------------------------------------
https://dagshub.com/aamir490/mlops-mini-project.mlflow

import dagshub
dagshub.init(repo_owner='aamir490', repo_name='mlops-mini-project', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)
------------------------------------------------------------------

# make a file under nootbooks 'dagub-setup.py'
---------------------------------------------------------------
import mlflow
import dagshub

dagshub.init(repo_owner='aamir490', repo_name='mlops-mini-project', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/aamir490/mlops-mini-project.mlflow')

with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)
---------------------------------------------------------------

# install these packages ;
pip install -q dvc mlflow pandas numpy dagshub scikit-learn

# check installed or not ;
pip show dvc mlflow pandas numpy dagshub scikit-learn


# in oneline  command
python -c "import dvc, mlflow, pandas, numpy, dagshub, sklearn; print('✅ All packages installed successfully!')"


# check the all installed installed version
import dvc, mlflow, pandas, numpy, dagshub, sklearn
print("✅ All packages installed successfully!")

# List all installed Python packages
pip list

# (Optional) Count how many packages are installed
pip list | measure-object

# (Optional) If you want a record of all installed libraries -OR- Save the list to a file 
pip freeze > requirements.txt

# (Optional) If you or someone else wants to recreate the same environment later
pip install -r requirements.txt

# run 'daghub_setup.py'
python E:\Campus-X\campusx_mlops\campusx_projectt\mlops-mini-project\notebooks\dagshub_setup.py
-----------------------------------------------------------------------------------------------
Accessing as aamir490
Initialized MLflow to track repo "aamir490/mlops-mini-project"
Repository aamir490/mlops-mini-project initialized!
🏃 View run gregarious-mole-106 at: https://dagshub.com/aamir490/mlops-mini-project.mlflow/#/experiments/0/runs/0caa006f4beb4889bdbfbc37e8c8b0a8
🧪 View experiment at: https://dagshub.com/aamir490/mlops-mini-project.mlflow/#/experiments/0
---------------------------------------------------------------------------------------------------------------------------------------------------------

#  go to 'noteboook'  E:\Campus-X\campusx_mlops\campusx_projectt\mlops-mini-project\notebooks\exp1_baseline_model (1).ipynb run this jupyter file


#git status
git status
----------------------------------------------------------------------------------------
On branch main
Your branch is up to date with 'origin/main'.

Untracked files:
  (use "git add <file>..." to include in what will be committed)
        notebooks/dagshub_setup.py
        notebooks/exp1_baseline_model (1).ipynb
        notebooks/logreg_model.pkl
        notebooks/mlruns/
------------------------------------------------------------------------------

# CHECK SIZE :- 
# Example for the pickle file
"{0:N2} MB" -f ((Get-Item "notebooks/logreg_model.pkl").Length / 1MB)

# Example for the notebook
"{0:N2} MB" -f ((Get-Item "notebooks/exp1_baseline_model (1).ipynb").Length / 1MB)

# 2️⃣ Check all files in the notebooks folder
Get-ChildItem "notebooks" | Select-Object Name, @{Name="Size_MB";Expression={"{0:N2}" -f ($_.Length/1MB)}}


git add .

git commit -m "second commit"

git branch  

git push origin -u main


pip install xgboost

python E:\Campus-X\campusx_mlops\campusx_projectt\mlops-mini-project\notebooks\exp1_bow_vs_tfidf.py


token
23519ac2940aab855769c067b98b978ddfe587f3

$env:DAGSHUB_PAT="23519ac2940aab855769c067b98b978ddfe587f3"

```










---

### 🔹 Step 8: Select Interpreter in VS Code

This is **important**, otherwise VS Code may not use your `.venv`.

1. Press `press F1` - OR - Go to `View` --> `Command Pallete` & then ;
2. Search / Type :-  **Python: Select Interpreter**
3. Choose the one that looks like:

   ```
   .venv(3.12.3)\Scripts\python.exe
   ```

   (inside your project folder)

✅ Now VS Code will use your project’s virtual environment for running scripts.

---

### 🔹 Step 9: Deactivate When Finished

When you’re done working:

```powershell
deactivate
```

---

✨ Now your VS Code project is fully set up with an isolated environment.






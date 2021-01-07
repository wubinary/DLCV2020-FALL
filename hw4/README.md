# HW4 - Few-Shot Classification
# Usage
To start working on this assignment, you should clone this repository into your local machine by using the following command.

    git clone https://github.com/DLCV-Fall-2020/hw4-<username>.git
Note that you should replace `<username>` with your own GitHub username.

For more details, please click [this link](https://drive.google.com/file/d/1YN_8gCIfxB5AvBZ7ruD6bFkEbRTHqL9L/view?usp=sharing) to view the slides of HW4. **Note that hw4 video and introduction pdf files can be accessed in your NTU COOL.**

### Dataset
In the starter code of this repository, we have provided a shell script for downloading and extracting the dataset for this assignment. For Linux users, simply use the following command.

    bash ./get_dataset.sh
The shell script will automatically download the dataset and store the data in a folder called `hw4_data`. Note that this command by default only works on Linux. If you are using other operating systems, you should download the dataset from [this link](https://drive.google.com/file/d/1c4nEjrUISeSl7LEf9VUmkpKwvdx3fnuj/view?usp=sharing) and unzip the compressed file manually.
> ⚠️ ***IMPORTANT NOTE*** ⚠️  
> You should keep a copy of the dataset only in your local machine. **DO NOT** upload the dataset to this remote repository. If you extract the dataset manually, be sure to put them in a folder called `hw4_data` under the root directory of your local repository so that it will be included in the default `.gitignore` file.

### Evaluation
To evaluate your models in Problems 1~3, you can run the evaluation script provided in the starter code by using the following command.

    python3 eval.py $1 $2

 - `$1` is the path to your predicted results (e.g. `output/val_pred.csv`)
 - `$2` is the path to the ground truth (e.g. `hw4_data/val_testcase_gt.csv	`)

Note that for `eval.py` to work, your predicted `.csv` files should have the same format as the ground truth files `val_testcase_gt.csv` provided in the dataset.

# Submission Rules
### Deadline
109/12/29 (Tue.) 02:00 AM (GMT+8)

### Late Submission Policy
You have a three-day delay quota for the whole semester. Once you have exceeded your quota, the credit of any late submission will be deducted by 30% each day.

Note that while it is possible to continue your work in this repository after the deadline, **we will by default grade your last commit before the deadline** specified above. If you wish to use your quota or submit an earlier version of your repository, please contact the TAs and let them know which commit to grade.

### Academic Honesty
-   Taking any unfair advantages over other class members (or letting anyone do so) is strictly prohibited. Violating university policy would result in an **F** grade for this course (**NOT** negotiable).    
-   If you refer to some parts of the public code, you are required to specify the references in your report (e.g. URL to GitHub repositories).      
-   You are encouraged to discuss homework assignments with your fellow class members, but you must complete the assignment by yourself. TAs will compare the similarity of everyone’s submission. Any form of cheating or plagiarism will not be tolerated and will also result in an **F** grade for students with such misconduct.

### Submission Format
Aside from your own Python scripts and model files, you should make sure that your submission includes *at least* the following files in the **root directory** of this repository:
 1.   `hw4_<StudentID>.pdf`  
The report of your homework assignment. Refer to the "*Grading*" section in the slides for what you should include in the report. Note that you should replace `<StudentID>` with your student ID, **NOT** your GitHub username.
 2.   `hw4_download.sh` 
 Download all the models needed for Problem 1~3. We will execute this script first.
 3.   `hw4_1.sh`  
The shell script file for running your Prototypical Network.
This script takes as input a folder containing testing images, and should output the predicted results in a `.csv` file.
 4.   `hw4_2.sh`  
The shell script file for running your network with data hallicination.
This script takes as input a folder containing testing images, and should output the predicted results in a `.csv` file.
 5.   `hw4_3.sh`  
The shell script file for running your network with improved data hallicination.. 
This script takes as input a folder containing testing images, and should output the predicted results in a `.csv` file.


TA will run your code in the following manner:

    bash hw4_download.sh
    bash hw4_1.sh $1 $2 $3 $4
    bash hw4_2.sh $1 $2 $3 $4
    bash hw4_3.sh $1 $2 $3 $4 $5 $6

-   `$1` testing images csv file (e.g., `hw4_data/val.csv`)
-   `$2` testing images directory (e.g., `hw4_data/val`)
-   `$3` path of test case on test set (e.g., `hw4_data/val_testcase.csv`)
-   `$4` path of output csv file (predicted labels) (e.g., `output/val_pred.csv`)
-   `$5`: training images csv file (e.g., `hw4_data/train.csv`)
-   `$6`: training images directory (e.g., `hw4_data/train`)

> 🆕 ***NOTE***  
> For the sake of conformity, please use the `python3` command to call your `.py` files in all your shell scripts. Do not use `python` or other aliases, otherwise your commands may fail in our autograding scripts.

### Packages
This homework should be done using python3.6. For a list of packages you are allowed to import in this assignment, please refer to the requirments.txt for more details.

You can run the following command to install all the packages listed in the requirements.txt:

    pip3 install -r requirements.txt

Note that using packages with different versions will very likely lead to compatibility issues, so make sure that you install the correct version if one is specified above. E-mail or ask the TAs first if you want to import other packages.


# Q&A
If you have any problems related to HW4, you may
- Use TA hours
- Contact TAs by e-mail ([ntudlcv@gmail.com](mailto:ntudlcv@gmail.com))
- Post your question under hw4 FAQ section in FB group

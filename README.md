# Facial Emotion Recognition of Spatial Image using Gabor Filter.
### Hardware and Software Used:-
<p>
This software is developed on one flavour of linux Kernel Ubuntu and all instructions are only valid if you are using Ubuntu20.04. We are also providing instructions if you are using windows to run. 
</p>

#### Hardware Requirement:

 A Personal Computer(PC) with following minimum requirements.
 * 8GB RAM recommended; 
 * Disk Space:10 GB

#### Software Requirement:
* Operating System should be only Ubuntu20.04 amd64. 
* Python3.8
* Web Browser to host application (Google Chrome, Firefox Recommended)


#### Instruction to use software in Ubuntu20.04 System:

**Step1:** Open Terminal (Make sure that you have **internet access** because system need to install other dependecies.)

**Step2:** Place the startup.sh in **$HOME** directory.

**Step3:** Run the command **bash startup.sh** (Wait for some time. It will install the directories and all the file dependencies on your system. It will approximetly take 30-45 min)

**Step4:**  A web application will open on your browser(Google chrome recommended) and give you option to choose an image.
 

**Step5:** On the left sidebar, you get small details about our software. In main page you will get the option to upload the image. We are providing some sample data for you to test the application. You get this image in the folder **Image sample**.
 
 ![image](/Images/1.jpg)
 
 ![image](/Images/3.jpg)

**Step6:** After uploading it will automatically show the original image in the before section and after on right hand side in the after section you get the image after applying gabor filter. In the last you will see the predicted emotion of that image. 

![image](/Images/4.jpg)

**You will get result.**



#### Instruction to use software in Windows System (if you are using Windows10 OS with Python3.8):

**Step1:** Before proceeding, First unzip the **Data.zip** file. There you will get two folders **dataset CK+** and **Image samples**. Folder **dataset ck+** contains the image used to train the Convolutional Neural Network.

**Step2:** Run “_pip install -r requirements.txt_” in the cmd to install all the important modules and frameworks of python. If you are using pip3 then run “_pip3 install -r requirements.txt_”.

**Step 3:** Navigate to your parent directory of software using
                                     “_cd route_to_your_parent_directory” in cmd_.

**Step4:** Run all cells of the file **Main.ipynb** in the jupyter instance. It is the file that makes the CNN model for our program. It is only one-time process. No need to run again and again. Make sure that there is a folder **my_model** in the parent directory.

**Step5:** Run “_streamlit run app.py_” in the cmd to run the application on your web browser.
 You get a result something like this.
 
 ![image](/Images/2.jpg)
 
 **Step6:** On the left sidebar, you get small details about our software. In main page you will get the option to upload the image. We are providing some sample data for you to test the application. You get this image in the folder **Image sample**.
 
 ![image](/Images/1.jpg)
 
 ![image](/Images/3.jpg)
 
**Step7:** After uploading it will automatically show the original image in the before section and after on right hand side in the after section you get the image after applying gabor filter. In the last you will see the predicted emotion of that image. 

![image](/Images/4.jpg)

**Note:** There are some extra files are present which is supporting file of **app.py** and **main.ipynb**. Removal of these file might cause the program failure.

### Thank you for reading me.


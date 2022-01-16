# Facial Emotion Recognition of Spatial Image using Gabor Filter.
### Hardware and Software Used:-
<p>
This software is developed on Windows 10 and all instructions are only valid if you are using Windows 10 OS.
</p>

#### Hardware Requirement:

 * 64-bit distribution capable of running 32-bit applications.
 * 8GB RAM recommended; 
 * GPU 4GB (Optional)
 * Disk Space:10 GB

#### Software Requirement:
* Web Browser to host application (Google Chrome, Firefox Recommended)
* IDE to run Python Program
* CUDA Toolkit, If you are PC supported CUDA. (optional)

#### Instruction to use software:

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


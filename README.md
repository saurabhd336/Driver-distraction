# Driver distraction detection system

# How to run
Download dlib.
```bash
git clone https://github.com/davisking/dlib.git
```

Install OpenCV.

Clone this repo inside dlib main folder
```bash
cd dlib
git clone https://github.com/saurabhd336/Driver-distraction.git
```


Download and extract file from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
Copy the shape_predictor_68_face_landmarks.dat file inside Driver-distraction folder

Then use cmake to build.

```bash
cd Driver-distraction
mkdir build
cd build
cmake ..
cmake --build .
cp ../model.bin .
cp ../shape_predictor_68_face_landmarks.dat .
./driver_dist
```

# What is this?
A modified implementation of http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7478592

# Results
The application could correctly detect and classify head poses using a webcam video feed at about 11 fps.
Images attached.
![Alt text](/Result/image_Centre_Stack.jpg?raw=true "Centre Stack")
![Alt text](/Result/image_Down.jpg?raw=true "Down")
![Alt text](/Result/image_Instrument_stack.jpg?raw=true "Instrument_stack")
![Alt text](/Result/image_Left.jpg?raw=true "Left")
![Alt text](/Result/image_Rear_view_mirror.jpg?raw=true "Rear view mirror")
![Alt text](/Result/image_Right.jpg?raw=true "Right")
![Alt text](/Result/image_Road.jpg?raw=true "Road")
![Alt text](/Result/image_Up.jpg?raw=true "Up")

# Todo
Add detailed explanation


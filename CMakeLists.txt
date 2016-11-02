#
# This is a CMake makefile.  You can find the cmake utility and
# information about it at http://www.cmake.org
#


cmake_minimum_required(VERSION 2.8.4)

PROJECT(driver_distraction)

include(../dlib/cmake)

# Tell CMake to compile a program.  We do this with the ADD_EXECUTABLE()
# statement which takes the name of the output executable and then a list of
# .cpp files to compile.  Here each example consists of only one .cpp file but
# in general you will make programs that const of many .cpp files. 


if (DLIB_NO_GUI_SUPPORT)
   message("No GUI support, so we won't build the driver_dist.")
else()
   find_package(OpenCV QUIET)
   if (OpenCV_FOUND)
      include_directories(${OpenCV_INCLUDE_DIRS})

      ADD_EXECUTABLE(driver_dist main.cpp svm.cpp)
      TARGET_LINK_LIBRARIES(driver_dist dlib ${OpenCV_LIBS} )
   else()
      message("OpenCV not found, so we won't build the driver_dist example.")
   endif()
endif()



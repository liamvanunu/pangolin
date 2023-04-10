# Find Pangolin (https://github.com/stevenlovegrove/Pangolin)
find_package(Pangolin 0.3 REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

find_package(OpenCV 3.0 QUIET)
if (NOT OpenCV_FOUND)
    find_package(OpenCV 2.4.3 QUIET)
    if (NOT OpenCV_FOUND)
        message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
    endif ()
endif ()

add_executable(ModelViewer main.cpp)
target_link_libraries(ModelViewer ${Pangolin_LIBRARIES} ${OpenCV_LIBS})

# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/liam/dev/rbd/addons/Pangolin

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/liam/dev/rbd/addons/Pangolin/build

# Include any dependencies generated for this target.
include examples/SimpleScene/CMakeFiles/SimpleScene.dir/depend.make

# Include the progress variables for this target.
include examples/SimpleScene/CMakeFiles/SimpleScene.dir/progress.make

# Include the compile flags for this target's objects.
include examples/SimpleScene/CMakeFiles/SimpleScene.dir/flags.make

examples/SimpleScene/CMakeFiles/SimpleScene.dir/main.cpp.o: examples/SimpleScene/CMakeFiles/SimpleScene.dir/flags.make
examples/SimpleScene/CMakeFiles/SimpleScene.dir/main.cpp.o: ../examples/SimpleScene/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liam/dev/rbd/addons/Pangolin/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/SimpleScene/CMakeFiles/SimpleScene.dir/main.cpp.o"
	cd /home/liam/dev/rbd/addons/Pangolin/build/examples/SimpleScene && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/SimpleScene.dir/main.cpp.o -c /home/liam/dev/rbd/addons/Pangolin/examples/SimpleScene/main.cpp

examples/SimpleScene/CMakeFiles/SimpleScene.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/SimpleScene.dir/main.cpp.i"
	cd /home/liam/dev/rbd/addons/Pangolin/build/examples/SimpleScene && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liam/dev/rbd/addons/Pangolin/examples/SimpleScene/main.cpp > CMakeFiles/SimpleScene.dir/main.cpp.i

examples/SimpleScene/CMakeFiles/SimpleScene.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/SimpleScene.dir/main.cpp.s"
	cd /home/liam/dev/rbd/addons/Pangolin/build/examples/SimpleScene && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liam/dev/rbd/addons/Pangolin/examples/SimpleScene/main.cpp -o CMakeFiles/SimpleScene.dir/main.cpp.s

# Object files for target SimpleScene
SimpleScene_OBJECTS = \
"CMakeFiles/SimpleScene.dir/main.cpp.o"

# External object files for target SimpleScene
SimpleScene_EXTERNAL_OBJECTS =

examples/SimpleScene/SimpleScene: examples/SimpleScene/CMakeFiles/SimpleScene.dir/main.cpp.o
examples/SimpleScene/SimpleScene: examples/SimpleScene/CMakeFiles/SimpleScene.dir/build.make
examples/SimpleScene/SimpleScene: libpango_display.so
examples/SimpleScene/SimpleScene: libpango_scene.so
examples/SimpleScene/SimpleScene: libpango_windowing.so
examples/SimpleScene/SimpleScene: libpango_vars.so
examples/SimpleScene/SimpleScene: libpango_opengl.so
examples/SimpleScene/SimpleScene: libpango_image.so
examples/SimpleScene/SimpleScene: libpango_core.so
examples/SimpleScene/SimpleScene: /usr/lib/x86_64-linux-gnu/libGLEW.so
examples/SimpleScene/SimpleScene: /usr/lib/x86_64-linux-gnu/libOpenGL.so
examples/SimpleScene/SimpleScene: /usr/lib/x86_64-linux-gnu/libGLX.so
examples/SimpleScene/SimpleScene: /usr/lib/x86_64-linux-gnu/libGLU.so
examples/SimpleScene/SimpleScene: examples/SimpleScene/CMakeFiles/SimpleScene.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/liam/dev/rbd/addons/Pangolin/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable SimpleScene"
	cd /home/liam/dev/rbd/addons/Pangolin/build/examples/SimpleScene && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/SimpleScene.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/SimpleScene/CMakeFiles/SimpleScene.dir/build: examples/SimpleScene/SimpleScene

.PHONY : examples/SimpleScene/CMakeFiles/SimpleScene.dir/build

examples/SimpleScene/CMakeFiles/SimpleScene.dir/clean:
	cd /home/liam/dev/rbd/addons/Pangolin/build/examples/SimpleScene && $(CMAKE_COMMAND) -P CMakeFiles/SimpleScene.dir/cmake_clean.cmake
.PHONY : examples/SimpleScene/CMakeFiles/SimpleScene.dir/clean

examples/SimpleScene/CMakeFiles/SimpleScene.dir/depend:
	cd /home/liam/dev/rbd/addons/Pangolin/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/liam/dev/rbd/addons/Pangolin /home/liam/dev/rbd/addons/Pangolin/examples/SimpleScene /home/liam/dev/rbd/addons/Pangolin/build /home/liam/dev/rbd/addons/Pangolin/build/examples/SimpleScene /home/liam/dev/rbd/addons/Pangolin/build/examples/SimpleScene/CMakeFiles/SimpleScene.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/SimpleScene/CMakeFiles/SimpleScene.dir/depend


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
include examples/BasicOpenGL/CMakeFiles/tutorial_2_gl_intro_classic_triangle_vbo.dir/depend.make

# Include the progress variables for this target.
include examples/BasicOpenGL/CMakeFiles/tutorial_2_gl_intro_classic_triangle_vbo.dir/progress.make

# Include the compile flags for this target's objects.
include examples/BasicOpenGL/CMakeFiles/tutorial_2_gl_intro_classic_triangle_vbo.dir/flags.make

examples/BasicOpenGL/CMakeFiles/tutorial_2_gl_intro_classic_triangle_vbo.dir/2_gl_intro_classic_triangle_vbo.cpp.o: examples/BasicOpenGL/CMakeFiles/tutorial_2_gl_intro_classic_triangle_vbo.dir/flags.make
examples/BasicOpenGL/CMakeFiles/tutorial_2_gl_intro_classic_triangle_vbo.dir/2_gl_intro_classic_triangle_vbo.cpp.o: ../examples/BasicOpenGL/2_gl_intro_classic_triangle_vbo.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liam/dev/rbd/addons/Pangolin/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/BasicOpenGL/CMakeFiles/tutorial_2_gl_intro_classic_triangle_vbo.dir/2_gl_intro_classic_triangle_vbo.cpp.o"
	cd /home/liam/dev/rbd/addons/Pangolin/build/examples/BasicOpenGL && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tutorial_2_gl_intro_classic_triangle_vbo.dir/2_gl_intro_classic_triangle_vbo.cpp.o -c /home/liam/dev/rbd/addons/Pangolin/examples/BasicOpenGL/2_gl_intro_classic_triangle_vbo.cpp

examples/BasicOpenGL/CMakeFiles/tutorial_2_gl_intro_classic_triangle_vbo.dir/2_gl_intro_classic_triangle_vbo.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tutorial_2_gl_intro_classic_triangle_vbo.dir/2_gl_intro_classic_triangle_vbo.cpp.i"
	cd /home/liam/dev/rbd/addons/Pangolin/build/examples/BasicOpenGL && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liam/dev/rbd/addons/Pangolin/examples/BasicOpenGL/2_gl_intro_classic_triangle_vbo.cpp > CMakeFiles/tutorial_2_gl_intro_classic_triangle_vbo.dir/2_gl_intro_classic_triangle_vbo.cpp.i

examples/BasicOpenGL/CMakeFiles/tutorial_2_gl_intro_classic_triangle_vbo.dir/2_gl_intro_classic_triangle_vbo.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tutorial_2_gl_intro_classic_triangle_vbo.dir/2_gl_intro_classic_triangle_vbo.cpp.s"
	cd /home/liam/dev/rbd/addons/Pangolin/build/examples/BasicOpenGL && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liam/dev/rbd/addons/Pangolin/examples/BasicOpenGL/2_gl_intro_classic_triangle_vbo.cpp -o CMakeFiles/tutorial_2_gl_intro_classic_triangle_vbo.dir/2_gl_intro_classic_triangle_vbo.cpp.s

# Object files for target tutorial_2_gl_intro_classic_triangle_vbo
tutorial_2_gl_intro_classic_triangle_vbo_OBJECTS = \
"CMakeFiles/tutorial_2_gl_intro_classic_triangle_vbo.dir/2_gl_intro_classic_triangle_vbo.cpp.o"

# External object files for target tutorial_2_gl_intro_classic_triangle_vbo
tutorial_2_gl_intro_classic_triangle_vbo_EXTERNAL_OBJECTS =

examples/BasicOpenGL/tutorial_2_gl_intro_classic_triangle_vbo: examples/BasicOpenGL/CMakeFiles/tutorial_2_gl_intro_classic_triangle_vbo.dir/2_gl_intro_classic_triangle_vbo.cpp.o
examples/BasicOpenGL/tutorial_2_gl_intro_classic_triangle_vbo: examples/BasicOpenGL/CMakeFiles/tutorial_2_gl_intro_classic_triangle_vbo.dir/build.make
examples/BasicOpenGL/tutorial_2_gl_intro_classic_triangle_vbo: libpango_display.so
examples/BasicOpenGL/tutorial_2_gl_intro_classic_triangle_vbo: libpango_windowing.so
examples/BasicOpenGL/tutorial_2_gl_intro_classic_triangle_vbo: libpango_opengl.so
examples/BasicOpenGL/tutorial_2_gl_intro_classic_triangle_vbo: libpango_image.so
examples/BasicOpenGL/tutorial_2_gl_intro_classic_triangle_vbo: /usr/lib/x86_64-linux-gnu/libGLEW.so
examples/BasicOpenGL/tutorial_2_gl_intro_classic_triangle_vbo: /usr/lib/x86_64-linux-gnu/libOpenGL.so
examples/BasicOpenGL/tutorial_2_gl_intro_classic_triangle_vbo: /usr/lib/x86_64-linux-gnu/libGLX.so
examples/BasicOpenGL/tutorial_2_gl_intro_classic_triangle_vbo: /usr/lib/x86_64-linux-gnu/libGLU.so
examples/BasicOpenGL/tutorial_2_gl_intro_classic_triangle_vbo: libpango_vars.so
examples/BasicOpenGL/tutorial_2_gl_intro_classic_triangle_vbo: libpango_core.so
examples/BasicOpenGL/tutorial_2_gl_intro_classic_triangle_vbo: examples/BasicOpenGL/CMakeFiles/tutorial_2_gl_intro_classic_triangle_vbo.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/liam/dev/rbd/addons/Pangolin/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable tutorial_2_gl_intro_classic_triangle_vbo"
	cd /home/liam/dev/rbd/addons/Pangolin/build/examples/BasicOpenGL && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/tutorial_2_gl_intro_classic_triangle_vbo.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/BasicOpenGL/CMakeFiles/tutorial_2_gl_intro_classic_triangle_vbo.dir/build: examples/BasicOpenGL/tutorial_2_gl_intro_classic_triangle_vbo

.PHONY : examples/BasicOpenGL/CMakeFiles/tutorial_2_gl_intro_classic_triangle_vbo.dir/build

examples/BasicOpenGL/CMakeFiles/tutorial_2_gl_intro_classic_triangle_vbo.dir/clean:
	cd /home/liam/dev/rbd/addons/Pangolin/build/examples/BasicOpenGL && $(CMAKE_COMMAND) -P CMakeFiles/tutorial_2_gl_intro_classic_triangle_vbo.dir/cmake_clean.cmake
.PHONY : examples/BasicOpenGL/CMakeFiles/tutorial_2_gl_intro_classic_triangle_vbo.dir/clean

examples/BasicOpenGL/CMakeFiles/tutorial_2_gl_intro_classic_triangle_vbo.dir/depend:
	cd /home/liam/dev/rbd/addons/Pangolin/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/liam/dev/rbd/addons/Pangolin /home/liam/dev/rbd/addons/Pangolin/examples/BasicOpenGL /home/liam/dev/rbd/addons/Pangolin/build /home/liam/dev/rbd/addons/Pangolin/build/examples/BasicOpenGL /home/liam/dev/rbd/addons/Pangolin/build/examples/BasicOpenGL/CMakeFiles/tutorial_2_gl_intro_classic_triangle_vbo.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/BasicOpenGL/CMakeFiles/tutorial_2_gl_intro_classic_triangle_vbo.dir/depend


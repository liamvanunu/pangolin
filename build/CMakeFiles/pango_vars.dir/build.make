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
include CMakeFiles/pango_vars.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/pango_vars.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/pango_vars.dir/flags.make

CMakeFiles/pango_vars.dir/components/pango_vars/src/vars.cpp.o: CMakeFiles/pango_vars.dir/flags.make
CMakeFiles/pango_vars.dir/components/pango_vars/src/vars.cpp.o: ../components/pango_vars/src/vars.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liam/dev/rbd/addons/Pangolin/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/pango_vars.dir/components/pango_vars/src/vars.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pango_vars.dir/components/pango_vars/src/vars.cpp.o -c /home/liam/dev/rbd/addons/Pangolin/components/pango_vars/src/vars.cpp

CMakeFiles/pango_vars.dir/components/pango_vars/src/vars.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pango_vars.dir/components/pango_vars/src/vars.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liam/dev/rbd/addons/Pangolin/components/pango_vars/src/vars.cpp > CMakeFiles/pango_vars.dir/components/pango_vars/src/vars.cpp.i

CMakeFiles/pango_vars.dir/components/pango_vars/src/vars.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pango_vars.dir/components/pango_vars/src/vars.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liam/dev/rbd/addons/Pangolin/components/pango_vars/src/vars.cpp -o CMakeFiles/pango_vars.dir/components/pango_vars/src/vars.cpp.s

CMakeFiles/pango_vars.dir/components/pango_vars/src/varstate.cpp.o: CMakeFiles/pango_vars.dir/flags.make
CMakeFiles/pango_vars.dir/components/pango_vars/src/varstate.cpp.o: ../components/pango_vars/src/varstate.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liam/dev/rbd/addons/Pangolin/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/pango_vars.dir/components/pango_vars/src/varstate.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pango_vars.dir/components/pango_vars/src/varstate.cpp.o -c /home/liam/dev/rbd/addons/Pangolin/components/pango_vars/src/varstate.cpp

CMakeFiles/pango_vars.dir/components/pango_vars/src/varstate.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pango_vars.dir/components/pango_vars/src/varstate.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liam/dev/rbd/addons/Pangolin/components/pango_vars/src/varstate.cpp > CMakeFiles/pango_vars.dir/components/pango_vars/src/varstate.cpp.i

CMakeFiles/pango_vars.dir/components/pango_vars/src/varstate.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pango_vars.dir/components/pango_vars/src/varstate.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liam/dev/rbd/addons/Pangolin/components/pango_vars/src/varstate.cpp -o CMakeFiles/pango_vars.dir/components/pango_vars/src/varstate.cpp.s

# Object files for target pango_vars
pango_vars_OBJECTS = \
"CMakeFiles/pango_vars.dir/components/pango_vars/src/vars.cpp.o" \
"CMakeFiles/pango_vars.dir/components/pango_vars/src/varstate.cpp.o"

# External object files for target pango_vars
pango_vars_EXTERNAL_OBJECTS =

libpango_vars.so: CMakeFiles/pango_vars.dir/components/pango_vars/src/vars.cpp.o
libpango_vars.so: CMakeFiles/pango_vars.dir/components/pango_vars/src/varstate.cpp.o
libpango_vars.so: CMakeFiles/pango_vars.dir/build.make
libpango_vars.so: libpango_core.so
libpango_vars.so: CMakeFiles/pango_vars.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/liam/dev/rbd/addons/Pangolin/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX shared library libpango_vars.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pango_vars.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/pango_vars.dir/build: libpango_vars.so

.PHONY : CMakeFiles/pango_vars.dir/build

CMakeFiles/pango_vars.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/pango_vars.dir/cmake_clean.cmake
.PHONY : CMakeFiles/pango_vars.dir/clean

CMakeFiles/pango_vars.dir/depend:
	cd /home/liam/dev/rbd/addons/Pangolin/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/liam/dev/rbd/addons/Pangolin /home/liam/dev/rbd/addons/Pangolin /home/liam/dev/rbd/addons/Pangolin/build /home/liam/dev/rbd/addons/Pangolin/build /home/liam/dev/rbd/addons/Pangolin/build/CMakeFiles/pango_vars.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/pango_vars.dir/depend


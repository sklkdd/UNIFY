cmake_minimum_required (VERSION 2.8.12)
project(hnsw_lib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Find and setup OpenMP
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
#   SET( CMAKE_CXX_FLAGS  "-Ofast -DNDEBUG -std=c++11 -DHAVE_CXX0X -openmp -fpic -ftree-vectorize")
#   else()
#     if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
#       SET( CMAKE_CXX_FLAGS  "-Ofast -DNDEBUG -std=c++11 -DHAVE_CXX0X -openmp -march=native -fpic -ftree-vectorize")
#     elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
#       SET( CMAKE_CXX_FLAGS  "-Ofast -lrt -DNDEBUG -std=c++11 -DHAVE_CXX0X -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0" )
#     elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
#       SET( CMAKE_CXX_FLAGS  "-Ofast -lrt -DNDEBUG -std=c++11 -DHAVE_CXX0X -openmp -march=native -fpic -w -fopenmp -ftree-vectorize" )
#     endif()
# endif()

# add_executable(test_updates examples/updates_test.cpp)
# target_link_libraries(test_updates hnswlib)

# add_executable(searchKnnCloserFirst_test examples/searchKnnCloserFirst_test.cpp)
# target_link_libraries(searchKnnCloserFirst_test hnswlib)

# add_executable(main main.cpp sift_1b.cpp)
# target_link_libraries(main hnswlib)

# add_subdirectory(googletest)

include_directories(.)

# Add tests subdirectory
add_subdirectory(tests)

# Original approach for other test files if they exist
file(GLOB TESTS tests/*.cpp)
FOREACH (path ${TESTS})
    get_filename_component(name ${path} NAME_WE)
    # Skip files that are handled by tests/CMakeLists.txt
    if (NOT name MATCHES "buildindex_wrapper|search_wrapper|fvecs_to_bin")
        add_executable(${name} ${path})
    endif()
ENDFOREACH ()




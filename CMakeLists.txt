project(AT-polyscope)
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
SET(WITH_LIBIGL OFF)
SET(WITH_CGAL OFF)
SET(WITH_GMP ON)
include(dgtal)
include(polyscope)

add_executable(main main.cpp)
target_link_libraries(main polyscope DGtal ${DGTAL_LIBRARIES})

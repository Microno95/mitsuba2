set(MTS_PLUGIN_PREFIX "integrators")

add_plugin(depth   depth.cpp)
add_plugin(direct  direct.cpp)
add_plugin(path    path.cpp)
add_plugin(aov     aov.cpp)
add_plugin(stokes  stokes.cpp)
add_plugin(moment  moment.cpp)
add_plugin(volpath  volpath.cpp)
add_plugin(volpath_emissive  volpath_emissive.cpp)
add_plugin(volpath_raymarching  volpath_raymarching.cpp)
add_plugin(volpathmis volpathmis.cpp)
add_plugin(volpathmis_emissive  volpathmis_emissive.cpp)
add_plugin(volpathmis_raymarching  volpathmis_raymarching.cpp)

# Register the test directory
add_tests(${CMAKE_CURRENT_SOURCE_DIR}/tests)

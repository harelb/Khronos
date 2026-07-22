find_package(GTest REQUIRED)

include(GoogleTest)
enable_testing()

add_executable(
  test_${PROJECT_NAME}
  test_instance_forwarding_depth_filter.cpp
)
target_link_libraries(
  test_${PROJECT_NAME} ${PROJECT_NAME} GTest::gtest_main
)
gtest_add_tests(TARGET test_${PROJECT_NAME})

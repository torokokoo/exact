#add_executable(exact_mt exact_mt)
#target_link_libraries(exact_mt exact_strategy exact_image_tools exact_common ${MYSQL_LIBRARIES} ${TIFF_LIBRARIES} pthread)

add_executable(examm_mt_single_series examm_mt_single_series.cxx)
target_link_libraries(examm_mt_single_series examm_strategy exact_time_series exact_common exact_weights examm_nn pthread)

add_executable(examm_mt examm_mt.cxx)
target_link_libraries(examm_mt examm_strategy exact_time_series exact_common exact_weights examm_nn pthread)

add_executable(snn_rl_mt snn_rl_mt.cxx)
target_link_libraries(snn_rl_mt examm_strategy exact_rl_tools pthread)
if(APPLE)
    target_link_options(snn_rl_mt PRIVATE -Wl,-no_warn_duplicate_libraries)
endif()

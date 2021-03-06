set(cpp_files
  base_types
  block_decoder
  block_encoder
  block_queues
  decoder
  log
  nal_reader
  nal_writer
  packets
  packets_rw
  protobuf_rw
  rng
  uep_decoder
)

foreach(cppfile IN LISTS cpp_files)
  add_library(${cppfile} STATIC ${cppfile}.cpp)
endforeach(cppfile)

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS controlMessage.proto)
#include_directories(${CMAKE_CURRENT_BINARY_DIR})
add_library(controlMessage.pb STATIC ${PROTO_SRCS} ${PROTO_HDRS})
target_link_libraries(controlMessage.pb ${PROTOBUF_LIBRARIES})

target_link_libraries(packets
  base_types
)
target_link_libraries(packets_rw
  packets
)
target_link_libraries(block_queues packets)
target_link_libraries(block_encoder rng packets)
target_link_libraries(block_decoder
  rng
  packets
  log
)
target_link_libraries(decoder
  block_decoder
  block_queues
  ${Boost_LIBRARIES}
)
target_link_libraries(uep_decoder
  decoder
  ${Boost_LIBRARIES}
)
target_link_libraries(log
  ${Boost_LIBRARIES}
)
target_link_libraries(nal_reader
  log
  packets
  ${Boost_LIBRARIES}
)
target_link_libraries(nal_writer
  nal_reader
  ${Boost_LIBRARIES}
)

add_executable(server server.cpp)
target_link_libraries(server
  block_encoder
  block_queues
  controlMessage.pb
  nal_reader
  packets_rw
  protobuf_rw
  ${Boost_LIBRARIES}
)

add_executable(client client.cpp)
target_link_libraries(client
  controlMessage.pb
  decoder
  nal_writer
  packets_rw
  protobuf_rw
  uep_decoder
  ${Boost_LIBRARIES}
)

add_executable(filter_received filter_received.cpp)
target_link_libraries(filter_received
  log
  base_types
  ${Boost_LIBRARIES}
)

add_executable(nal_overhead nal_overhead.cpp)
target_link_libraries(nal_overhead
  ${Boost_LIBRARIES}
  log
  nal_reader
)

add_library(mppy SHARED message_passing_python.cpp)
set_target_properties(mppy PROPERTIES PREFIX "")
target_link_libraries(mppy
  ${PYTHON_LIBRARIES}
)

set(python_files
  avg_nal_segsizes
  ber
  overhead
  plot_ber_iid
  plot_ber_markov2_vs_k
  plot_overhead
  uep_dec_time
  uep_enc_time
)

foreach(pyf IN LISTS python_files)
  add_custom_target(${pyf}.py ALL
    ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_SOURCE_DIR}/${pyf}.py
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${pyf}.py
    COMMENT "Copying ${pyf}.py to ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${pyf}.py"
  )

  add_custom_target(${pyf} ALL
    ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_SOURCE_DIR}/${pyf}.py
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${pyf}
    COMMENT "Copying ${pyf}.py to ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${pyf}"
  )
endforeach(pyf)

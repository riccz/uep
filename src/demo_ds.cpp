#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>

#include "data_client_server.hpp"
#include "encoder.hpp"
#include "log.hpp"

using namespace uep;
using namespace uep::net;

namespace logging = boost::log;
namespace src = boost::log::sources;
namespace sinks = boost::log::sinks;
namespace keywords = boost::log::keywords;

struct data_parameter_set {
  bool enable_ack;
  std::size_t expected_pkt_count;
  double target_send_rate;
  std::size_t max_sequence_number;
  double timeout;
};

struct msg_pkt_src {
  struct parameter_set {
    std::string msg;
    std::size_t max_count;
  };

  std::string msg;
  std::size_t max_count;
  std::size_t count;

  std::vector<packet> original;

  explicit msg_pkt_src(const parameter_set &ps) :
    msg(ps.msg), max_count(ps.max_count), count(0) {};

  packet next_packet() {
    if (count == max_count) throw std::runtime_error("Max packet count");
    else ++count;

    std::stringstream ss;
    ss << msg << " " << std::setfill('0')<< std::setw(5) <<count-1;
    std::string s = ss.str();
    const char *cs = s.c_str();
    packet p;
    p.resize(s.size() + 1);
    for (std::size_t i=0; i < s.size()+1; ++i) {
      p[i] = cs[i];
    }
    original.push_back(p);
    return p;
  }

  explicit operator bool() const {
    return count < max_count;
  }

  bool operator!() const { return !static_cast<bool>(*this); }
};

const std::size_t nblocks = 2, K = 20000;
const msg_pkt_src::parameter_set src_ps{"Test fixed-size packet",
    nblocks*K};
const robust_lt_parameter_set lt_par{K, 0.1, 0.5};
const data_parameter_set data_par{true,
    nblocks*K,
    1000,
    (size_t)(1.2*K),
    60};

int main(int argc, char **argv) {
  using namespace uep::log;

  log::init("demo_ds.log");
  default_logger basic_lg(boost::log::keywords::channel = basic);
  default_logger perf_lg(boost::log::keywords::channel = performance);

  boost::asio::io_service io;

  data_server<lt_encoder<>,msg_pkt_src> ds(io);
  ds.setup_encoder(lt_par);
  ds.setup_source(src_ps);
  ds.enable_ack(data_par.enable_ack);
  ds.target_send_rate(data_par.target_send_rate);
  //ds.max_sequence_number(data_par.max_sequence_number);

  std::string dest_ip;
  std::string dest_port;
  if (argc == 3) {
    dest_ip = argv[1];
    dest_port = argv[2];
  }
  else {
    std::cerr << "Give 2 args" << std::endl;
    return 1;
  }

  ds.open(dest_ip, dest_port);
  auto srv_endp = ds.server_endpoint();

  std::cerr << "Start the client then press enter";
  char c;
  std::cin.read(&c, 1);

  ds.start();

  BOOST_LOG_SEV(basic_lg, debug) << "Start io_service loop";
  io.run();
  BOOST_LOG_SEV(basic_lg, debug) << "io_service loop ended";
  return 0;
}

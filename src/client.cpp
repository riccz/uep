#include <iostream>
#include <vector>

#include <boost/asio.hpp>

#include "controlMessage.pb.h"
#include "data_client_server.hpp"
#include "log.hpp"
#include "uep_decoder.hpp"

using namespace boost::asio;
using namespace std;
using namespace uep::log;
using namespace uep::net;
using namespace uep;

using boost::asio::ip::tcp;
using boost::asio::ip::udp;

struct memory_sink;

typedef uep_decoder dec_type;
typedef memory_sink sink_type;
typedef data_client<dec_type,sink_type> dc_type;
typedef all_parameter_set<dec_type::parameter_set> all_params;

// MEMORY SINK
struct memory_sink {
  typedef all_params parameter_set;
  explicit memory_sink(const parameter_set&) {}

  std::vector<packet> received;

  void push(const packet &p) {
    packet p_copy(p);
    using std::move;
    push(move(p));
  }

  void push(packet &&p) {
    using std::move;
    received.push_back(move(p));
  }

  explicit operator bool() const { return true; }
  bool operator!() const { return false; }
};

const std::string default_udp_port = "12345";
const std::string default_tcp_port = "12312";

int main(int argc, char* argv[]) {
	log::init("client.log");
	default_logger basic_lg(boost::log::keywords::channel = basic);
	default_logger perf_lg(boost::log::keywords::channel = performance);

	boost::asio::io_service io;
	dc_type dc(io);
	all_params ps;
	ps.udp_port_num = default_udp_port;
	ps.tcp_port_num = default_tcp_port;

	if (argc < 3) {
		std::cerr << "Usage: client <stream> <server> [<server_port>] [<listen_port>]"
							<< std::endl;
		return 1;
	}

	ps.streamName = argv[1];
	string host = argv[2];
	if (argc > 3) {
		ps.tcp_port_num = argv[3];
	}
	if (argc > 4) {
		ps.udp_port_num = argv[4];
	}
	if (argc > 5) {
		std::cerr << "Too many args" << std::endl;
		return 1;
	}

	tcp::resolver resolver(io);
	tcp::resolver::query query(host, ps.tcp_port_num);
	tcp::resolver::iterator endpoint_iterator = resolver.resolve(query);

	tcp::socket socket(io);
	boost::asio::connect(socket, endpoint_iterator);

	constexpr std::size_t recv_size = 4096;
	std::string parse_buf;
	std::string serialize_buf;
	boost::system::error_code error;
	std::size_t written;

	controlMessage::StreamName firstMessage;
	firstMessage.set_streamname(ps.streamName);
	if (!firstMessage.SerializeToString(&serialize_buf)) {
		throw std::runtime_error("Serialization of firstMessage failed");
	}
	written = socket.write_some(boost::asio::buffer(serialize_buf));
	std::cout << "Sent " << written << " bytes with stream name: \""<<ps.streamName<<"\"\n";

	parse_buf.resize(recv_size);
	size_t len = socket.read_some(boost::asio::buffer(&parse_buf.front(),
																										parse_buf.size()));
	parse_buf.resize(len);

	//std::cout.write(buf.data(), len);
	controlMessage::TXParam secondMessage;
	if (!secondMessage.ParseFromString(parse_buf)) {
		throw std::runtime_error("Parsing of secondMessage failed");
	}
	std::cout << "Message received from server:\n";

	for (int i=0; i<secondMessage.ks_size(); i++) {
		std::cout << "k_" << i << ":" << secondMessage.ks(i) << ",";
	}
	std::cout << "c="<<secondMessage.c()<<"; ";
	std::cout << "Delta="<<secondMessage.delta()<<"; ";
	for (int i=0; i<secondMessage.rfs_size(); i++) {
		std::cout << "RF_" << i << ":" << secondMessage.rfs(i) << ",";
	}
	std::cout << "EF="<<secondMessage.ef()<<"; ";
	std::cout << "ACK="<<(secondMessage.ack()?"TRUE":"FALSE")<<"; ";
	std::cout << "fileSize="<<secondMessage.filesize()<<";";

	std::vector<std::string> head;
	//header = secondMessage.header();
	std::cout <<  "headerLength="<<secondMessage.header_size() << "\n";
	for (int i=0; i<secondMessage.header_size(); i++) {
		head.push_back(secondMessage.header(i));
		//std::cout << head[i] << ",";
	}

	std::cout << "\nCreation of decoder...\n";
	// CREATION OF DECODER
	assert(ps.Ks.size() == 2);
	ps.Ks[0] = secondMessage.ks(0);
	ps.Ks[1] = secondMessage.ks(1);
	ps.c = secondMessage.c();
	ps.delta = secondMessage.delta();
	ps.RFs[0] = secondMessage.rfs(0);
	ps.RFs[1] = secondMessage.rfs(1);
	ps.EF = secondMessage.ef();

	dc.setup_decoder(ps);
	dc.setup_sink(ps);
	dc.enable_ack(ps.ack);
	dc.bind(ps.udp_port_num);
	std::cout << "Bound to " << dc.client_endpoint() << std::endl;

	controlMessage::Connect connectMessage;
	connectMessage.set_port(dc.client_endpoint().port());
	if (!connectMessage.SerializeToString(&serialize_buf)) {
		throw std::runtime_error("Serialization of connectMessage failed");
	}
	written = socket.write_some(boost::asio::buffer(serialize_buf));
	std::cout << "Connecting now...\n";
	// waiting for ConnACK
	parse_buf.resize(recv_size);
	len = socket.read_some(boost::asio::buffer(&parse_buf.front(),
																						 parse_buf.size()));
	parse_buf.resize(len);

	//std::cout.write(buf.data(), len);
	controlMessage::ConnACK connACKMessage;
	if (!connACKMessage.ParseFromString(parse_buf)) {
		throw std::runtime_error("Parsing of connACKMessage failed");
	}
	std::cout << "Connection ACK message received from server:\n";
	std::cout << "udp port="<<connACKMessage.port()<<"\n";
	ip::address remAddr = socket.remote_endpoint().address();
	udp::endpoint remote_ep(remAddr,
													static_cast<unsigned short>(connACKMessage.port()));
	std::cout << "Start receiving from " << remote_ep << std::endl;
	dc.start_receive(remote_ep);

	std::cout << "Sending PLAY command\n";
	controlMessage::Play playMessage;
	if (!playMessage.SerializeToString(&serialize_buf)) {
		throw std::runtime_error("Serialization of playMessage failed");
	}
	written = socket.write_some(boost::asio::buffer(serialize_buf));
	std::cout << "PLAY command sent.\n";

	std::cout << "Run" << std::endl;
	io.run();
	std::cout << "Done" << std::endl;

	return 0;
}

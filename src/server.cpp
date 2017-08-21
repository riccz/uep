#include <ctime>
#include <fstream>
#include <iostream>
#include <ostream>
#include <string>
<<<<<<< HEAD
#include<boost/algorithm/string.hpp>
#include <sstream>
#include <fstream>
#include <codecvt>

//#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/asio.hpp>
#include "controlMessage.pb.h"
#include <boost/array.hpp>
#include "encoder.hpp"
#include "log.hpp"
#include <ostream>
=======

#include <boost/array.hpp>
#include <boost/asio.hpp>
#include <boost/enable_shared_from_this.hpp>
>>>>>>> 685ba94e9f6719f6e078a2417ac3cc753e23d112
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/shared_ptr.hpp>

#include "controlMessage.pb.h"
#include "data_client_server.hpp"
#include "encoder.hpp"

/*	
	1: client to server: streamName
	2: server to client: TXParam
		2.1 decoder parameters
			2.1.1 K	size_t (unsigned long)
			2.1.2 c (double)
			2.1.3 delta (double)
			2.1.4 RFM (uint8_t)
			2.1.5 RFL (uint8_t)
			2.1.6 EF (uint8_t)
		2.2 ACK enabled
		2.3 File size
	3: client to server: Connect
		3.1 udp port where to send data
		When the server receives it the encoder must be created
	4: server to client: ConnACK
		4.1 udp port where to receive ack
	5: client to server: Play
		DataServer.start
*/

//std::vector<unsigned char> readByteFromFile(const char* filename, int from, int len) {
char * readByteFromFile(std::string filename, int from, int len) {
	std::ifstream ifs(filename, std::ifstream::binary);
	//std::vector<unsigned char> vec;
	if (ifs) {
		ifs.seekg (from, std::ios::beg);
		char * buffer = new char [len];
		ifs.read(buffer, len);
		ifs.close();
		return buffer;
	} else 
		return NULL;	
}


using boost::asio::ip::tcp;
using namespace std;
using namespace uep::net;
<<<<<<< HEAD
using namespace uep::log;
=======
using namespace uep;
>>>>>>> 685ba94e9f6719f6e078a2417ac3cc753e23d112

// DEFAULT PARAMETER SET
struct all_params: public lt_uep_parameter_set, public robust_lt_parameter_set {
	all_params() {
		Ks = {2, 4};
		K = 8;
		robust_lt_parameter_set::c = 0.1;
		robust_lt_parameter_set::delta = 0.01;
		lt_uep_parameter_set::c = 0.1;
		lt_uep_parameter_set::delta = 0.01;
		RFM = 3;
		RFL = 1;
		EF = 2;
	}
	std::string streamName;
	bool ack = true;
	double sendRate = 10240;
	size_t fileSize = 20480;
	int tcp_port_num = 12312;
	/* PARAMETERS CHOICE */ 
	int Ls = 64;
};
all_params ps;


struct streamTrace {
	unsigned int startPos;
	int len;
	int lid;
	int tid;
	int qid;
	int packetType; // 1: StreamHeader, 2: ParameterSet, 3: SliceData
	bool discardable;
	bool truncatable;
};
streamTrace *videoTrace;

streamTrace* loadTrace(std::string streamName) {
	std::ifstream file;
	file = std::ifstream("dataset/"+streamName+".trace", std::ios::in|std::ios::binary|std::ios::ate);
	if (!file.is_open()) throw std::runtime_error("Failed opening file");
	file.seekg (0, std::ios::beg);
	std::string line;
	std::string header;
	//std::regex regex("^0x\\([0-9]+\\)=[0-9]+No$");
	uint16_t lineN = 0;
	uint16_t nRows = 0;
	while (!file.eof()) {
		std::getline(file,line);
		nRows++;
	}
	file.close();
	file = std::ifstream("dataset/"+streamName+".trace", std::ios::in|std::ios::binary|std::ios::ate);
	if (!file.is_open()) throw std::runtime_error("Failed opening file");
	file.seekg (0, std::ios::beg);
	streamTrace saved[nRows-3];
	streamTrace *sTp;
	sTp = saved;
	//std::cout << nRows << std::endl;
	while (!file.eof()) {
		std::getline(file,line);
		//std::cout << line << std::endl;
		if (line.length()==0) {
			break;
		}
		lineN++;
		std::istringstream iss(line);
		if (lineN>2) {
			std::string whiteSpacesTrimmed;
			int n = line.find(" ");
			std::string s = line;
			while (n>=0) {
				if (n==0) {
					s = s.substr(1,s.length());
				} else if (n>0) {
					whiteSpacesTrimmed += s.substr(0,n) + " ";
					s = s.substr(n,s.length());
				}
				n = s.find(" ");
			}
			whiteSpacesTrimmed += s;
			for (int i=0; i<8; i++) {
				int n = whiteSpacesTrimmed.find(" ");
				std::string s = whiteSpacesTrimmed.substr(0,n);
				int nn;
				//std::cout << i << std::endl;
				switch (i) {
				case (0):
					saved[lineN-3].startPos = strtoul(s.substr(s.find("x")+1,s.length()).c_str(), NULL, 16);
					break;
				case (1): case (2): case(3): case (4):
					if (s == "0") {
						nn = 0;
					} else {
						nn = stoi( s );
					}
					break;
				case (5):
					if (s == "StreamHeader") {
						saved[lineN-3].packetType = 1;
					} else if (s == "ParameterSet") {
						saved[lineN-3].packetType = 2;
					} else if (s == "SliceData") {
						saved[lineN-3].packetType = 3;
					}
					break;
				case (6):
					if (s == "Yes") {
						saved[lineN-3].discardable = true;
					} else if (s == "No") {
						saved[lineN-3].discardable = false;
					}
					break;
				case (7):
					if (s == "Yes") {
						saved[lineN-3].truncatable = true;
					} else if (s == "No") {
						saved[lineN-3].truncatable = false;
					}
				break;
				}
				switch (i) {
				case (1):
					saved[lineN-3].len = nn;
					break;
				case (2):
					saved[lineN-3].lid = nn;
					break;
				case (3):
					saved[lineN-3].tid = nn;
					break;
				case (4):
					saved[lineN-3].qid = nn;
					break;
				}
				whiteSpacesTrimmed = whiteSpacesTrimmed.substr(n+1,whiteSpacesTrimmed.length());
			}
		}
	}
	file.close();
	return (sTp);
}

// PACKET SOURCE
char * header;
int headerTo;
struct packet_source {
	typedef all_params parameter_set;
	int Ls;
	int rfm;
	int rfl;
	int ef;
	size_t max_count;
	size_t currInd;
	int rfmReal;
	int rflReal;
	int efReal;
	std::string streamName;
	std::ifstream file;
	
	explicit packet_source(const parameter_set &ps) {
		Ls = ps.Ls;
		rfm = ps.RFM;
		rfl = ps.RFL;
		ef = ps.EF;
		//max_count = ps.fileSize;
		currInd = 0;
		rfmReal = 0;
		rflReal = 0;
		efReal = 0;
		streamName = ps.streamName;
		/* RANDOM GENERATION OF FILE */
		/*
		bool textFile = true;
		int fileSize = max_count; // file size = fileSize * Ls;
		std::ofstream newFile (streamName+".txt");
		//std::cout << streamName << ".txt\n";
		if (newFile.is_open()) { 
			for (int ii = 0; ii<fileSize; ii++) {
				if (!textFile) {
					std::uniform_int_distribution<> dist(0, 255);
					newFile << ((char) dist(gen));
				} else {
					std::uniform_int_distribution<> dist(65, 90);
					int randN = dist(gen);
					newFile << ((char) randN);
				}
			}
		}
		newFile.close();
		*/
		videoTrace = loadTrace(streamName);
		/*
		for (int i=5; i<10; i++) {
			std::cout << videoTrace[i].startPos << "," << videoTrace[i].len << "," << videoTrace[i].lid<< "," << videoTrace[i].tid<< ",";
			std::cout << videoTrace[i].qid << "," << videoTrace[i].packetType << "," << videoTrace[i].discardable<< "," << videoTrace[i].truncatable<< ";\n";
		}
		*/
		// parse .trace to produce a txt with parts repeated
		// first rows of videoTrace are: stream header and parameter set. must be passed through TCP
		int from = videoTrace[0].startPos;
		int to;
		for (to = from; videoTrace[to].packetType < 3; ++to) { }
		int sliceDataInd = to-1;
		to = videoTrace[to-1].startPos + videoTrace[to-1].len;
		header = readByteFromFile("dataset/"+streamName+".264",videoTrace[sliceDataInd].startPos,videoTrace[sliceDataInd].len);
		headerTo = *header + to-from;
		file = std::ifstream("dataset/"+streamName+".264", std::ios::in|std::ios::binary|std::ios::ate);
		if (!file.is_open()) throw std::runtime_error("Failed opening file");
		else max_count = file.tellg();


			//file.seekg (from, std::ios::beg);
			//wchar_t mDataBuffer[to-from]; 
		//std::vector<unsigned char> mDataBuffer;
		//mDataBuffer.resize( to-from );
		//file.read( (char*)( &mDataBuffer[0]), to-from );
			//file.read( (char*)mDataBuffer, to-from );
			//wprintf(L"%s\n", mDataBuffer);
		//std::string videoParams( mDataBuffer.begin(), mDataBuffer.end());
		//std::wstring_convert<std::codecvt_utf8_utf16<char16_t>,char16_t> cv;
		//std::string str8 = cv.to_bytes(videoParams);
		//file.close();
		//header = videoParams;

		//std::bitset<((from-to)*8)> vp (videoParams);
		/*
		std::string videoParams;
		videoParams.resize(to-from);
		
		std::ostringstream out;
		
		//videoParams.reserve(to-from);
		//videoParams.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>(from,to));
		file.read (&videoParams[0], videoParams.size()); */

		/*for (int i=0; i<to-from; i++) {
			std::cout << mDataBuffer[i] << std::endl;
		}
		*/
		//std::cout << videoParams.length();
	}

	packet next_packet() {
		if (currInd >= max_count) throw std::runtime_error("Max packet count");
		if (efReal<ef) {
			if (rfmReal<rfm) rfmReal++;
			else {
				if (rfmReal==rfm) {
					rfmReal++;
					currInd += Ls;
				} 
				if (rflReal<rfl) rflReal++;
				else {
					rfmReal = 0;
					rflReal = 0;
					efReal++;
					currInd -= Ls;
				}
			}	
		} else {
			currInd += 2*Ls;
			rfmReal = 0;
			rflReal = 0;
			efReal = 0;
		}
		file.seekg (currInd, std::ios::beg);
		char * memblock;
		memblock = new char [Ls];
		file.read (memblock, Ls);
		packet p;
		p.resize(Ls);
		p.assign(memblock, Ls+memblock);
		/*for (int i=0; i < Ls; i++) {
			p[i] = memblock[i];
		}*/	
		return p;
	}

	explicit operator bool() const {
		return currInd < max_count;
	}

	bool operator!() const { return !static_cast<bool>(*this); }
	
	std::mt19937 gen;
};

class tcp_connection: public boost::enable_shared_from_this<tcp_connection> {
	/* 	Using shared_ptr and enable_shared_from_this 
		because we want to keep the tcp_connection object alive 
		as long as there is an operation that refers to it.*/
	public:
		typedef boost::shared_ptr<tcp_connection> pointer;
		typedef uep::net::data_server<lt_encoder<std::mt19937>,packet_source> ds_type;
		static pointer create(boost::asio::io_service& io_service) {
			return pointer(new tcp_connection(io_service));
		}

		tcp::socket& socket() {
			return socket_;
		}

		void firstHandler(const boost::system::error_code& error, std::size_t bytes_transferred ) {
			std::string s = std::string(buf.data(), bytes_transferred);
			controlMessage::StreamName firstMessage;
			if (firstMessage.ParseFromString(s)) {
				s = firstMessage.streamname();
			}
			std::cout << "Stream name received from client: \"" << s << "\"\n";
			ps.streamName = s;
			
			/* CREATION OF DATA SERVER */
			//BOOST_LOG_SEV(basic_lg, debug) << "Creation of encoder...\n";
			std::cout << "Creation of encoder...\n";
			ds_type *ds = new ds_type(io);
			ds_p.reset(ds);
			ds->setup_encoder(ps); // setup the encoder inside the data_server
			ds->setup_source(ps); // setup the source  inside the data_server
			ds->target_send_rate(ps.sendRate); // Set a target send rate of 10240 byte/s = 10 pkt/s
			ds->enable_ack(ps.ack);
			
			// SENDING ENCODER PARAMETERS
			controlMessage::TXParam secondMessage;
			secondMessage.set_k(ps.K);
			secondMessage.set_c(ps.robust_lt_parameter_set::c);
			secondMessage.set_delta(ps.robust_lt_parameter_set::delta);
			secondMessage.set_rfm(ps.RFM);
			secondMessage.set_rfl(ps.RFL);
			secondMessage.set_ef(ps.EF);
			secondMessage.set_ack(ps.ack);
			secondMessage.set_filesize(ps.fileSize);
			
			
			uint8_t * head = new uint8_t[headerTo - *header +1];
			//uint16_t * head2 = new uint16_t[(headerTo - *header)/2+1];
			//uint32_t * head4 = new uint32_t[(headerTo - *header)/4+1];

			auto iter2 = header;
			for (int i=0; i<headerTo - *header; i++) {
				head[i] = (uint8_t)*iter2;
				iter2 = iter2 + 1;
			}
			/*
			for (int i=0; i<(headerTo - *header); i+=2) {
				head2[i/2] = head[i] | (head[i+1] << 8);
			}
			for (int i=0; i<(headerTo - *header); i+=4) {
				head4[i/4] = head2[i/2] | (head2[i/2+1] << 16);
			}
			for (int i=0; i<(headerTo - *header); i+=4) {
				secondMessage.add_header(head4[i/4]);
			}
			*/
			uint16_t * head2 = new uint16_t[headerTo - *header + 1];
			std::cout << (headerTo - *header) << std::endl;
			auto iter = header;
			for (int i=0; i<headerTo - *header; i++) {
				head2[i] = *iter;
				iter = iter + 1;
			}

			uint32_t head4 [(headerTo - *header)/2];
			for (int i=0; i<(headerTo - *header); i+=2) {
				head4[i/2] = head2[i] | (head2[i+1] << 16);
				secondMessage.add_header(head4[i/2]);
			}


			std::cout << secondMessage.header_size() << std::endl;
			std::cout << "\n8 bit:\n";
			for (int i=0; i<headerTo - *header; i++)
				std::cout << head[i]<<",";
			std::cout << "\n16 bit:\n";
			for (int i=0; i<headerTo - *header; i+=2)
				std::cout << head2[i/2]<<",";
				//std::cout << header[i] << std::endl;
			std::cout << "\n32 bit:\n";
			for (int i=0; i<(headerTo - *header); i+=4) 
				std::cout << head4[i/4]<<",";
			
			if (secondMessage.SerializeToString(&s)) {
				std::cout << "Sending encoder's parameters to client...\n";
				/*	Call boost::asio::async_write() to serve the data to the client. 
					We are using boost::asio::async_write(), 
					rather than ip::tcp::socket::async_write_some(), 
					to ensure that the entire block of data is sent.*/
				boost::asio::async_write(socket_, boost::asio::buffer(s),std::bind(
					&tcp_connection::handle_write,
					shared_from_this(),
					std::placeholders::_1,
					std::placeholders::_2));
				// waiting for Connection request, then secondHandler
				boost::system::error_code error;
				socket_.async_read_some(boost::asio::buffer(buf), std::bind(
					&tcp_connection::secondHandler,
					shared_from_this(),
					std::placeholders::_1, 
					std::placeholders::_2));
				if (error == boost::asio::error::eof)
					std::cout.write("error",5); // Connection closed cleanly by peer.
				else if (error)
					throw boost::system::system_error(error); // Some other error.
			}			
		}

		void secondHandler(const boost::system::error_code& error, std::size_t bytes_transferred ) {
			//	data received stored in string s
			std::string s = std::string(buf.data(), bytes_transferred);
			//	converting received data to udp port number of the client
			controlMessage::Connect connectMessage;
			uint32_t port;
			if (connectMessage.ParseFromString(s)) {
				port = connectMessage.port();
			}
			else throw std::runtime_error("No port");
			std::cout << "Connect req. received from client on port " << port << "\n";
			// Opening UDP connection
			char portStr[10];
			sprintf(portStr, "%u", port);
			/*
			asio::ip::address remote_ad = socket_.remote_endpoint().address();
			std::string s = remote_ad.to_string();
			*/
			// GET REMOTE ADDRESS FROM TCP CONNECTION
			std::string remAddr = socket_.remote_endpoint().address().to_string();
			std::cout << "Binding of data server to "<<remAddr<<":"<<port<<"\n";
			ds_p->open(remAddr, portStr);
			//boost::asio::ip::udp::endpoint udpEndpoint = ds_p->server_endpoint();
			uint32_t udpPort = ds_p->server_endpoint().port(); // =(udp port for ACK to server)
			std::cout << "UDP port for ACKs: "<< udpPort << std::endl;
			std::cout << "Sending to client..." << std::endl;
			controlMessage::ConnACK connACKMessage;
			connACKMessage.set_port(udpPort);
			if (connACKMessage.SerializeToString(&s)) {
				boost::asio::async_write(socket_, boost::asio::buffer(s),std::bind(
					&tcp_connection::handle_write,
					shared_from_this(),
					std::placeholders::_1,
					std::placeholders::_2));
				std::cout << "Connection ACK sent...\n";
				// waiting for play command
				boost::system::error_code error;
				socket_.async_read_some(boost::asio::buffer(buf), std::bind(
					&tcp_connection::thirdHandler,
					shared_from_this(),
					std::placeholders::_1,
					std::placeholders::_2));
			}
		}

		void thirdHandler(const boost::system::error_code& error, std::size_t bytes_transferred ) {
			std::string s = std::string(buf.data(), bytes_transferred);
			// s should be empty
			std::cout << "PLAY.\n";
			ds_p->start();
			std::cout << "Called start" << std::endl;
		}

		void start() {	
			boost::system::error_code error;
			socket_.async_read_some(boost::asio::buffer(buf), std::bind(
				&tcp_connection::firstHandler,
				shared_from_this(),
				std::placeholders::_1, std::placeholders::_2));

			if (error == boost::asio::error::eof)
				std::cout.write("error",5); // Connection closed cleanly by peer.
			else if (error)
				throw boost::system::system_error(error); // Some other error.
			//std::cout << buf.data() << std::endl;

		}

	private:
		boost::array<char,128> buf;
		std::unique_ptr<ds_type> ds_p;
		tcp::socket socket_;
		boost::asio::io_service& io;
		
		//uep::lt_encoder<std::mt19937> enc;
		//uep::net::data_server<lt_encoder<std::mt19937>,random_packet_source> ds;
		//boost::asio::io_service io;
		tcp_connection(boost::asio::io_service& io_service): socket_(io_service), io(io_service) {
			
		}
		// handle_write() is responsible for any further actions 
		// for this client connection.
		void handle_write(const boost::system::error_code& /*error*/,size_t /*bytes_transferred*/) {

		}

};

class tcp_server {
	public:
		// Constructor: initialises an acceptor to listen on TCP port tcp_port_num.
		tcp_server(boost::asio::io_service& io_service): 
			acceptor_(io_service, tcp::endpoint(tcp::v4(), ps.tcp_port_num)) {
			// start_accept() creates a socket and 
			// initiates an asynchronous accept operation 
			// to wait for a new connection.
			start_accept();
		}

	private:
		void start_accept() {
			// creates a socket
			tcp_connection::pointer new_connection =
				tcp_connection::create(acceptor_.get_io_service());
			active_conns.push_front(new_connection);

			// initiates an asynchronous accept operation 
			// to wait for a new connection. 
			acceptor_.async_accept(new_connection->socket(),
				std::bind(&tcp_server::handle_accept, this, new_connection,std::placeholders::_1));
		}

		// handle_accept() is called when the asynchronous accept operation 
		// initiated by start_accept() finishes. It services the client request
		void handle_accept(tcp_connection::pointer new_connection,const boost::system::error_code& error) {
			if (!error) {
				new_connection->start();
			}
			// Call start_accept() to initiate the next accept operation.
			start_accept();
	
		}

		tcp::acceptor acceptor_;

	std::list<boost::shared_ptr<tcp_connection>> active_conns;
};

int main() {
	/*
	log::init("demo_ds.log");
	default_logger basic_lg(boost::log::keywords::channel = basic);
	default_logger perf_lg(boost::log::keywords::channel = performance);
	*/
	try {
		// We need to create a server object to accept incoming client connections.
		boost::asio::io_service io_service;

		// The io_service object provides I/O services, such as sockets, 
		// that the server object will use.
		tcp_server server(io_service);

		// Run the io_service object to perform asynchronous operations.
		io_service.run();
	} catch (std::exception& e) {
		std::cerr << e.what() << std::endl;
	}
	return 0;
}

// Local Variables:
// tab-width: 2
// End:

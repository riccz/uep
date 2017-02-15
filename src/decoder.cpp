#include "decoder.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

using namespace std;

fountain_decoder::fountain_decoder(const degree_distribution &distr) :
  fountain_decoder(fountain(distr)) {
}

fountain_decoder::fountain_decoder(const fountain &f) :
  fount(f), blockno_(0) {
}

void fountain_decoder::push_coded(fountain_packet &&p) {
  if (blockno_ == 0 && received_pkts.empty()) { // First packet
    block_seed_ = p.block_seed();
    fount.reset(block_seed_);
    PUT_STAT_COUNTER(loggers.newblock);
    BOOST_LOG_SEV(loggers.text, debug) << "Start to receive the first block: " <<
      "blockno=" << blockno_ << ", seed=" << block_seed_;
  }
  else if (p.block_number() != blockno_) { // Packet from another block
    int dist;
    if (p.block_number() > blockno_)
      dist = p.block_number() - blockno_;
    else if (p.block_number() < blockno_)
      dist = MAX_BLOCKNO - blockno_ + p.block_number() + 1;

    if (dist > BLOCK_WINDOW) { // Old block
      PUT_STAT_COUNTER(loggers.old_dropped);
      BOOST_LOG_SEV(loggers.text, info) << "Drop packet for an old block: " << p;
      return;
    }

    // New block
    blockno_ = p.block_number();
    block_seed_ = p.block_seed();

    received_pkts.clear();
    original_connections.clear();
    fount.reset(block_seed_);
    decoded.clear();
    bg.clear();

    PUT_STAT_COUNTER(loggers.newblock);
    BOOST_LOG_SEV(loggers.text, debug) << "Start to receive the next block: " <<
      "blockno=" << blockno_ << ", seed=" << block_seed_;
  }

  int p_seqno = p.sequence_number();
  PUT_STAT_COUNTER(loggers.in_pkts);
  BOOST_LOG_SEV(loggers.text, debug) << "Received an encoded packet: " << p;
  auto ins_res = received_pkts.insert(move(p));
  if (!ins_res.second) {
    PUT_STAT_COUNTER(loggers.dup_pkts);
    BOOST_LOG_SEV(loggers.text, info) << "Drop a duplicate packet: " << p;
    return;
  }
  while ((size_t)p_seqno >= original_connections.size()) {
    fountain::row_type row = fount.next_row();
    original_connections.push_back(row);
    PUT_STAT_SCALAR(loggers.rows, row.size());
  }

  if (/*received_pkts.size() >= (size_t)K() &&*/ !has_decoded()) {
    run_message_passing();
  }
}

void fountain_decoder::push_coded(const fountain_packet &p) {
  fountain_packet p_copy(p);
  push_coded(move(p_copy));
}

std::vector<packet>::const_iterator fountain_decoder::decoded_begin() const {
  return decoded.cbegin();
}

std::vector<packet>::const_iterator fountain_decoder::decoded_end() const {
  return decoded.cend();
}

bool fountain_decoder::has_decoded() const {
  return decoded.size() == (size_t)K();
}

int fountain_decoder::K() const {
  return fount.K();
}

int fountain_decoder::blockno() const {
  return blockno_;
}

int fountain_decoder::block_seed() const {
  return block_seed_;
}

size_t fountain_decoder::received_count() const {
  return received_pkts.size();
}

fountain fountain_decoder::the_fountain() const {
  return fount;
}

std::set<fountain_packet>::const_iterator
fountain_decoder::received_packets_begin() const {
  return received_pkts.cbegin();
}

std::set<fountain_packet>::const_iterator
fountain_decoder::received_packets_end() const {
    return received_pkts.cend();
}

void fountain_decoder::init_bg() {
  bg.clear();
  bg_decoded_count = 0;
  bg.resize_input(K());
  bg.resize_output(received_pkts.size());
  for (auto i = received_pkts.cbegin(); i != received_pkts.cend(); ++i) {
    int i_seqno = i->sequence_number();
    if ((size_t)i_seqno >= bg.output_size()) {
      bg.resize_output(i_seqno + 1);
    }
    bg.output_at(i_seqno) = *i;
    const fountain::row_type &conns = original_connections[i_seqno];
    bg.add_links_to_output(i_seqno, conns.cbegin(), conns.cend());
  }
}

void fountain_decoder::decode_degree_one(std::set<bg_size_type> &ripple) {
  ripple.clear();

  std::vector<std::pair<bg_size_type, bg_size_type> > delete_conns;
  for (auto i = bg.output_degree_one_begin();
       i != bg.output_degree_one_end();
       ++i) {
    bg_size_type out = *i;
    bg_size_type in = bg.find_input(out);
    if (!bg.output_at(out).empty()) {
      auto ins_ret = ripple.insert(in);
      if (ins_ret.second) {
	swap(bg.input_at(in), bg.output_at(out));
	++bg_decoded_count;
      }
    }
    delete_conns.push_back(std::pair<bg_size_type, bg_size_type>(in, out));
  }

  for (auto i = delete_conns.begin(); i != delete_conns.end(); ++i) {
    bg.remove_link(i->first, i->second);
  }
}

void fountain_decoder::run_message_passing() {
  PUT_STAT_SCALAR(loggers.recv_size, received_pkts.size());
  BOOST_LOG_SEV(loggers.text, debug) << "Run message passing with " <<
    received_pkts.size() << " received pkts";

  init_bg();

  std::set<bg_size_type> ripple;
  for (;;) {
    decode_degree_one(ripple);
    BOOST_LOG_SEV(loggers.text, debug) << "The ripple has size " << ripple.size();
    if (bg_decoded_count == K() || ripple.empty()) break;
    process_ripple(ripple);
  }

  PUT_STAT_SCALAR(loggers.decodeable, bg_decoded_count);

  if (bg_decoded_count == K()) {
    decoded.resize(K());
    copy(bg.input_begin(), bg.input_end(), decoded.begin());
    PUT_STAT_COUNTER(loggers.dec_blocks);
    BOOST_LOG_SEV(loggers.text, debug) << "Decoding complete";
  }
}

void fountain_decoder::process_ripple(const std::set<bg_size_type> &ripple) {
  for (auto i = ripple.cbegin(); i != ripple.cend(); ++i) {
    bg_size_type in = *i;
    const fountain_packet &in_p = bg.input_at(in);
    std::set<std::pair<bg_size_type, bg_size_type> > delete_conns;
    auto out_range = bg.find_all_outputs(in);
    for (; out_range.first != out_range.second; ++out_range.first) {
      bg_size_type out = *(out_range.first);
      fountain_packet &out_p = bg.output_at(out);
      if (!out_p.empty()) out_p ^= in_p;
      delete_conns.insert(std::pair<bg_size_type, bg_size_type>(in, out));
    }

    for (auto i = delete_conns.begin(); i != delete_conns.end(); ++i) {
      bg.remove_link(i->first, i->second);
    }
  }
}

#include "block_decoder.hpp"

using namespace std;

namespace uep {

block_decoder::block_decoder(const lt_row_generator &rg) :
  rowgen(rg) {
  link_cache.reserve(rg.K());
}

void block_decoder::check_correct_block(const fountain_packet &p) {
  if (received_pkts.empty()) {
    blockno = p.block_number();
    rowgen.reset(p.block_seed());
    pktsize = p.size();
  }
  else if (blockno != p.block_number() ||
	   seed() != (seed_t)p.block_seed()) {
    throw std::runtime_error("All packets must belong to the same block");
  }
  else if (pktsize != p.size()) {
    throw std::runtime_error("All packets must have the same size");
  }
}

bool block_decoder::push(fountain_packet &&p) {
  check_correct_block(p);
  size_t p_seqno = p.sequence_number();
  auto ins_ret = received_pkts.insert(move(p));
  if (!ins_ret.second) return false;

  if (link_cache.size() <= p_seqno) {
    size_t prev_size = link_cache.size();
    link_cache.resize(p_seqno+1);
    for (size_t i = prev_size; i < p_seqno+1; ++i) {
      link_cache[i] = rowgen.next_row();
    }
  }

  if (!has_decoded() &&
      (received_pkts.size() >= rowgen.K()/* || do_partial_decoding*/)) {
    run_message_passing();
  }

  return true;
}

bool block_decoder::push(const fountain_packet &p) {
  fountain_packet p_copy(p);
  return push(move(p_copy));
}

void block_decoder::reset() {
  rowgen.reset();
  received_pkts.clear();
  link_cache.clear();
  mp_ctx.clear();
}

block_decoder::seed_t block_decoder::seed() const {
  return rowgen.seed();
}

int block_decoder::block_number() const {
  return blockno;
}

bool block_decoder::has_decoded() const {
  return mp_ctx.decoded_count() == rowgen.K();
}

std::size_t block_decoder::decoded_count() const {
  return mp_ctx.decoded_count();
}

block_decoder::const_block_iterator block_decoder::block_begin() const {
  return const_block_iterator(mp_ctx.decoded_symbols_begin(), lazy2p_conv());
}

block_decoder::const_block_iterator block_decoder::block_end() const {
  return const_block_iterator(mp_ctx.decoded_symbols_end(), lazy2p_conv());
}

block_decoder::operator bool() const {
  return has_decoded();
}

bool block_decoder::operator!() const {
  return !has_decoded();
}

void block_decoder::run_message_passing() {
  // Setup the context
  typedef boost::transform_iterator<fp2lazy_conv,
				    received_t::const_iterator
				    > mp_ctx_out_iter;
  mp_ctx_out_iter out_b(received_pkts.cbegin(), fp2lazy_conv());
  mp_ctx_out_iter out_e(received_pkts.cend(), fp2lazy_conv());
  mp_ctx = mp_ctx_t(rowgen.K(), out_b, out_e);

  for (auto j = received_pkts.cbegin(); j != received_pkts.cend(); ++j) {
    const lt_row_generator::row_type &row = link_cache[j->sequence_number()];
    for (auto i = row.cbegin(); i != row.cend(); ++i) {
      mp_ctx.add_edge(*i, j->sequence_number());
    }
  }

  // Run mp
  mp_ctx.run();
}

}

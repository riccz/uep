#include "block_decoder.hpp"

#include <chrono>
#include <iterator>
#include <stdexcept>

using namespace std;
using namespace std::chrono;

namespace uep {

block_decoder::block_decoder(const lt_row_generator &rg) :
  block_decoder(std::make_unique<lt_row_generator>(rg)) {
}

block_decoder::block_decoder(std::unique_ptr<base_row_generator> &&rg) :
  basic_lg(boost::log::keywords::channel = log::basic),
  perf_lg(boost::log::keywords::channel = log::performance),
  rowgen(std::move(rg)),
  mp_ctx(rowgen->K()),
  mp_pristine(rowgen->K()) {
  link_cache.reserve(rowgen->K());
}

void block_decoder::check_correct_block(const fountain_packet &p) {
  // First packet: set blockno, length, seed
  if (received_seqnos.empty()) {
    blockno = p.block_number();
    rowgen->reset(p.block_seed());
    pktsize = p.size();
  }
  // Other packets: check blockno, seed
  else if (blockno != static_cast<size_t>(p.block_number()) ||
	   seed() != static_cast<seed_t>(p.block_seed())) {
    throw std::runtime_error("All packets must belong to the same block");
  }
  // Check length
  else if (pktsize != p.size()) {
    throw std::runtime_error("All packets must have the same size");
  }
}

bool block_decoder::push(fountain_packet &&p) {
  // Make a size-one iter range with p, *mv_ptr should be an rvalue
  auto mv_ptr = std::make_move_iterator(&p);
  auto mv_end = std::make_move_iterator(&p + 1);
  std::size_t n = push(mv_ptr, mv_end);
  return n == 1;
}

bool block_decoder::push(const fountain_packet &p) {
  fountain_packet p_copy(p);
  return push(move(p_copy));
}

void block_decoder::reset() {
  rowgen->reset();
  received_seqnos.clear();
  link_cache.clear();
  last_received.clear();
  mp_ctx.reset();
  mp_pristine.reset();
  avg_mp.reset();
  avg_setup.reset();
}

block_decoder::seed_t block_decoder::seed() const {
  if (received_seqnos.empty()) throw std::runtime_error("No received packets");
  return rowgen->seed();
}

std::size_t block_decoder::block_number() const {
  if (received_seqnos.empty()) throw std::runtime_error("No received packets");
  return blockno;
}

bool block_decoder::has_decoded() const {
  return mp_ctx.has_decoded();
}

std::size_t block_decoder::decoded_count() const {
  return mp_ctx.decoded_count();
}

std::size_t block_decoder::received_count() const {
  return mp_ctx.output_size();
}

std::size_t block_decoder::block_size() const {
  return rowgen->K();
}

block_decoder::const_block_iterator block_decoder::block_begin() const {
  return const_block_iterator(mp_ctx.decoded_symbols_begin(),
			      lazy2p_conv<LX_MAX_SIZE>());
}

block_decoder::const_block_iterator block_decoder::block_end() const {
  return const_block_iterator(mp_ctx.decoded_symbols_end(),
			      lazy2p_conv<LX_MAX_SIZE>());
}

block_decoder::const_partial_iterator block_decoder::partial_begin() const {
  return const_partial_iterator(mp_ctx.input_symbols_begin(),
				lazy2p_conv<LX_MAX_SIZE>());
}

block_decoder::const_partial_iterator block_decoder::partial_end() const {
  return const_partial_iterator(mp_ctx.input_symbols_end(),
				lazy2p_conv<LX_MAX_SIZE>());
}

double block_decoder::average_message_passing_time() const {
  return avg_mp.value();
}

double block_decoder::average_mp_setup_time() const {
  return avg_setup.value();
}

block_decoder::operator bool() const {
  return has_decoded();
}

bool block_decoder::operator!() const {
  return !has_decoded();
}

void block_decoder::run_message_passing() {
  auto tic = high_resolution_clock::now();

  for (auto i = last_received.cbegin(); i != last_received.cend(); ++i) {
    // Update the context
    const base_row_generator::row_type &row = link_cache[i->sequence_number()];
    mp_pristine.add_output(sym_t(std::move(i->buffer())), row.cbegin(), row.cend());
  }
  last_received.clear();

  // Run on a copy
  mp_ctx = mp_pristine;

  duration<double> mp_tdiff = high_resolution_clock::now() - tic;

  BOOST_LOG(perf_lg) << "block_decoder::run_message_passing mp_setup_time="
		     << mp_tdiff.count();

  mp_ctx.run();

  avg_setup.add_sample(mp_tdiff.count());
  avg_mp.add_sample(mp_ctx.run_duration());

  BOOST_LOG(perf_lg) << "block_decoder::run_message_passing decoded_pkts="
		     << mp_ctx.decoded_count()
		     << " received_pkts="
		     << mp_ctx.output_size();
}

const base_row_generator &block_decoder::row_generator() const {
  return *rowgen;
}

}

#include "encoder.hpp"

#include <stdexcept>
#include <limits>

using namespace std;

fountain_encoder::fountain_encoder(const degree_distribution &distr) :
  fountain_encoder(fountain(distr)) {
}

fountain_encoder::fountain_encoder(const fountain &f) :
  fount(f), blockno_(0), seqno_(0) {
  input_block.reserve(f.K());
  block_seed_ = next_seed();
  fount.reset(block_seed_); 
}

void fountain_encoder::push_input(packet &&p) {
  input_queue.push(move(p));
  check_has_block();
}

void fountain_encoder::push_input(const packet &p) {
  packet p_copy(p);
  push_input(move(p_copy));
}

fountain_packet fountain_encoder::next_coded() {
  if (!has_block())
    throw runtime_error("Does not have a full block");
  auto sel = fount.next_row(); // genera riga
  auto i = sel.begin(); // iteratore sulla riga
  fountain_packet first(input_block[*i]); // mette dentro first una copia dell'input
  i++;
  for (; i != sel.end(); i++) { // begin == end è la condizione di fine ciclo
    first ^= input_block[*i];
  }
  first.block_number(blockno_);
  first.block_seed(block_seed_);
  if (seqno_ == numeric_limits<uint_fast16_t>::max()) // seqno: = prossimo seq. number del pacchetto da inviare
    throw new runtime_error("Seqno overflow");
  first.sequence_number(seqno_++);
  return first;
}

void fountain_encoder::discard_block() {
  input_block.clear();
  block_seed_ = next_seed();
  fount.reset(block_seed_);
  if (blockno_ == numeric_limits<uint_fast16_t>::max())
    throw runtime_error("Block number overflow");
  blockno_++;
  seqno_ = 0;
  check_has_block();
}

bool fountain_encoder::has_block() const {
  return input_block.size() == K();
}

std::uint_fast16_t fountain_encoder::K() const {
  return fount.K();
}

std::uint_fast16_t fountain_encoder::blockno() const {
  return blockno_;
}

std::uint_fast16_t fountain_encoder::seqno() const {
  return seqno_;
}

std::uint_fast32_t fountain_encoder::block_seed() const {
  return block_seed_;
}

fountain fountain_encoder::the_fountain() const {
  return fount;
}

std::vector<packet>::const_iterator
fountain_encoder::current_block_begin() const {
  return input_block.cbegin();
}

std::vector<packet>::const_iterator
fountain_encoder::current_block_end() const {
  return input_block.cend();
}

void fountain_encoder::check_has_block() {
  if (!has_block() && input_queue.size() >= K()) {
    for (uint_fast16_t i = 0; i < K(); i++) {
      packet p;
      swap(p, input_queue.front());
      input_queue.pop();
      input_block.push_back(move(p));
    }
  }
}

std::uint_fast32_t fountain_encoder::next_seed() {
  return rd();
  //return mt19937::default_seed;
}

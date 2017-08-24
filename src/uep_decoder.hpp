#ifndef UEP_UEP_DECODER_HPP
#define UEP_UEP_DECODER_HPP

#include "decoder.hpp"

#include <boost/iterator/iterator_adaptor.hpp>

namespace uep {

/** Unequal error protection decoder.
 *  This class wraps an lt_decoder and deduplicates the packets of the
 *  expanded blocks. The output blocks are enqueued in a FIFO queue
 *  following the structure defined by the Ks, RFs and EF parameters.
 */
class uep_decoder {
public:
  /** The collection of parameters required to setup the decoder. */
  typedef lt_uep_parameter_set parameter_set;

  static constexpr std::size_t MAX_BLOCKNO = lt_decoder::MAX_BLOCKNO;
  static constexpr std::size_t BLOCK_WINDOW = lt_decoder::BLOCK_WINDOW;

  explicit uep_decoder(const parameter_set &ps);

  /** Pass a received packet. \sa push(fountain_packet&&) */
  void push(const fountain_packet &p);
  /** Pass a received packet.
   *  This method causes the message-passing algorithm to be run on
   *  the accumulated set of packets, unless the current block is
   *  already fully decoded. Duplicate and old packets are silently
   *  discarded.  The decoder switches to a new block if a more recent
   *  block number is received.
   */
  void push(fountain_packet &&p);

  /** Pass many received packets at once. The packets can be reordered
   *  before being decoded. To move from the packets the iterators can
   *  be wrapped in std::move_iterator.
   */
  template <class Iter>
  void push(Iter first, Iter last);

  /** Extract the oldest decoded packet from the FIFO queue. The
   *  original priority level is set.
   */
  fountain_packet next_decoded();

  /** Const iterator pointing to the start of the last decoded block.
   *  This can become invalid after a call to push().
   */
  //const_block_iterator decoded_begin() const;
  /** Const iterator pointing to the end of the last decoded block.
   *  This can become invalid after a call to push().
   */
  //const_block_iterator decoded_end() const;

  /** Push the current incomplete block to the queue and wait for the
   *  next block.
   */
  void flush();

  /** Push the current incomplete block to the queue, assume all
   *  blocks are failed up to the given one and wait for packets
   *  belonging to the given block.
   */
  void flush(std::size_t blockno_);

  /** Push the current incomplete block and `n-1` additional empty
   *  blocks to the queue
   */
  void flush_n_blocks(std::size_t n);

  /** Return true if the current block has been decoded. */
  bool has_decoded() const;
  /** Return the output block size. */
  std::size_t block_size() const;
  /** Return the input block size. */
  std::size_t block_size_in() const;
  /** Alias for block_size(). */
  std::size_t block_size_out() const;
  /** Alias for block_size(). */
  std::size_t K() const;
  /** Return the current block number. */
  std::size_t blockno() const;
  /** Return a copy of the current block number counter. */
  circular_counter<std::size_t> block_number_counter() const;
  /** Return the current block seed. */
  int block_seed() const;
  /** Number of unique packets received for the current block. */
  std::size_t received_count() const;
  /** Number of packets decoded for the current block. */
  //std::size_t decoded_count() const;
  /** Number of output queued packets. */
  std::size_t queue_size() const;
  /** True if there are decoded packets still in the queue. */
  bool has_queued_packets() const;
  /** Return the total number of unique received packets. */
  std::size_t total_received_count() const;
  /** Return the total number of packets that were decoded and passed
   *  to the queue. This excludes the packets decoded in the current
   *  block.
   */
  std::size_t total_decoded_count() const;
  /** Return the total number of failed packets that were passed to
   *  the queue. This does not count the undecoded packets in the
   *  current block.
   */
  std::size_t total_failed_count() const;

  /** Return the average time to push a packet. */
  double average_push_time() const;

  /** True if there are decoded packets still in the queue. */
  explicit operator bool() const;
  /** True if all the decoded packets have been extracted. */
  bool operator!() const;

private:
  log::default_logger basic_lg, perf_lg;

  std::vector<std::size_t> Ks; /**< Array of sub-block sizes. */
  std::vector<std::size_t> RFs; /**< Array of repetition factors. */
  std::size_t EF; /**< Expansion factor. */

  std::queue<fountain_packet> out_queue; /**< Hold the deduplicated
					    packets with their
					    original priority
					    level. */
  std::unique_ptr<lt_decoder> std_dec; /**< The standard LT
					*    decoder. Use a pointer to
					*    delay the construction.
					*/

  std::size_t tot_dec_count;
  std::size_t tot_fail_count;

  /** Check if there are new decoded blocks and deduplicate them. */
  void deduplicate_queued();
  /** Map an index in the expanded block to an (index, priority) in
   *  the original block.
   */
  std::pair<std::size_t,std::size_t> map_in2out(std::size_t i);
};

//		   uep_decoder template definitions

template <class Iter>
void uep_decoder::push(Iter first, Iter last) {
  std_dec->push(first, last);
  deduplicate_queued();
}

}

#endif

#ifndef UEP_UEP_DECODER_HPP
#define UEP_UEP_DECODER_HPP

#include <limits>

#include <boost/iterator/iterator_adaptor.hpp>

#include "decoder.hpp"

namespace uep {

/** Unequal error protection decoder.
 *  This class wraps an lt_decoder and deduplicates the packets of the
 *  expanded blocks. The output blocks are enqueued in a FIFO queue
 *  following the structure defined by the Ks, RFs and EF parameters.
 */
class uep_decoder {
  typedef std::queue<uep_packet> queue_type;

public:
  /** The collection of parameters required to setup the decoder. */
  typedef lt_uep_parameter_set parameter_set;

  static constexpr std::size_t BLOCK_WINDOW = lt_decoder::BLOCK_WINDOW;
  static constexpr std::size_t MAX_BLOCKNO = lt_decoder::MAX_BLOCKNO;

  explicit uep_decoder(const parameter_set &ps);
  template<typename KsIter, typename RFsIter>
  explicit uep_decoder(KsIter ks_begin, KsIter ks_end,
		       RFsIter rfs_begin, RFsIter rfs_end,
		       std::size_t EF,
		       double c,
		       double delta);

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
  /** Total number of padding packets successfully decoded. */
  std::size_t total_padding_count() const;

  /** Return the average time to push a packet. */
  double average_push_time() const;

  /** True if there are decoded packets still in the queue. */
  explicit operator bool() const;
  /** True if all the decoded packets have been extracted. */
  bool operator!() const;

  const uep_row_generator &row_generator() const;

private:
  log::default_logger basic_lg, perf_lg;

  std::vector<queue_type> out_queues; /**< Hold the deduplicated
					 packets with their
					 original priority
					 level. */
  std::size_t empty_queued_count; /**< Count separately the empty
				   *   packets: their aeqno is lost.
				   */
  circular_counter<> seqno_ctr; /**< Counter for the sequence number
				 *   of the UEP packets.
				 */
  std::unique_ptr<lt_decoder> std_dec; /**< The standard LT
					*    decoder. Use a pointer to
					*    delay the construction.
					*/

  std::size_t tot_dec_count;
  std::size_t tot_fail_count;
  stat::sum_counter<std::size_t> padding_cnt; /**< Counter for the
					       *   padding packets
					       *   decoded.
					       */
  stat::average_counter _avg_dec_time; /**< Keep the average of the
					*   push time.
					*/

  /** Check if there are new decoded blocks and deduplicate them. */
  void deduplicate_queued();
  /** Find the queue with the given seqno on top. */
  std::vector<queue_type>::iterator
  find_decoded(std::size_t seqno);
};

	      //// uep_decoder template definitions ////

template<typename KsIter, typename RFsIter>
uep_decoder::uep_decoder(KsIter ks_begin, KsIter ks_end,
			 RFsIter rfs_begin, RFsIter rfs_end,
			 std::size_t ef,
			 double c,
			 double delta) :
  basic_lg(boost::log::keywords::channel = log::basic),
  perf_lg(boost::log::keywords::channel = log::performance),
  empty_queued_count(0),
  seqno_ctr(std::numeric_limits<uep_packet::seqno_type>::max()),
  tot_dec_count(0),
  tot_fail_count(0) {
  auto uep_rowgen = std::make_unique<uep_row_generator>(ks_begin, ks_end,
							rfs_begin, rfs_end,
							ef,
							c,
							delta);
  out_queues.resize(uep_rowgen->Ks().size());

  std_dec = std::make_unique<lt_decoder>(std::move(uep_rowgen));

  seqno_ctr.set(0);

  BOOST_LOG_SEV(basic_lg, log::debug) << "Constructed a uep_decoder."
				      << " Ks=" << row_generator().Ks()
				      << " RFs=" << row_generator().RFs()
				      << " EF=" << row_generator().EF()
				      << " c=" << row_generator().c()
				      << " delta=" << row_generator().delta();
}

template <class Iter>
void uep_decoder::push(Iter first, Iter last) {
  using namespace std::chrono;

  auto tic = high_resolution_clock::now();

  std_dec->push(first, last);
  deduplicate_queued();

  duration<double> push_tdiff = high_resolution_clock::now() - tic;
  _avg_dec_time.add_sample(push_tdiff.count());
  BOOST_LOG(perf_lg) << "uep_decoder::push push_time="
		     << push_tdiff.count();
}

}

#endif

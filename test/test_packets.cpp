#define BOOST_TEST_MODULE packets_test
#include <boost/test/unit_test.hpp>

#include "packets.hpp"
#include <boost/numeric/conversion/cast.hpp>

using namespace std;
using namespace uep;

BOOST_AUTO_TEST_CASE(packet_shallow_copy) {
  packet p(5, 0x11);
  packet q(p);
  q[0] = 0x22;
  BOOST_CHECK(q != p);
  packet r = p;
  r[1] = 0x33;
  BOOST_CHECK(r != p);

  BOOST_CHECK_EQUAL(p.shared_count(), 1);
  packet pp = p.shallow_copy();
  BOOST_CHECK_EQUAL(p.shared_count(), 2);
  pp[2] = 0x44;
  BOOST_CHECK(pp == p);
}

BOOST_AUTO_TEST_CASE(fountain_packet_shallow_copy) {
  fountain_packet p(1,2,3, 5, 0x11);
  fountain_packet q(p);
  q[0] = 0x22;
  BOOST_CHECK(q != p);
  fountain_packet r = p;
  r[1] = 0x33;
  BOOST_CHECK(r != p);

  BOOST_CHECK_EQUAL(p.shared_count(), 1);
  fountain_packet pp = p.shallow_copy();
  BOOST_CHECK_EQUAL(p.shared_count(), 2);

  BOOST_CHECK_EQUAL(pp.block_number(), 1);
  pp[2] = 0x44;
  BOOST_CHECK(pp == p);

  q = p;
  q.block_seed(42);
  BOOST_CHECK(q != p);
}

BOOST_AUTO_TEST_CASE(packet_xor) {
  packet p(10, 0x11);
  packet q(10, 0x22);
  packet expected(10, 0x33);

  BOOST_CHECK((p ^ q) == expected);
}

BOOST_AUTO_TEST_CASE(packet_wrong_xor) {
  packet p(10, 0x11);
  packet q(9, 0x22);
  fountain_packet r(11, 0x33);
  BOOST_CHECK_THROW(p ^ q, runtime_error);
  BOOST_CHECK_THROW(q ^ p, runtime_error);
  BOOST_CHECK_THROW(r ^ p, runtime_error);
}

BOOST_AUTO_TEST_CASE(uep_to_packet) {
  buffer_type b1(10, 0x11);
  uep_packet up(b1);
  BOOST_CHECK_EQUAL(up.sequence_number(), 0);

  packet p = up.to_packet();
  BOOST_CHECK_EQUAL(p.size(), b1.size() + sizeof(uep_packet::seqno_type));
  BOOST_CHECK(equal(p.buffer().begin(),
		    p.buffer().begin() + sizeof(uep_packet::seqno_type),
		    "\x00\x00\x00\x00"));
  BOOST_CHECK(equal(p.buffer().begin() + sizeof(uep_packet::seqno_type),
		    p.buffer().end(),
		    b1.cbegin()));

  up.sequence_number(0xff);
  BOOST_CHECK_EQUAL(up.sequence_number(), 0xff);
  BOOST_CHECK_EQUAL(boost::numeric_cast<uep_packet::seqno_type>(up.sequence_number()), 0xff);
  p = up.to_packet();
  BOOST_CHECK_EQUAL(p.size(), b1.size() + sizeof(uep_packet::seqno_type));
  BOOST_CHECK(equal(p.buffer().begin(),
		    p.buffer().begin() + sizeof(uep_packet::seqno_type),
		    "\x00\x00\x00\xff"));
  BOOST_CHECK(equal(p.buffer().begin() + sizeof(uep_packet::seqno_type),
		    p.buffer().end(),
		    b1.cbegin()));

  up.sequence_number(0xffff00ff);
  BOOST_CHECK_EQUAL(up.sequence_number(), 0xffff00ff);
  BOOST_CHECK_EQUAL(boost::numeric_cast<uep_packet::seqno_type>(up.sequence_number()), 0xffff00ff);
  p = up.to_packet();
  BOOST_CHECK_EQUAL(p.size(), b1.size() + sizeof(uep_packet::seqno_type));
  BOOST_CHECK(equal(p.buffer().begin(),
		    p.buffer().begin() + sizeof(uep_packet::seqno_type),
		    "\xff\xff\x00\xff"));
  BOOST_CHECK(equal(p.buffer().begin() + sizeof(uep_packet::seqno_type),
		    p.buffer().end(),
		    b1.cbegin()));

  up.sequence_number(0xffffffff);
  BOOST_CHECK_EQUAL(up.sequence_number(), 0xffffffff);
  BOOST_CHECK_EQUAL(boost::numeric_cast<uep_packet::seqno_type>(up.sequence_number()), 0xffffffff);
  p = up.to_packet();
  BOOST_CHECK_EQUAL(p.size(), b1.size() + sizeof(uep_packet::seqno_type));
  BOOST_CHECK(equal(p.buffer().begin(),
		    p.buffer().begin() + sizeof(uep_packet::seqno_type),
		    "\xff\xff\xff\xff"));
  BOOST_CHECK(equal(p.buffer().begin() + sizeof(uep_packet::seqno_type),
		    p.buffer().end(),
		    b1.cbegin()));
}

BOOST_AUTO_TEST_CASE(uep_from_packet) {
  const char raw[] = "\xff\x00\x00\x00\x11\x22\x33\x44\x55";
  const char exp_data[] = "\x11\x22\x33\x44\x55";
  const size_t exp_sn = 0xff000000;

  packet p;
  p.buffer().resize(5+4);
  p.buffer().assign(raw, raw+5+4);

  uep_packet up = uep_packet::from_packet(p);
  BOOST_CHECK_EQUAL(up.priority(), 0);
  BOOST_CHECK_EQUAL(up.sequence_number(), exp_sn);
  BOOST_CHECK(equal(up.buffer().begin(), up.buffer().end(),
		    exp_data));
}

BOOST_AUTO_TEST_CASE(uep_to_from_packet) {
  uep_packet up;
  up.buffer() = {0x11, 0x22, 0x33, 0x44, 0x55};
  up.priority(123);
  up.sequence_number(12345678);

  packet p = up.to_packet();

  uep_packet up2 = uep_packet::from_packet(p);

  BOOST_CHECK_EQUAL(up2.sequence_number(), up.sequence_number());
  BOOST_CHECK(up2.buffer() == up.buffer());
}

BOOST_AUTO_TEST_CASE(uep_to_from_fountain_packet) {
  uep_packet up;
  up.buffer() = {0x11, 0x22, 0x33, 0x44, 0x55};
  up.priority(123);
  up.sequence_number(12345678);

  fountain_packet p = up.to_fountain_packet();

  uep_packet up2 = uep_packet::from_fountain_packet(p);

  BOOST_CHECK_EQUAL(up2.sequence_number(), up.sequence_number());
  BOOST_CHECK_EQUAL(up2.priority(), up.priority());
  BOOST_CHECK(up2.buffer() == up.buffer());
}

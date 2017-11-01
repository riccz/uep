#ifndef UEP_UTILS_HPP
#define UEP_UTILS_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <iterator>
#include <random>

#include <boost/iterator/filter_iterator.hpp>

#include "base_types.hpp"

namespace uep { namespace utils {

/** Predicate that just converts the input to bool. */
template <class T>
struct to_bool {
  bool operator()(const T &x) const {
    return static_cast<bool>(x);
  }
};

/** Iterator adaptor that skips over the values that convert to false. */
template <class BaseIter>
using skip_false_iterator = boost::filter_iterator<
  to_bool<typename std::iterator_traits<BaseIter>::value_type>,
  BaseIter>;

/** Type traits that provide the interface for the symbols used in the
 *  mp_context class.
 */
template <class Symbol>
class symbol_traits {
private:
  static void swap_syms(Symbol &lhs, Symbol &rhs) {
    using std::swap;
    swap(lhs, rhs);
  }

  static_assert(std::is_move_constructible<Symbol>::value,
		"Symbol must be move-constructible");
  static_assert(std::is_move_assignable<Symbol>::value,
		"Symbol must be move-assignable");

public:
  /** Create an empty symbol. */
  static Symbol create_empty() {
    return Symbol();
  }

  /** True when a symbol is empty. */
  static bool is_empty(const Symbol &s) {
    return !s;
  }

  /** Perform the in-place XOR between two symbols. */
  static void inplace_xor(Symbol &lhs, const Symbol &rhs) {
    lhs ^= rhs;
  }

  /** Swap two symbols. */
  static void swap(Symbol &lhs, Symbol &rhs) {
    // Avoid using the function with the same name
    swap_syms(lhs,rhs);
  }
};

/** Integer hashing function. */
template <class T>
struct knuth_mul_hasher {
  std::size_t operator()(const T *k) const {
    std::ptrdiff_t v = (std::ptrdiff_t)k * UINT32_C(2654435761);
    // Right-shift v by the size difference between a pointer and a
    // 32-bit integer (0 for x86, 32 for x64)
    v >>= ((sizeof(std::ptrdiff_t) - sizeof(std::uint32_t)) * 8);
    return (std::size_t)(v & UINT32_MAX);
  }
};

/** Drop empty lines from an input stream. Return the number of
 *  dropped lines.
 */
inline std::size_t skip_empty(std::istream &istr) {
  std::size_t empty_count = 0;
  std::string nextline;
  std::istream::pos_type oldpos;
  while (nextline.empty() && istr) {
    oldpos = istr.tellg();
    std::getline(istr, nextline);
    ++empty_count;
  }
  if (!nextline.empty()) {
    istr.seekg(oldpos);
    --empty_count;
  }
  return empty_count;
}

}}

template<typename Iter>
void write_iterable(std::ostream &out, Iter begin, Iter end) {
  std::string sep;
  out << '[';
  for (; begin != end; ++begin) {
    out << sep;
    out << *begin;
    sep = ", ";
  }
  out << ']';
}

template<typename T>
std::vector<T> read_list(std::istream &in) {
  std::vector<T> v;
  char c = 0;
  T e;
  in >> c;
  in >> std::ws;
  if (c != '[') throw std::runtime_error("List parsing failed");
  while (in.peek() != ']') {
    in >> e;
    if (!in) throw std::runtime_error("List parsing failed");
    v.push_back(e);
    in >> std::ws;
    if (in.peek() == ',') {
      in >> c;
      in >> std::ws;
    }
  }
  in >> c;
  if (c != ']') throw std::runtime_error("List parsing failed");
  return v;
}

/** Generate the random states of a 2-state Markov chain. */
class markov2_distribution {
public:
  explicit markov2_distribution(double p) : markov2_distribution(p, 1-p) {
  }

  explicit markov2_distribution(double p01, double p10) :
    tx_01(p01), tx_10(p10) {
    std::bernoulli_distribution initial(p01 / (p01 + p10));
    rng.seed(std::random_device{}());
    state = initial(rng) ? 1 : 0;
  }

  void set_tx_probs(double p01, double p10) {
    using dist_t = decltype(tx_01);
    tx_01 = dist_t(p01);
    tx_10 = dist_t(p10);
  }

  f_uint operator()() {
    switch (state) {
    case 0:
      state = tx_01(rng) ? 1 : 0;
      break;
    case 1:
      state = tx_10(rng) ? 0 : 1;
      break;
    }
    return state;
  }

  double stationary_p0() const {
    return tx_10.p() / (tx_10.p() + tx_01.p());
  }

  double stationary_p1() const {
    return tx_01.p() / (tx_10.p() + tx_01.p());
  }

  double p_01() const {
    return tx_01.p();
  }

  double p_10() const {
    return tx_10.p();
  }

private:
  f_uint state;
  std::mt19937 rng;
  std::bernoulli_distribution tx_01;
  std::bernoulli_distribution tx_10;
};

namespace std {

/** Write a text representation of a vector. */
template<typename T>
ostream &operator<<(ostream &out, const vector<T> &vec) {
  write_iterable(out, vec.begin(), vec.end());
  return out;
}

/** Read a list from a stream. */
template<typename T>
istream &operator>>(istream &in, vector<T> &vec) {
  vec = read_list<T>(in);
  return in;
}

}

#endif

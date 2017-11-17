#include "rng.hpp"

#include <cmath>
#include <set>

using namespace std;
using namespace std::placeholders;

degree_distribution::degree_distribution(std::size_t K, const pmd_t &pmd) :
  K_(K), pmd_(pmd) {
  vector<double> weights;
  for (size_t d = 1; d <= K; ++d) {
    weights.push_back(pmd_(d));
  }

  typedef decltype(distrib)::param_type p_type;
  p_type p(weights.begin(), weights.end());
  distrib.param(p);
}

std::size_t degree_distribution::K() const {
  return K_;
}

degree_distribution::pmd_t degree_distribution::pmd() const {
  return pmd_;
}

soliton_distribution::soliton_distribution(std::size_t input_pkt_count) :
  degree_distribution(input_pkt_count, bind(soliton_pmd, input_pkt_count, _1)) {
}

double soliton_distribution::soliton_pmd(std::size_t K, std::size_t d) {
  if (d == 1) return 1/(double)K;
  else if (d >= 2 && d <= K) return 1/((double)d*(d-1));
  else return 0;
}

double robust_soliton_distribution::S(std::size_t K, double c, double delta) {
  return c * log(K/delta) * sqrt(K);
}

double robust_soliton_distribution::tau(std::size_t K_S, double S_delta, std::size_t i) {
  if (i >= 1 && i <= K_S - 1)
    return 1/((double)K_S * i);
  else if (i == K_S)
    return log(S_delta) / (double) K_S;
  else
    return 0;
}

double robust_soliton_distribution::beta(std::size_t K, std::size_t K_S, double S_delta) {
  double sum = 0;
  for (size_t i = 1; i <= K; ++i) {
    sum += soliton_distribution::soliton_pmd(K,i) + tau(K_S, S_delta, i);
  }
  return sum;
}

double robust_soliton_distribution::robust_pmd(std::size_t K, double c, double delta,
					       std::size_t d) {
  if (d >= 1 && d <= K) {
    double S_ = S(K, c, delta);
    size_t K_S = lround(K/S_);
    double S_delta = S_/delta;
    double beta_ = beta(K, K_S, S_delta);
    return (soliton_distribution::soliton_pmd(K,d) +
	    tau(K_S,S_delta,d)) / beta_;
  }
  else return 0;
}

robust_soliton_distribution::robust_soliton_distribution(std::size_t input_pkt_count,
							 double c,
							 double delta) :
  degree_distribution(input_pkt_count,
		      bind(robust_pmd, input_pkt_count, c, delta, _1)),
  c_(c), delta_(delta) {
}

double robust_soliton_distribution::c() const {
  return c_;
}

double robust_soliton_distribution::delta() const {
  return delta_;
}

double robust_soliton_distribution::beta() const {
  double S_ = S(K(), c_, delta_);
  size_t K_S = lround(K()/S_);
  double S_delta = S_/delta_;
  return beta(K(), K_S, S_delta);
}

base_row_generator::base_row_generator(rng_type::result_type seed) :
  rng(seed), sel_count(0), last_seed(seed) {
}

lt_row_generator::lt_row_generator(const degree_distribution &deg) :
  lt_row_generator(deg, rng_type::default_seed) {
}

lt_row_generator::lt_row_generator(const degree_distribution &deg,
				   rng_type::result_type seed) :
  base_row_generator(seed),
  degree_distr(deg),
  packet_distr(0, deg.K()-1) {
}

lt_row_generator::row_type lt_row_generator::next_row() {
  size_t degree = degree_distr(rng);
  row_type s;
  s.reserve(degree);
  for (size_t i = 0; i < degree; ++i) {
    size_t si;
    do {
      si = packet_distr(rng);
    }	while (find(s.begin(), s.end(), si) != s.end());
    s.push_back(si);
  }
  ++sel_count;
  return s;
}

std::size_t base_row_generator::generated_rows() const {
  return sel_count;
}

std::size_t lt_row_generator::K() const {
  return degree_distr.K();
}

base_row_generator::rng_type::result_type base_row_generator::seed() const {
  return last_seed;
}

void base_row_generator::reset(rng_type::result_type seed) {
  rng.seed(seed);
  sel_count = 0;
  last_seed = seed;
}

lt_row_generator make_robust_lt_row_generator(std::size_t K, double c, double delta) {
  return lt_row_generator(robust_soliton_distribution(K, c, delta));
}

namespace uep {

base_row_generator::row_type uep_row_generator::next_row() {
  row_type s_mapped;
  row_type s;

  std::size_t degree;
  do {
    degree = _deg_dist(rng);
  } while (degree > _k_in);

  std::set<std::size_t> row;
  while (row.size() < degree) {
    std::size_t index = _pos_map(_p_dist(rng));
    row.insert(index);
  }

  ++sel_count;
  return row_type(row.cbegin(), row.cend());
}

std::size_t uep_row_generator::K() const {
  return _k_in;
}

std::size_t uep_row_generator::K_in() const {
  return _k_in;
}

std::size_t uep_row_generator::K_out() const {
  return _k_out;
}

const std::vector<std::size_t> &uep_row_generator::Ks() const {
  return _ks;
}

const std::vector<std::size_t> &uep_row_generator::RFs() const {
  return _rfs;
}

std::size_t uep_row_generator::EF() const {
  return _ef;
}

double uep_row_generator::c() const {
  return _c;
}

double uep_row_generator::delta() const {
  return _delta;
}

}

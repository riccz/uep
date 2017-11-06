%module uep_fast_run

%{
#include "uep_fast_run.hpp"
#include "log.hpp"
%}

%init %{
  uep::log::init();
  auto warn_filter = boost::log::expressions::attr<
    uep::log::severity_level>("Severity") >= uep::log::warning;
  boost::log::core::get()->set_filter(warn_filter);
%}

%include "std_string.i"
%include "std_vector.i"
%include "typemaps.i"

%inline %{
  typedef long unsigned int size_t;
%}

%apply int { size_t };

%copyctor simulation_params;
%copyctor simulation_results;
%include "uep_fast_run.hpp"

namespace std {
  %template(DoubleVector) vector<double>;
  %template(SizeTVector) vector<size_t>;
}

%extend simulation_params {
%pythoncode {
    def __getstate__(self):
      state = dict()
      state['Ks'] = [k for k in self.Ks]
      state['RFs'] = [rf for rf in self.RFs]
      state['EF'] = self.EF
      state['c'] = self.c
      state['delta'] = self.delta
      state['L'] = self.L
      state['nblocks'] = self.nblocks
      state['overhead'] = self.overhead
      state['chan_pGB'] = self.chan_pGB
      state['chan_pBG'] = self.chan_pBG
      state['nCycles'] = self.nCycles
      return state

    def __setstate__(self, state):
      self.__init__()
      self.Ks[:] = state['Ks']
      self.RFs[:] = state['RFs']
      self.EF = state['EF']
      self.c = state['c']
      self.delta = state['delta']
      self.L = state['L']
      self.nblocks = state['nblocks']
      self.overhead = state['overhead']
      self.chan_pGB = state['chan_pGB']
      self.chan_pBG = state['chan_pBG']
      self.nCycles = state.get('nCycles', 1)
}
}

%extend simulation_results {
%pythoncode {
    def __getstate__(self):
      state = dict()
      state['avg_pers'] = [p for p in self.avg_pers]
      state['rec_counts'] = [c for c in self.rec_counts]
      state['dropped_count'] = self.dropped_count
      state['avg_enc_time'] = self.avg_enc_time
      return state

    def __setstate__(self, state):
      self.__init__()
      self.avg_pers[:] = state['avg_pers']
      self.rec_counts[:] = state['rec_counts']
      self.avg_enc_time = state['avg_enc_time']
}
}

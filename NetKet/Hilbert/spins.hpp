// Copyright 2018 The Simons Foundation, Inc. - All Rights Reserved.
// Copyright 2018 Tom Westerhout
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef NETKET_SPIN_HPP
#define NETKET_SPIN_HPP

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include <Eigen/Dense>
#include "Graph/abstract_graph.hpp"
#include "Utils/json_utils.hpp"
#include "Utils/python_helper.hpp"
#include "Utils/random_utils.hpp"
#include "abstract_hilbert.hpp"

namespace netket {

/**
  Hilbert space for integer or half-integer spins.
  Notice that here integer values are always used to represent the local quantum
  numbers, such that for example if total spin is S=3/2, the allowed quantum
  numbers are -3,-1,1,3, and if S=1 we have -2,0,2.
*/

class Spin : public AbstractHilbert {
  const AbstractGraph &graph_;

  double S_;
  double totalS_;
  bool constraintSz_;

  std::vector<double> local_;

  int nstates_;

  int nspins_;

 public:
  template <class Ptype>
  explicit Spin(const AbstractGraph &graph, const Ptype &pars) : graph_(graph) {
    const int nspins = graph.Nsites();
    const double S = FieldVal<double>(pars, "S", "Hilbert");

    Init(nspins, S);

    if (FieldExists(pars, "TotalSz")) {
      auto totalSz = FieldVal<double>(pars, "TotalSz");
      SetConstraint(totalSz);
    } else {
      constraintSz_ = false;
    }
  }

  void Init(int nspins, double S) {
    S_ = S;
    nspins_ = nspins;

    if (S <= 0) {
      throw InvalidInputError("Invalid spin value");
    }

    if (std::floor(2. * S) != 2. * S) {
      throw InvalidInputError("Spin value is neither integer nor half integer");
    }

    nstates_ = std::floor(2. * S) + 1;

    local_.resize(nstates_);

    int sp = -std::floor(2. * S);
    for (int i = 0; i < nstates_; i++) {
      local_[i] = sp;
      sp += 2;
    }
  }

  void SetConstraint(double totalS) {
    constraintSz_ = true;
    totalS_ = totalS;
  }

  bool IsDiscrete() const override { return true; }

  int LocalSize() const override { return nstates_; }

  int Size() const override { return nspins_; }

  std::vector<double> LocalStates() const override { return local_; }

  void RandomVals(Eigen::VectorXd &state,
                  netket::default_random_engine &rgen) const override {
    std::uniform_int_distribution<int> distribution(0, nstates_ - 1);

    assert(state.size() == nspins_);

    if (!constraintSz_) {
      // unconstrained random
      for (int i = 0; i < state.size(); i++) {
        state(i) = 2. * (distribution(rgen) - S_);
      }
    } else if (S_ == 0.5) {
      using std::begin;
      using std::end;
      // Magnetisation as a count
      auto const m = static_cast<int>(2 * totalS_);
      if (std::abs(m) > nspins_) {
        throw InvalidInputError(
            "Cannot fix the total magnetization: 2|M| cannot "
            "exceed Nspins.");
      }
      if ((nspins_ + m) % 2 != 0) {
        throw InvalidInputError(
            "Cannot fix the total magnetization: Nspins + "
            "totalSz must be even.");
      }
      auto const nup = (nspins_ + m) / 2;
      auto const ndown = (nspins_ - m) / 2;
      std::fill_n(state.data(), nup, 1.0);
      std::fill_n(state.data() + nup, ndown, -1.0);
      std::shuffle(state.data(), state.data() + nspins_, rgen);
      return;
    } else {
      std::vector<int> sites;
      for (int i = 0; i < nspins_; ++i) sites.push_back(i);

      state.setConstant(-2 * S_);
      int ss = nspins_;

      for (int i = 0; i < S_ * nspins_ + totalS_; ++i) {
        std::uniform_int_distribution<int> distribution_ss(0, ss - 1);
        int s = distribution_ss(rgen);
        state(sites[s]) += 2;
        if (state(sites[s]) > 2 * S_ - 1) {
          sites.erase(sites.begin() + s);
          ss -= 1;
        }
      }
    }
  }

  void UpdateConf(Eigen::Ref<Eigen::VectorXd> v,
                  const std::vector<int> &tochange,
                  const std::vector<double> &newconf) const override {
    assert(v.size() == nspins_);

    int i = 0;
    for (auto sf : tochange) {
      v(sf) = newconf[i];
      i++;
    }
  }

  const AbstractGraph &GetGraph() const override { return graph_; }
};

}  // namespace netket
#endif

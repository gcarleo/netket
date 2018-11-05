// Copyright 2018 The Simons Foundation, Inc. - All Rights Reserved.
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

#ifndef NETKET_BOSONS_HPP
#define NETKET_BOSONS_HPP

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
  Hilbert space for integer or bosons.
  The hilbert space is truncated to some maximum occupation number.
*/

class Boson : public AbstractHilbert {
  const AbstractGraph &graph_;

  int nsites_;

  std::vector<double> local_;

  // total number of bosons
  // if constraint is activated
  int nbosons_;

  bool constraintN_;

  // maximum local occupation number
  int nmax_;

  int nstates_;

 public:
  template <class Ptype>
  explicit Boson(const AbstractGraph &graph, const Ptype &pars)
      : graph_(graph) {
    nsites_ = graph.Nsites();

    CheckFieldExists(pars, "Nmax", "Hilbert");

    nmax_ = FieldVal<int>(pars, "Nmax");

    Init();

    if (FieldExists(pars, "Nbosons")) {
      auto nbosons = FieldVal<int>(pars, "Nbosons");
      SetNbosons(nbosons);
    } else {
      constraintN_ = false;
    }
  }

  void Init() {
    if (nsites_ <= 0) {
      throw InvalidInputError("Invalid number of sites");
    }

    if (nmax_ <= 0) {
      throw InvalidInputError("Invalid maximum occupation number");
    }

    nstates_ = nmax_ + 1;

    local_.resize(nstates_);

    for (int i = 0; i < nstates_; i++) {
      local_[i] = i;
    }
  }

  void SetNbosons(int nbosons) {
    constraintN_ = true;
    nbosons_ = nbosons;

    if (nbosons_ > nsites_ * nmax_) {
      throw InvalidInputError("Cannot set the desired number of bosons");
    }
  }

  bool IsDiscrete() const override { return true; }

  int LocalSize() const override { return nstates_; }

  int Size() const override { return nsites_; }

  std::vector<double> LocalStates() const override { return local_; }

  void RandomVals(Eigen::VectorXd &state,
                  netket::default_random_engine &rgen) const override {
    assert(state.size() == nsites_);

    if (!constraintN_) {
      std::uniform_int_distribution<int> distribution(0, nstates_ - 1);
      // unconstrained random
      for (int i = 0; i < state.size(); i++) {
        state(i) = distribution(rgen);
      }
    } else {
      state.setZero();

      std::uniform_int_distribution<int> distribution(0, nsites_ - 1);
      for (int i = 0; i < nbosons_; i++) {
        int rsite = distribution(rgen);

        while (state(rsite) >= nmax_) {
          rsite = distribution(rgen);
        }

        state(rsite) += 1;
      }
    }
  }

  bool CheckConstraint(const Eigen::VectorXd &v) const {
    int tot = 0;
    for (int i = 0; i < v.size(); i++) {
      tot += int(v(i));
    }

    return tot == nbosons_;
  }

  void UpdateConf(Eigen::Ref<Eigen::VectorXd> v,
                  const std::vector<int> &tochange,
                  const std::vector<double> &newconf) const override {
    assert(v.size() == nsites_);

    int i = 0;
    for (auto sf : tochange) {
      v(sf) = newconf[i];
      i++;
      assert(v(sf) <= nmax_);
    }

    if (constraintN_) {
      assert(CheckConstraint(v));
    }
  }

  const AbstractGraph &GetGraph() const override { return graph_; }
};

}  // namespace netket
#endif

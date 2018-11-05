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

#include <cfloat>
#include <fstream>
#include <iostream>
#include <set>
#include <vector>
#include "Hilbert/hilbert.hpp"
#include "Utils/json_utils.hpp"
#include "catch.hpp"
#include "hilbert_input_tests.hpp"
#include "netket.hpp"

netket::Hilbert MakeHilbert(const netket::json& pars) {
  if (netket::FieldExists(pars, "Graph")) {
    auto graph = netket::make_graph(pars);
    return netket::Hilbert(*graph, pars);
  } else {
    return netket::Hilbert(pars);
  }
}

TEST_CASE("hilbert has consistent sizes and definitions", "[hilbert]") {
  auto input_tests = GetHilbertInputs();
  std::size_t ntests = input_tests.size();

  for (std::size_t i = 0; i < ntests; i++) {
    std::string parname = "Hilbert";

    REQUIRE(netket::FieldExists(input_tests[i], "Hilbert"));
    std::string name = input_tests[i][parname].dump();

    SECTION("Hilbert test (" + std::to_string(i) + ") (" + std::to_string(i) +
            ") on " + name) {
      netket::Hilbert hilbert = MakeHilbert(input_tests[i]);

      REQUIRE(hilbert.Size() > 0);
      REQUIRE(hilbert.LocalSize() > 0);

      if (hilbert.IsDiscrete()) {
        const auto lstate = hilbert.LocalStates();

        REQUIRE(int(lstate.size()) == hilbert.LocalSize());

        for (std::size_t k = 0; k < lstate.size(); k++) {
          REQUIRE(std::isfinite(lstate[k]));
        }
      }
    }
  }
}

TEST_CASE("hilbert generates consistent random states", "[hilbert]") {
  auto input_tests = GetHilbertInputs();
  std::size_t ntests = input_tests.size();

  for (std::size_t i = 0; i < ntests; i++) {
    std::string parname = "Hilbert";
    REQUIRE(netket::FieldExists(input_tests[i], "Hilbert"));
    std::string name = input_tests[i][parname].dump();

    SECTION("Hilbert test (" + std::to_string(i) + ") on " + name) {
      netket::Hilbert hilbert = MakeHilbert(input_tests[i]);

      REQUIRE(hilbert.Size() > 0);
      REQUIRE(hilbert.LocalSize() > 0);

      if (hilbert.IsDiscrete()) {
        netket::default_random_engine rgen(3421);
        Eigen::VectorXd rstate(hilbert.Size());

        const auto lstate = hilbert.LocalStates();
        REQUIRE(int(lstate.size()) == hilbert.LocalSize());

        std::set<int> lset(lstate.begin(), lstate.end());

        for (int it = 0; it < 100; it++) {
          hilbert.RandomVals(rstate, rgen);
          for (int k = 0; k < rstate.size(); k++) {
            REQUIRE(lset.count(rstate(k)) > 0);
          }
        }
      }
    }
  }
}

TEST_CASE("hilbert index generates consistent mappings", "[hilbert]") {
  auto input_tests = GetHilbertInputs();
  std::size_t ntests = input_tests.size();

  for (std::size_t i = 0; i < ntests; i++) {
    std::string parname = "Hilbert";
    REQUIRE(netket::FieldExists(input_tests[i], "Hilbert"));
    std::string name = input_tests[i][parname].dump();

    SECTION("Hilbert test (" + std::to_string(i) + ") on " + name) {
      netket::Hilbert hilbert = MakeHilbert(input_tests[i]);

      REQUIRE(hilbert.Size() > 0);
      REQUIRE(hilbert.LocalSize() > 0);

      // Only do the test for small hilbert spaces
      if (hilbert.Size() * std::log(hilbert.LocalSize()) <
          std::log(netket::HilbertIndex::MaxStates)) {
        netket::HilbertIndex hilb_index(hilbert);

        for (std::size_t k = 0; k < hilb_index.NStates(); k++) {
          const auto state = hilb_index.NumberToState(k);

          REQUIRE(hilb_index.StateToNumber(state) == k);
        }
      }
    }
  }
}
